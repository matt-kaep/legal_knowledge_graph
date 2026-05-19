"""Streaming driver + CLI. Spec §7/§8. Parquet & client injected for tests.
Bounded client-side concurrency: vLLM continuous-batches server-side, so N
in-flight requests is the throughput lever. analyze_record is pure; ALL result
handling (quarantine, buckets, flush, circuit breaker) runs in this coordinator
thread only -> shard atomicity (§9) preserved. concurrency=1 == sequential."""
import argparse
import itertools
from pathlib import Path

from analyzer.jp_analyzer import analyze_record, CircuitBreaker, RunConfig, RetryableError
from ledger import (atomic_write_shard, derive_done_ids, append_jsonl,
                    load_quarantine)

PARQUET = "05-Technique/benchmark/baseline_b2/jp_index.parquet"

def _iter_parquet(path, limit=None, juris=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["id", "number", "juris", "text"])
        for r in tbl.to_pylist():
            if juris and r["juris"] != juris:
                continue
            yield r
            seen += 1
            if limit and seen >= limit:
                return

def run(rows, client, cfg: RunConfig, out_root: Path, shard_size=500,
        concurrency: int = 16):
    """Stream rows, dispatch analyze_record across a bounded thread pool.
    analyze_record is a pure per-record function -> thread-safe. ALL result
    handling runs in this coordinator thread only -> shard atomicity preserved.
    concurrency=1 == sequential."""
    import concurrent.futures as cf
    out_root = Path(out_root)
    done = derive_done_ids(out_root)
    quarantine = load_quarantine(out_root / "_quarantine.jsonl")
    cb = CircuitBreaker()
    buckets: dict[str, list] = {}
    counters: dict[str, int] = {}

    def flush(juris):
        recs = buckets.get(juris)
        if not recs:
            return
        n = counters.get(juris, 0)
        atomic_write_shard(out_root / juris / f"part-{n:05d}.jsonl",
                            [{k: v for k, v in r.items() if k != "_anomalies"}
                             for r in recs])
        for r in recs:
            for a in r.get("_anomalies", []) or []:
                append_jsonl(out_root / "_themes_anomalies.jsonl",
                             {"id": r["id"], **a})
            append_jsonl(out_root / "_metrics.jsonl",
                         {k: r.get(k) for k in ("id", "status", "error_class",
                          "attempt_count", "tokens_in", "tokens_out",
                          "duration_ms", "model")})
        counters[juris] = n + 1
        buckets[juris] = []

    def handle(row, result_exc):
        """Coordinator-thread post-processing of one finished record."""
        rec, exc = result_exc
        if isinstance(exc, RetryableError):
            attempts = quarantine.get(row["id"], 0) + 1
            cb.record(ok=False)
            if attempts >= cfg.max_attempts:
                rec = {"id": row["id"], "number": row.get("number", ""),
                       "juris": row["juris"], "status": "failed_terminal",
                       "failed": True, "error_class": "retryable",
                       "error_message": str(exc), "attempt_count": attempts}
            else:
                append_jsonl(out_root / "_quarantine.jsonl",
                             {"id": row["id"], "attempt_count": attempts,
                              "error_message": str(exc)})
                return cb.tripped()
        else:
            cb.record(ok=(rec["status"] == "ok"))
        buckets.setdefault(row["juris"], []).append(rec)
        if len(buckets[row["juris"]]) >= shard_size:
            flush(row["juris"])
        return cb.tripped()

    def work(row):
        try:
            return analyze_record(row, client, cfg), None
        except RetryableError as exc:
            return None, exc

    pending = (r for r in rows if r["id"] not in done)
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        inflight: dict = {}
        exhausted = False
        while True:
            while not exhausted and len(inflight) < max(1, concurrency):
                nxt = next(pending, None)
                if nxt is None:
                    exhausted = True
                    break
                inflight[pool.submit(work, nxt)] = nxt
            if not inflight:
                break
            done_fut, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in done_fut:
                row = inflight.pop(fut)
                if handle(row, fut.result()):
                    for j in list(buckets):
                        flush(j)
                    raise RuntimeError(
                        "circuit breaker tripped — pausing run "
                        "(retryable failure rate too high; infra degraded)")
    for j in list(buckets):
        flush(j)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=int, default=0,
                   help="N stratified records (CC/CA/TJ) smoke run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--juris", choices=["CC", "CA", "TJ"], default=None)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--model", default="gemma4-31B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--out", default="outputs/step1")
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--concurrency", type=int, default=16,
                   help="in-flight vLLM requests (1 = sequential)")
    args = p.parse_args(argv)

    from openai import OpenAI
    from transformers import AutoTokenizer
    from budget import compute_threshold, verify_max_model_len
    from prompts.step1.build_prompt import build_system_prompt

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    verify_max_model_len(client, args.max_model_len)
    tok = AutoTokenizer.from_pretrained(args.model) if False else None  # set real id at deploy
    overhead = max(len(build_system_prompt(j)[0]) // 3 for j in ("CC", "CA", "TJ"))
    threshold = compute_threshold(args.max_model_len, overhead, 4000)
    cfg = RunConfig(model=args.model, threshold=threshold, tokenizer=tok)

    if args.pilot:
        per = max(1, args.pilot // 3)
        rows = itertools.chain(
            _iter_parquet(args.parquet, per, "CC"),
            _iter_parquet(args.parquet, per, "CA"),
            _iter_parquet(args.parquet, per, "TJ"))
    else:
        rows = _iter_parquet(args.parquet, args.limit, args.juris)
    run(rows, client, cfg, Path(args.out), concurrency=args.concurrency)

if __name__ == "__main__":
    main()
