"""Streaming driver + CLI. Spec §7/§8. Parquet & client injected for tests.
Bounded client-side concurrency: vLLM continuous-batches server-side, so N
in-flight requests is the throughput lever. analyze_record is pure; ALL result
handling (quarantine, buckets, flush, circuit breaker) runs in this coordinator
thread only -> shard atomicity (§9) preserved. concurrency=1 == sequential."""
import argparse
import itertools
import re
from pathlib import Path

from analyzer.jp_analyzer import (analyze_record, CircuitBreaker, RunConfig,
                                  RetryableError, build_terminal_record)
from ledger import (atomic_write_shard, derive_done_ids, append_jsonl,
                    load_quarantine)

PARQUET = "05-Technique/benchmark/baseline_b2/jp_index.parquet"

# Filename -> juris (Judilibre dumps). Matches baseline_b2/build_jp_index.py.
_JURIS_FROM_FILENAME = {
    # cluster naming (extensionless, spaces)
    "cour de cassation": "CC", "cours d'appel": "CA", "cour d'appel": "CA",
    "tribunal judiciaire": "TJ",
    # local naming (cc.jsonl / ca.jsonl / tj.jsonl)
    "cc": "CC", "ca": "CA", "tj": "TJ",
}

def _juris_from_filename(p):
    name = p.name.lower(); stem = p.stem.lower()
    if name in _JURIS_FROM_FILENAME: return _JURIS_FROM_FILENAME[name]
    if stem in _JURIS_FROM_FILENAME: return _JURIS_FROM_FILENAME[stem]
    if "cassation" in name: return "CC"
    if "appel" in name: return "CA"
    if "tribunal" in name or "judiciaire" in name: return "TJ"
    raise ValueError(f"cannot derive juris from filename {p.name!r}")

def _best_text(d, juris):
    """Same heuristic as baseline_b2/build_jp_index.py:31."""
    if juris == "CC":
        s = (d.get("summary") or "").strip()
        if len(s) > 100:
            return s
    return (d.get("text") or "").strip()

def _iter_jsonl_file(path, juris_filter):
    import json
    from pathlib import Path as _P
    fp = _P(path); juris = _juris_from_filename(fp)
    if juris_filter and juris_filter != juris:
        return
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = d.get("id")
            if not uid:
                continue
            # Judilibre dump uses `number` (singular) on CC and `numbers` (array)
            # on CA/TJ. Take whichever exists; fall back to empty string.
            num = d.get("number") or ""
            if not num and isinstance(d.get("numbers"), list) and d["numbers"]:
                num = d["numbers"][0] or ""
            yield {"id": uid, "number": num, "juris": juris,
                   "text": _best_text(d, juris)}

def _iter_parquet_file(path, juris_filter):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["id", "number", "juris", "text"])
        for r in tbl.to_pylist():
            if juris_filter and r["juris"] != juris_filter:
                continue
            yield r

def _iter_corpus(path, limit=None, juris=None):
    """Stream rows from a parquet/JSONL FILE or a DIR containing either.
    JSONL juris is derived from the filename (matches build_jp_index.py)."""
    from pathlib import Path as _P
    p = _P(path)
    if p.is_dir():
        files = sorted({*p.glob("*.parquet"), *p.glob("*.jsonl"),
                        *(f for f in p.iterdir()
                          if f.is_file() and f.suffix == ""
                          and not f.name.startswith("."))})
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"no corpus files at {path}")
    seen = 0
    for fpath in files:
        # Parquet magic = last 4 bytes "PAR1". JSONL = anything else.
        is_parquet = False
        try:
            with open(fpath, "rb") as fh:
                fh.seek(-4, 2)
                is_parquet = fh.read(4) == b"PAR1"
        except OSError:
            is_parquet = fpath.suffix == ".parquet"
        it = _iter_parquet_file(fpath, juris) if is_parquet \
            else _iter_jsonl_file(fpath, juris)
        for r in it:
            yield r
            seen += 1
            if limit and seen >= limit:
                return

# Backward-compat alias for any caller that still imports _iter_parquet.
_iter_parquet = _iter_corpus

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
    # Resume: seed next shard number per juris from existing shards so a
    # resumed run never reuses part-00000 and atomically clobbers prior
    # committed records (shards = single source of truth, spec finding #3).
    for p in out_root.glob("*/part-*.jsonl"):
        m = re.search(r"part-(\d+)\.jsonl$", p.name)
        if m:
            jn = p.parent.name
            counters[jn] = max(counters.get(jn, 0), int(m.group(1)) + 1)

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
                # CRITICAL 2: exhausted retryable → FULL-key terminal record
                # via the shared factory (spec §3.1/§3.3 key set can't drift);
                # error_class stays "retryable" so it's traceable per spec §9.
                rec = build_terminal_record(
                    row["id"], row.get("number", ""), row["juris"],
                    "failed_terminal", variant=None, err=str(exc),
                    error_class="retryable", attempt_count=attempts)
            else:
                append_jsonl(out_root / "_quarantine.jsonl",
                             {"id": row["id"], "attempt_count": attempts,
                              "error_message": str(exc)})
                return cb.tripped()
        elif isinstance(exc, tuple) and exc and exc[0] == "__terminal__":
            # CRITICAL 3: a non-RetryableError escaped work() (e.g. route()
            # ValueError on a bad juris, KeyError on a malformed row). At
            # 1.12M rows ONE bad row must not abort the pool — it becomes a
            # terminal record (NOT quarantined, non-retryable) and the run
            # continues. Same factory as Critical 2 so the key set holds.
            bad = exc[1]
            cb.record(ok=False)
            rec = build_terminal_record(
                row.get("id", ""), row.get("number", ""),
                row.get("juris", ""), "failed_terminal", variant=None,
                err=f"{type(bad).__name__}: {bad}",
                error_class="terminal", attempt_count=1)
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
        except Exception as exc:  # noqa: BLE001
            # CRITICAL 3: any non-retryable exception escaping analyze_record
            # (route() ValueError, KeyError on a malformed row, a bug in
            # canonicalize_themes...) must NOT propagate via fut.result() and
            # kill the whole pool at 1.12M rows. Signal it as a non-retryable
            # terminal so handle() turns it into a terminal record.
            return None, ("__terminal__", exc)

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
            ready, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in ready:
                row = inflight.pop(fut)
                if handle(row, fut.result()):
                    for j in list(buckets):
                        flush(j)
                    # In-flight futures abandoned; pool.__exit__ waits but
                    # discards results. Their ids aren't in any shard ->
                    # retried next run. Intentional (resumable).
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
    p.add_argument("--tokenizer-id", default=None,
                   help="HF tokenizer id for exact near-threshold token "
                        "counts; None=char estimate")
    args = p.parse_args(argv)

    from openai import OpenAI
    from budget import compute_threshold, verify_max_model_len
    from prompts.step1.build_prompt import build_system_prompt

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    verify_max_model_len(client, args.max_model_len)
    tok = None
    if args.tokenizer_id:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_id)
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
