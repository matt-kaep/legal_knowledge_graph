import json
from pathlib import Path
from run_step1 import run
from analyzer.jp_analyzer import RunConfig
from tests.test_analyzer import FakeClient, _good_payload

def _rows():
    return [
        {"id": "a", "number": "1", "juris": "CC", "text": "arrêt a"},
        {"id": "b", "number": "2", "juris": "CA", "text": "arrêt b"},
        {"id": "c", "number": "3", "juris": "TJ", "text": ""},  # no_fulltext
    ]

def test_run_writes_shards_and_is_resumable(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    run(_rows(), FakeClient(_good_payload()), cfg, out_root=out)
    ids = set()
    for shard in out.glob("*/part-*.jsonl"):
        for line in shard.read_text().splitlines():
            ids.add(json.loads(line)["id"])
    assert ids == {"a", "b", "c"}
    before = sorted(p.read_text() for p in out.glob("*/part-*.jsonl"))
    run(_rows(), FakeClient(_good_payload()), cfg, out_root=out)
    after = sorted(p.read_text() for p in out.glob("*/part-*.jsonl"))
    assert before == after  # idempotent

def test_retryable_goes_to_quarantine_not_shard(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    run([{"id": "z", "number": "", "juris": "CC", "text": "a"}],
        FakeClient(TimeoutError("down")), cfg, out_root=out)
    assert not list(out.glob("*/part-*.jsonl"))            # no terminal record
    q = (out / "_quarantine.jsonl").read_text().strip()
    assert json.loads(q)["id"] == "z"

def test_concurrent_run_processes_every_id_exactly_once(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    rows = [{"id": f"r{i}", "number": str(i), "juris": "CC", "text": "x"}
            for i in range(40)]
    run(rows, FakeClient(_good_payload()), cfg, out_root=out, concurrency=8)
    ids = [json.loads(l)["id"]
           for shard in out.glob("*/part-*.jsonl")
           for l in shard.read_text().splitlines()]
    assert sorted(ids) == sorted(r["id"] for r in rows)   # no loss
    assert len(ids) == len(set(ids))                       # no duplicate

def test_resume_does_not_overwrite_prior_shards(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    r1 = [{"id": f"a{i}", "number": str(i), "juris": "CC", "text": "x"} for i in range(3)]
    r2 = [{"id": f"b{i}", "number": str(i), "juris": "CC", "text": "y"} for i in range(3)]
    run(r1, FakeClient(_good_payload()), cfg, out_root=out, shard_size=1)
    shards_after_r1 = sorted(p.name for p in (out / "CC").glob("part-*.jsonl"))
    run(r2, FakeClient(_good_payload()), cfg, out_root=out, shard_size=1)
    ids = [json.loads(l)["id"] for s in (out / "CC").glob("part-*.jsonl")
           for l in s.read_text().splitlines()]
    assert set(ids) == {f"a{i}" for i in range(3)} | {f"b{i}" for i in range(3)}
    assert len(ids) == len(set(ids))
    # the original shard filenames from run 1 still exist (not overwritten)
    assert set(shards_after_r1).issubset(p.name for p in (out / "CC").glob("part-*.jsonl"))

_FULL_KEYS = {"themes_valid", "themes_taxonomy_version", "schema_version",
              "model", "prompt_variant", "tokens_in", "tokens_out",
              "duration_ms"}

def test_exhausted_retryable_writes_full_key_terminal_record(tmp_path: Path):
    # CRITICAL 2 regression: with max_attempts=1 the FIRST retryable failure
    # exhausts attempts (attempts = quarantine.get(id,0)+1 = 1 >= 1) → a
    # terminal record MUST be written carrying the full spec §3.1/§3.3 key
    # set, status=failed_terminal, error_class=retryable (traceable, §9).
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000, max_attempts=1)
    run([{"id": "z", "number": "", "juris": "CC", "text": "a"}],
        FakeClient(TimeoutError("vllm down")), cfg, out_root=out)
    shards = list(out.glob("*/part-*.jsonl"))
    assert shards, "exhausted retryable must produce a terminal shard record"
    recs = [json.loads(l) for s in shards for l in s.read_text().splitlines()]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["id"] == "z"
    assert rec["status"] == "failed_terminal"
    assert rec["error_class"] == "retryable"
    assert rec["attempt_count"] == 1
    assert _FULL_KEYS.issubset(rec.keys()), \
        f"missing required keys: {_FULL_KEYS - rec.keys()}"

def test_non_retryable_exception_does_not_abort_run(tmp_path: Path):
    # CRITICAL 3 regression: the middle row has an invalid juris="XX" so
    # analyze_record → build_system_prompt → route raises ValueError OUTSIDE
    # analyze_record's internal try. At 1.12M rows one malformed row must NOT
    # abort the whole pool: run() must not raise, all 3 ids end up in shards,
    # the bad one terminal (error_class=terminal), the others ok.
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    rows = [
        {"id": "g1", "number": "1", "juris": "CC", "text": "bon arrêt"},
        {"id": "bad", "number": "2", "juris": "XX", "text": "arrêt malformé"},
        {"id": "g2", "number": "3", "juris": "CA", "text": "bon arrêt"},
    ]
    run(rows, FakeClient(_good_payload()), cfg, out_root=out)  # must NOT raise
    recs = {json.loads(l)["id"]: json.loads(l)
            for s in out.glob("*/part-*.jsonl")
            for l in s.read_text().splitlines()}
    assert set(recs) == {"g1", "bad", "g2"}
    assert recs["bad"]["status"] == "failed_terminal"
    assert recs["bad"]["error_class"] == "terminal"
    assert _FULL_KEYS.issubset(recs["bad"].keys())
    assert recs["g1"]["status"] == "ok"
    assert recs["g2"]["status"] == "ok"

def test_iter_corpus_reads_jsonl_with_juris_from_filename(tmp_path: Path):
    """Reproduces the build_jp_index.py mapping on a tiny synthetic dump."""
    from run_step1 import _iter_corpus
    db = tmp_path / "database-judilibre"; db.mkdir()
    # CC: summary > 100 chars -> used as text (best_text rule)
    cc = db / "Cour de cassation"
    cc.write_text(
        '{"id":"i1","number":"00-12.345","summary":"' + ("x"*150) + '","text":"raw cc"}\n'
        '{"id":"i2","numbers":["01-99.999"],"summary":"short","text":"fallback to text"}\n',
        encoding="utf-8")
    # CA: numbers (array), no summary used
    ca = db / "Cours d'appel"
    ca.write_text(
        '{"id":"i3","numbers":["95-00807"],"text":"ca text body"}\n',
        encoding="utf-8")
    # TJ: same pattern
    tj = db / "Tribunal judiciaire"
    tj.write_text(
        '{"id":"i4","numbers":["23/08541"],"text":"tj text body"}\n',
        encoding="utf-8")
    rows = list(_iter_corpus(db))
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"i1", "i2", "i3", "i4"}
    assert by_id["i1"]["juris"] == "CC" and by_id["i1"]["text"] == "x"*150
    assert by_id["i2"]["juris"] == "CC" and by_id["i2"]["text"] == "fallback to text"
    assert by_id["i2"]["number"] == "01-99.999"          # picked from numbers[]
    assert by_id["i3"]["juris"] == "CA" and by_id["i3"]["number"] == "95-00807"
    assert by_id["i4"]["juris"] == "TJ" and by_id["i4"]["text"] == "tj text body"
