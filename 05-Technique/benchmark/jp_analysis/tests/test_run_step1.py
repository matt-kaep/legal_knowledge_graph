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
