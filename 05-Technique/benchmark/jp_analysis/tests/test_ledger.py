import json
from pathlib import Path
from ledger import atomic_write_shard, derive_done_ids, append_jsonl

def test_atomic_write_and_derive(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    recs = [{"id": "a", "status": "ok"}, {"id": "b", "status": "oversized"}]
    atomic_write_shard(out / "CC" / "part-00000.jsonl", recs)
    assert derive_done_ids(out) == {"a", "b"}

def test_no_partial_file_on_crash(tmp_path: Path, monkeypatch):
    out = tmp_path / "outputs" / "step1"
    target = out / "CC" / "part-00001.jsonl"
    import ledger
    def boom(*a, **k): raise OSError("disk full")
    monkeypatch.setattr(ledger.os, "replace", boom)
    try:
        atomic_write_shard(target, [{"id": "x", "status": "ok"}])
    except OSError:
        pass
    assert not target.exists()
    assert not list((out / "CC").glob("*.tmp*")) if (out / "CC").exists() else True

def test_append_jsonl_roundtrip(tmp_path: Path):
    f = tmp_path / "q.jsonl"
    append_jsonl(f, {"id": "z", "attempt_count": 1})
    append_jsonl(f, {"id": "z2", "attempt_count": 2})
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert rows == [{"id": "z", "attempt_count": 1}, {"id": "z2", "attempt_count": 2}]
