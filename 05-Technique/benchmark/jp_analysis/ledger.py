"""Idempotent ledger. Source of truth = committed JSONL shards (atomic write).
derive_done_ids scans shards at startup. Resolves adversarial finding #3."""
import json
import os
import tempfile
from pathlib import Path

def atomic_write_shard(path: Path, records: list[dict]) -> None:
    """Write all records to a shard via temp file + fsync + atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic on POSIX
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

def derive_done_ids(out_root: Path) -> set[str]:
    """Done = any id with a terminal record in any <juris>/part-*.jsonl."""
    out_root = Path(out_root)
    done: set[str] = set()
    if not out_root.exists():
        return done
    for shard in out_root.glob("*/part-*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done

def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_quarantine(path: Path) -> dict[str, int]:
    """id -> max attempt_count seen (so reruns increment, not reset)."""
    path = Path(path)
    q: dict[str, int] = {}
    if not path.exists():
        return q
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            q[r["id"]] = max(q.get(r["id"], 0), int(r.get("attempt_count", 1)))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return q
