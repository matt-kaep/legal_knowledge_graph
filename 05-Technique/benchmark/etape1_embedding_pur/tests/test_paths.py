from __future__ import annotations

import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from etape1.paths import resolve_data_root, resolve_repo_root


def test_environment_roots_are_resolved_without_personal_fallback(monkeypatch, tmp_path):
    code_root = tmp_path / "code-checkout"
    data_root = tmp_path / "data-checkout"
    monkeypatch.setenv("LKG_REPO", str(code_root))
    monkeypatch.setenv("LKG_DATA_ROOT", str(data_root))

    assert resolve_repo_root() == code_root.resolve()
    assert resolve_data_root() == data_root.resolve()


def test_repo_root_is_discovered_from_an_anchor_inside_the_checkout(monkeypatch):
    monkeypatch.delenv("LKG_REPO", raising=False)
    monkeypatch.delenv("LKG_DATA_ROOT", raising=False)

    assert resolve_repo_root(Path(__file__)) == Path(__file__).resolve().parents[4]
    assert resolve_data_root(Path(__file__)) == Path(__file__).resolve().parents[4]
