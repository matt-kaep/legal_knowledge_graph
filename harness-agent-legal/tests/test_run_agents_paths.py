"""Regression test for portable per-question artifact paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_agents.py"
SPEC = importlib.util.spec_from_file_location("agent_runner", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_output_paths_use_short_deterministic_names_for_long_qids(tmp_path: Path) -> None:
    qid = "very-long-question-id-" * 30

    first = runner.output_paths(tmp_path, qid, "articles")
    second = runner.output_paths(tmp_path, qid, "articles")
    other = runner.output_paths(tmp_path, qid + "other", "articles")

    assert first == second
    assert first != other
    assert all(len(path.name.encode("utf-8")) <= 255 for path in first)
    assert all(path.parent == tmp_path for path in first)


def test_repository_root_can_be_supplied_by_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LKG_REPO", str(tmp_path))

    assert runner.resolve_lkg_repo() == tmp_path.resolve()
