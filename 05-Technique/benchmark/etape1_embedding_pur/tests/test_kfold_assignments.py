from pathlib import Path
import importlib.util
import json

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "41_make_kfold_assignments.py"
)
spec = importlib.util.spec_from_file_location("make_folds", SCRIPT)
make_folds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_folds)


def test_build_fold_assignments_is_deterministic():
    questions = [
        {"qid": "q1", "n_articles_strict": 1, "n_jp_resolues": 1},
        {"qid": "q2", "n_articles_strict": 1, "n_jp_resolues": 2},
        {"qid": "q3", "n_articles_strict": 2, "n_jp_resolues": 1},
        {"qid": "q4", "n_articles_strict": 3, "n_jp_resolues": 2},
        {"qid": "q5", "n_articles_strict": 1, "n_jp_resolues": 1},
    ]
    df1 = make_folds.build_fold_assignments(questions, n_folds=5, seed=42)
    df2 = make_folds.build_fold_assignments(questions, n_folds=5, seed=42)
    assert df1.equals(df2)
    assert sorted(df1["fold"].tolist()) == [0, 1, 2, 3, 4]


def test_main_writes_shared_canonical_folds_from_official_train_bench(tmp_path, monkeypatch):
    bench_root = tmp_path / "doctrine_v3plus_bench"
    bench_dir = bench_root / "train_augmented_retrievable_strict"
    bench_dir.mkdir(parents=True)
    payload = {
        "questions": [
            {"qid": "q1", "n_articles_strict": 1, "n_jp_resolues": 1},
            {"qid": "q2", "n_articles_strict": 1, "n_jp_resolues": 2},
            {"qid": "q3", "n_articles_strict": 2, "n_jp_resolues": 1},
            {"qid": "q4", "n_articles_strict": 3, "n_jp_resolues": 2},
            {"qid": "q5", "n_articles_strict": 1, "n_jp_resolues": 1},
        ]
    }
    (bench_dir / "bench_global.json").write_text(json.dumps(payload))
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", bench_root)

    rc = make_folds.main([])

    assert rc == 0
    out_dir = bench_root / "_protocol" / "train_augmented_retrievable_strict"
    df = pd.read_csv(out_dir / "fold_assignments.csv")
    meta = json.loads((out_dir / "fold_assignments_meta.json").read_text())
    assert sorted(df["fold"].tolist()) == [0, 1, 2, 3, 4]
    assert meta["split"] == "train_augmented_retrievable_strict"
    assert meta["n_folds"] == 5
    assert meta["source_bench_dir"] == str(bench_dir)


def test_main_rejects_non_official_split(tmp_path, monkeypatch):
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        make_folds.main(["--split", "eval_rich_retrievable_strict"])


def test_main_rejects_non_five_fold_count(tmp_path, monkeypatch):
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        make_folds.main(["--n-folds", "4"])
