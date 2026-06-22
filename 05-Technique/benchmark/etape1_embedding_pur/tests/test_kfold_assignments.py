from pathlib import Path
import importlib.util

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
