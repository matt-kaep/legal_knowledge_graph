from pathlib import Path
import importlib.util
import hashlib
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


def test_same_provenance_stays_in_same_fold():
    questions = [
        {
            "qid": "q1",
            "source": "source-a",
            "doc_id": "doc-a",
            "section_id": "section-a",
            "enonce": "Premiere question",
            "n_articles_strict": 1,
            "n_jp_resolues": 1,
        },
        {
            "qid": "q2",
            "source": "source-a",
            "doc_id": "doc-a",
            "section_id": "section-a",
            "enonce": "Seconde question",
            "n_articles_strict": 2,
            "n_jp_resolues": 2,
        },
        {"qid": "q3", "n_articles_strict": 1, "n_jp_resolues": 1},
    ]

    df = make_folds.build_fold_assignments(questions)

    assert df.loc[df["qid"].isin(["q1", "q2"]), "fold"].nunique() == 1


def test_normalized_duplicate_text_stays_in_same_fold():
    questions = [
        {
            "qid": "q1",
            "enonce": "Question numero ① : le TEST!",
            "n_articles_strict": 1,
            "n_jp_resolues": 1,
        },
        {
            "qid": "q2",
            "enonce": "question numero 1 le test",
            "n_articles_strict": 1,
            "n_jp_resolues": 1,
        },
        {"qid": "q3", "n_articles_strict": 1, "n_jp_resolues": 1},
    ]

    df = make_folds.build_fold_assignments(questions)

    assert make_folds.normalize_question_text(questions[0]["enonce"]) == questions[1]["enonce"]
    assert df.loc[df["qid"].isin(["q1", "q2"]), "fold"].nunique() == 1


def test_grouping_is_transitive():
    questions = [
        {
            "qid": "q1",
            "source": "source-a",
            "doc_id": "doc-a",
            "section_id": "section-a",
            "enonce": "Texte partage",
            "n_articles_strict": 1,
            "n_jp_resolues": 1,
        },
        {
            "qid": "q2",
            "source": "source-a",
            "doc_id": "doc-a",
            "section_id": "section-a",
            "enonce": "Texte intermediaire",
            "n_articles_strict": 2,
            "n_jp_resolues": 2,
        },
        {
            "qid": "q3",
            "source": "source-b",
            "doc_id": "doc-b",
            "section_id": "section-b",
            "enonce": "texte partage",
            "n_articles_strict": 3,
            "n_jp_resolues": 3,
        },
    ]

    df = make_folds.build_fold_assignments(questions)

    assert df["group_id"].nunique() == 1
    assert df["fold"].nunique() == 1
    assert df["group_size"].tolist() == [3, 3, 3]


def test_questions_without_provenance_or_text_stay_in_distinct_groups():
    questions = [
        {"qid": "q1", "n_articles_strict": 1, "n_jp_resolues": 1},
        {"qid": "q2", "n_articles_strict": 1, "n_jp_resolues": 1},
    ]

    df = make_folds.build_fold_assignments(questions, n_folds=2)

    assert df["group_id"].nunique() == 2
    assert df["group_size"].tolist() == [1, 1]
    assert all(len(group_id) == len("group_") + 64 for group_id in df["group_id"])


def test_assignment_is_deterministic_and_uses_all_folds():
    questions = [
        {"qid": f"q{index}", "n_articles_strict": 1, "n_jp_resolues": 1}
        for index in range(10)
    ]

    df1 = make_folds.build_fold_assignments(questions, n_folds=5, seed=42)
    df2 = make_folds.build_fold_assignments(list(reversed(questions)), n_folds=5, seed=42)

    assert df1.equals(df2)
    assert sorted(df1["fold"].unique().tolist()) == [0, 1, 2, 3, 4]


def test_assignment_uses_all_folds_when_question_count_is_not_divisible():
    questions = [
        {"qid": f"q{index}", "n_articles_strict": 1, "n_jp_resolues": 1}
        for index in range(52)
    ]

    df = make_folds.build_fold_assignments(questions, n_folds=5, seed=42)

    assert sorted(df["fold"].unique().tolist()) == [0, 1, 2, 3, 4]


def test_assignment_balances_each_requested_bucket_on_controlled_fixture():
    article_values = [1, 2, 4]
    jp_values = [1, 2, 4]
    question_types = ["type-a", "type-b", "type-c"]
    granularities = ["large", "precise"]
    questions = []
    for group_index in range(5):
        for question_index in range(6):
            questions.append(
                {
                    "qid": f"q{group_index}-{question_index}",
                    "source": "controlled",
                    "doc_id": f"doc-{group_index}",
                    "section_id": "section",
                    "n_articles_strict": article_values[question_index % len(article_values)],
                    "n_jp_resolues": jp_values[question_index % len(jp_values)],
                    "question_type": question_types[question_index % len(question_types)],
                    "granularity": granularities[question_index % len(granularities)],
                }
            )

    assignments = make_folds.build_fold_assignments(questions, n_folds=5, seed=42)
    metadata = make_folds.build_fold_metadata(assignments, questions, n_folds=5)
    summary = metadata["fold_distribution_summary"]

    for dimension, buckets in {
        "n_articles_strict": ["1", "2-3", "4+"],
        "n_jp_resolues": ["1", "2-3", "4+"],
        "question_type": question_types,
        "granularity": granularities,
    }.items():
        for bucket in buckets:
            counts = [summary[str(fold)][dimension].get(bucket, 0) for fold in range(5)]
            assert max(counts) - min(counts) <= 1


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
    out_dir = bench_root / "_protocol" / "grouped_v2" / "train_augmented_retrievable_strict"
    df = pd.read_csv(out_dir / "fold_assignments.csv")
    meta = json.loads((out_dir / "fold_metadata.json").read_text())
    assert sorted(df["fold"].tolist()) == [0, 1, 2, 3, 4]
    assert meta["protocol_version"] == "grouped_v2"
    assert meta["dataset_split"] == "train_augmented_retrievable_strict"
    assert meta["n_folds"] == 5
    assert meta["source_bench_dir"] == str(bench_dir)
    assert meta["dataset_sha256"] == hashlib.sha256(
        (bench_dir / "bench_global.json").read_bytes()
    ).hexdigest()
    assert meta["fold_assignment_sha256"] == hashlib.sha256(
        (out_dir / "fold_assignments.csv").read_bytes()
    ).hexdigest()
    assert meta["created_at"]
    assert meta["provenance_groups_crossing_folds"] == 0
    assert meta["normalized_text_groups_crossing_folds"] == 0

    with pytest.raises(FileExistsError, match="already exist"):
        make_folds.main([])


def test_main_rejects_non_official_split(tmp_path, monkeypatch):
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        make_folds.main(["--split", "eval_rich_retrievable_strict"])


def test_main_writes_folds_for_the_named_frozen_snapshot(tmp_path, monkeypatch):
    bench_root = tmp_path / "doctrine_v3plus_bench"
    split = make_folds.FROZEN_TRAIN_SPLIT
    bench_dir = bench_root / split
    bench_dir.mkdir(parents=True)
    payload = {
        "questions": [
            {"qid": f"q{index}", "enonce": f"Question {index}", "n_articles_strict": 1, "n_jp_resolues": 1}
            for index in range(10)
        ]
    }
    (bench_dir / "bench_global.json").write_text(json.dumps(payload))
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", bench_root)

    rc = make_folds.main(
        [
            "--split",
            split,
            "--protocol-version",
            make_folds.FROZEN_PROTOCOL_VERSION,
        ]
    )

    out_dir = bench_root / "_protocol" / make_folds.FROZEN_PROTOCOL_VERSION / split
    metadata = json.loads((out_dir / "fold_metadata.json").read_text())
    assert rc == 0
    assert metadata["dataset_split"] == split
    assert metadata["protocol_version"] == make_folds.FROZEN_PROTOCOL_VERSION
    assert metadata["n_questions"] == 10


def test_main_writes_folds_for_candidate_covered_snapshot(tmp_path, monkeypatch):
    bench_root = tmp_path / "doctrine_v3plus_bench"
    split = make_folds.CANDIDATE_COVERED_TRAIN_SPLIT
    bench_dir = bench_root / split
    bench_dir.mkdir(parents=True)
    (bench_dir / "bench_global.json").write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "qid": f"q{index}",
                        "enonce": f"Question {index}",
                        "n_articles_strict": 1,
                        "n_jp_resolues": 1,
                    }
                    for index in range(10)
                ]
            }
        )
    )
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", bench_root)

    rc = make_folds.main(
        [
            "--split",
            split,
            "--protocol-version",
            make_folds.CANDIDATE_COVERED_PROTOCOL_VERSION,
        ]
    )

    out_dir = bench_root / "_protocol" / make_folds.CANDIDATE_COVERED_PROTOCOL_VERSION / split
    metadata = json.loads((out_dir / "fold_metadata.json").read_text())
    assert rc == 0
    assert metadata["dataset_split"] == split
    assert metadata["protocol_version"] == make_folds.CANDIDATE_COVERED_PROTOCOL_VERSION
    assert metadata["n_questions"] == 10


def test_main_rejects_non_five_fold_count(tmp_path, monkeypatch):
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        make_folds.main(["--n-folds", "4"])


def test_main_rejects_noncanonical_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(make_folds.graph_protocol, "BENCH_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        make_folds.main(["--seed", "43"])
