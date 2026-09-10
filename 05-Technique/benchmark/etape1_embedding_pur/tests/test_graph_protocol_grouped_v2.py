import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "graph_protocol.py"
spec = importlib.util.spec_from_file_location("graph_protocol_grouped_v2", SCRIPT)
graph_protocol = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(graph_protocol)


def test_grouped_v2_paths_do_not_overlap_legacy_protocol(tmp_path):
    legacy_root = tmp_path / "_protocol" / "train_augmented_retrievable_strict"

    assert graph_protocol.protocol_root(tmp_path) == tmp_path / "_protocol" / "grouped_v2"
    assert graph_protocol.cv_root(tmp_path) == tmp_path / "_cv_grouped_v2"
    assert graph_protocol.final_root(tmp_path) == tmp_path / "_final_grouped_v2"
    assert graph_protocol.protocol_root(tmp_path) / "train_augmented_retrievable_strict" != legacy_root


def test_shared_fold_helpers_preserve_legacy_defaults_and_require_explicit_grouped_v2(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(graph_protocol, "BENCH_ROOT", tmp_path)
    split = "train_augmented_retrievable_strict"

    legacy_dir = graph_protocol.resolve_shared_protocol_dir(split)
    legacy_csv, legacy_meta = graph_protocol.resolve_shared_fold_paths(split)
    grouped_dir = graph_protocol.resolve_shared_protocol_dir(
        split, version=graph_protocol.PROTOCOL_VERSION
    )
    grouped_csv, grouped_meta = graph_protocol.resolve_shared_fold_paths(
        split, version=graph_protocol.PROTOCOL_VERSION
    )

    assert legacy_dir == tmp_path / "_protocol" / split
    assert legacy_csv == legacy_dir / "fold_assignments.csv"
    assert legacy_meta == legacy_dir / "fold_assignments_meta.json"
    assert grouped_dir == tmp_path / "_protocol" / "grouped_v2" / split
    assert grouped_csv == grouped_dir / "fold_assignments.csv"
    assert grouped_meta == grouped_dir / "fold_metadata.json"


def test_metadata_reports_zero_group_leakage():
    spec = importlib.util.spec_from_file_location(
        "make_folds_grouped_v2",
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "41_make_kfold_assignments.py",
    )
    make_folds = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(make_folds)
    questions = [
        {
            "qid": "q1",
            "source": "source-a",
            "doc_id": "doc-a",
            "section_id": "section-a",
            "enonce": "Question A",
            "n_articles_strict": 1,
            "n_jp_resolues": 1,
        },
        {
            "qid": "q2",
            "source": "source-a",
            "doc_id": "doc-a",
            "section_id": "section-a",
            "enonce": "Question B",
            "n_articles_strict": 2,
            "n_jp_resolues": 2,
        },
        {"qid": "q3", "n_articles_strict": 3, "n_jp_resolues": 3},
    ]

    assignments = make_folds.build_fold_assignments(questions)
    metadata = make_folds.build_fold_metadata(assignments, questions, n_folds=5)

    assert metadata["n_questions"] == 3
    assert metadata["n_groups"] == 2
    assert metadata["largest_group_size"] == 2
    assert metadata["provenance_groups_crossing_folds"] == 0
    assert metadata["normalized_text_groups_crossing_folds"] == 0
    assert metadata["fold_question_counts"]
    assert metadata["fold_distribution_summary"]
