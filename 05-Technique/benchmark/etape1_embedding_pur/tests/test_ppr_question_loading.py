from pathlib import Path
import importlib.util
import json


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "25_ppr_kin_sweep.py"
)
spec = importlib.util.spec_from_file_location("ppr_sweep", SCRIPT)
ppr_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppr_sweep)


def test_load_questions_accepts_historical_pourvois_cc_schema(tmp_path):
    bench_path = tmp_path / "bench_global.json"
    bench_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "qid": "legacy-q1",
                        "articles_attendus": ["CPP:1"],
                        "articles_attendus_etendu": ["CPP:1", "CPP:2"],
                        "pourvois_cc": ["12-34.567"],
                        "n_jp_resolues": 1,
                    }
                ]
            }
        )
    )

    questions = ppr_sweep.load_questions(bench_path)

    assert questions == [
        {
            "id": "legacy-q1",
            "gt_strict": {"CPP:1"},
            "gt_ext": {"CPP:1", "CPP:2"},
            "pourvois": {"12-34.567"},
            "gold_jp_ids": set(),
        }
    ]


def test_load_questions_keeps_legacy_pourvoi_only_question_when_no_jp_resolved(tmp_path):
    bench_path = tmp_path / "bench_global.json"
    bench_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "qid": "legacy-q2",
                        "articles_attendus": ["CPP:10"],
                        "pourvois_cc": ["98-76.543"],
                        "n_jp_resolues": 0,
                        "gold_jp_ids": [],
                    }
                ]
            }
        )
    )

    questions = ppr_sweep.load_questions(bench_path)

    assert [question["id"] for question in questions] == ["legacy-q2"]
    assert questions[0]["pourvois"] == {"98-76.543"}
    assert questions[0]["gold_jp_ids"] == set()


def test_load_questions_keeps_modern_gate_for_questions_without_gold_or_pourvoi(tmp_path):
    bench_path = tmp_path / "bench_global.json"
    bench_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "qid": "modern-q0",
                        "articles_attendus": ["CPP:20"],
                        "gold_jp_ids": [],
                        "pourvois_cc": [],
                        "n_jp_resolues": 0,
                    }
                ]
            }
        )
    )

    questions = ppr_sweep.load_questions(bench_path)

    assert questions == []


def test_resolve_question_cache_paths_falls_back_to_legacy_global_bench_names(tmp_path):
    (tmp_path / "questions_977_emb.npy").write_bytes(b"")
    (tmp_path / "questions_977_ids.npy").write_bytes(b"")

    emb_path, ids_path = ppr_sweep.resolve_question_cache_paths(tmp_path)

    assert emb_path == tmp_path / "questions_977_emb.npy"
    assert ids_path == tmp_path / "questions_977_ids.npy"


def test_write_progress_writes_json_payload(tmp_path):
    progress_path = tmp_path / "progress.json"

    ppr_sweep.write_progress(progress_path, {"status": "running", "question_index": 12})

    assert progress_path.exists()
    payload = json.loads(progress_path.read_text())
    assert payload["status"] == "running"
    assert payload["question_index"] == 12
