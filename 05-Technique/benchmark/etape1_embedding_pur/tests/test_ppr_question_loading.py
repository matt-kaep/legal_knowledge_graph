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


def test_resolve_question_cache_paths_falls_back_to_legacy_global_bench_names(tmp_path):
    (tmp_path / "questions_977_emb.npy").write_bytes(b"")
    (tmp_path / "questions_977_ids.npy").write_bytes(b"")

    emb_path, ids_path = ppr_sweep.resolve_question_cache_paths(tmp_path)

    assert emb_path == tmp_path / "questions_977_emb.npy"
    assert ids_path == tmp_path / "questions_977_ids.npy"
