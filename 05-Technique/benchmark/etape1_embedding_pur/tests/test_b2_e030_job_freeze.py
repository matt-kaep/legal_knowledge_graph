import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "111_freeze_b2_e030_judge_jobs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("b2_e030_job_freeze", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path: Path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"qid": "q1", "enonce": "Quelle règle ?"}]}), encoding="utf-8")
    rankings = tmp_path / "rankings.parquet"
    pd.DataFrame(
        [{"qid": "q1", "modality": "jp", "rank": rank, "item_id": f"j{rank}"} for rank in range(1, 11)]
    ).to_parquet(rankings, index=False)
    texts = tmp_path / "texts.parquet"
    pd.DataFrame([{"jp_id": f"j{rank}", "synthese": f"Synthese {rank}"} for rank in range(1, 11)]).to_parquet(texts, index=False)
    order = tmp_path / "order.npy"
    np.save(order, np.array([f"j{rank}" for rank in range(1, 11)], dtype=object))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("QUESTION: {question}\\nFICHE: {document}", encoding="utf-8")
    return questions, rankings, texts, order, prompt


def test_freeze_jobs_binds_sources_and_only_embeds_visible_card(tmp_path):
    freezer = _load_module()
    questions, rankings, texts, order, prompt = _inputs(tmp_path)
    output = tmp_path / "jobs.jsonl"

    receipt = freezer.freeze_jobs(
        rankings_path=rankings,
        questions_path=questions,
        texts_path=texts,
        candidate_order_path=order,
        prompt_path=prompt,
        output_path=output,
        family="ppr_reranked_kin70",
        modality="jp",
        model_id="model",
        model_revision="revision",
    )

    jobs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert receipt["jobs"] == 10
    assert len(jobs) == 10
    assert jobs[0]["document"] == {"synthese": "Synthese 1"}
    assert "rank" not in jobs[0]
    assert "source_method" not in jobs[0]
    assert jobs[0]["source_ranking_sha256"] == freezer.sha256(rankings)
    assert jobs[-1]["position"] == 10


def test_freeze_jobs_rejects_duplicate_or_outside_candidate(tmp_path):
    freezer = _load_module()
    questions, rankings, texts, order, prompt = _inputs(tmp_path)
    frame = pd.read_parquet(rankings)
    frame.loc[9, "item_id"] = "outside"
    frame.to_parquet(rankings, index=False)

    with pytest.raises(ValueError, match="outside the frozen A3 candidate universe"):
        freezer.freeze_jobs(
            rankings_path=rankings,
            questions_path=questions,
            texts_path=texts,
            candidate_order_path=order,
            prompt_path=prompt,
            output_path=tmp_path / "jobs.jsonl",
            family="cosine",
            modality="jp",
            model_id="model",
            model_revision="revision",
        )


def test_freeze_jobs_never_overwrites_an_immutable_list(tmp_path):
    freezer = _load_module()
    questions, rankings, texts, order, prompt = _inputs(tmp_path)
    output = tmp_path / "jobs.jsonl"
    output.write_text("existing\\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable E030 job list"):
        freezer.freeze_jobs(
            rankings_path=rankings,
            questions_path=questions,
            texts_path=texts,
            candidate_order_path=order,
            prompt_path=prompt,
            output_path=output,
            family="cosine",
            modality="jp",
            model_id="model",
            model_revision="revision",
        )


def test_freeze_jobs_selects_one_explicit_condition_before_validating_ranks(tmp_path):
    freezer = _load_module()
    questions, rankings, texts, order, prompt = _inputs(tmp_path)
    selected = pd.read_parquet(rankings)
    selected["condition_id"] = "G7-jp"
    distractor = selected.copy()
    distractor["condition_id"] = "G1-jp"
    pd.concat([selected, distractor], ignore_index=True).to_parquet(rankings, index=False)

    receipt = freezer.freeze_jobs(
        rankings_path=rankings,
        questions_path=questions,
        texts_path=texts,
        candidate_order_path=order,
        prompt_path=prompt,
        output_path=tmp_path / "jobs.jsonl",
        family="ppr_g7",
        modality="jp",
        model_id="model",
        model_revision="revision",
        filters={"condition_id": "G7-jp"},
    )

    assert receipt["ranking_filters"] == {"condition_id": "G7-jp"}
    assert len((tmp_path / "jobs.jsonl").read_text(encoding="utf-8").splitlines()) == 10


def test_freeze_jobs_preserves_direct_llm_unresolved_reference_as_zero_slot(tmp_path):
    freezer = _load_module()
    questions, rankings, texts, order, prompt = _inputs(tmp_path)
    frame = pd.read_parquet(rankings)
    frame.loc[0, "item_id"] = None
    frame.to_parquet(rankings, index=False)

    freezer.freeze_jobs(
        rankings_path=rankings,
        questions_path=questions,
        texts_path=texts,
        candidate_order_path=order,
        prompt_path=prompt,
        output_path=tmp_path / "jobs.jsonl",
        family="direct_llm",
        modality="jp",
        model_id="model",
        model_revision="revision",
        allow_zero_slots=True,
    )

    first = json.loads((tmp_path / "jobs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["zero_slot"] is True
    assert first["candidate_id_internal"] is None
    assert first["document"] is None
