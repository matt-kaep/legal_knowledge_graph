import importlib.util
from pathlib import Path
import unittest
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "114_audit_b2_e030_context.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("b2_e030_context_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _job(*, modality: str, qid: str, position: int, document: Optional[Dict], zero_slot: bool = False) -> Dict:
    return {
        "job_id": f"{modality}-{qid}-{position}",
        "family": "cosine",
        "modality": modality,
        "qid": qid,
        "position": position,
        "question": "Quelle règle est applicable ?",
        "candidate_id_internal": None if zero_slot else f"candidate-{position}",
        "document": document,
        "zero_slot": zero_slot,
    }


class E030ContextAuditTest(unittest.TestCase):
    def test_chat_counter_uses_input_ids_not_batch_encoding_key_count(self):
        """A tokenizer returning input_ids/attention_mask must count token ids, not two mapping keys."""
        auditor = _load_auditor()

        class BatchEncodingLikeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return {"input_ids": [10, 11, 12, 13], "attention_mask": [1, 1, 1, 1]}

        tokenizer = BatchEncodingLikeTokenizer()
        counter = auditor._chat_token_counter_from_tokenizer(tokenizer)

        self.assertEqual(counter("un prompt"), 4)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "un prompt"}])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_reserves_completion_budget_and_keeps_zero_slots_out_of_model_prompts(self):
        """A future change that forgets completion tokens must fail this audit."""
        auditor = _load_auditor()
        jobs = [
            _job(modality="article", qid="q1", position=1, document={"texte": "règle courte"}),
            _job(modality="article", qid="q1", position=2, document=None, zero_slot=True),
        ]

        report = auditor.audit_jobs(
            jobs=jobs,
            templates={"article": "QUESTION {question}\nARTICLE {document}", "jp": "QUESTION {question}\nFICHE {document}"},
            max_model_tokens=14,
            max_output_tokens=5,
            count_tokens=lambda text: len(text.split()),
        )

        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(report["jobs"], 2)
        self.assertEqual(report["zero_slots"], 1)
        self.assertEqual(report["model_jobs"], 1)
        self.assertEqual(report["incompatible_jobs"], 0)
        self.assertEqual(report["maximum_prompt_tokens"], 9)
        self.assertEqual(report["maximum_total_tokens"], 14)

    def test_reports_a_prompt_that_exceeds_input_plus_completion_budget(self):
        """A future change that accepts an overflowing prompt must fail this audit."""
        auditor = _load_auditor()
        jobs = [
            _job(modality="jp", qid="q2", position=1, document={"synthese": "un deux trois quatre cinq six"}),
        ]

        report = auditor.audit_jobs(
            jobs=jobs,
            templates={"article": "QUESTION {question}\nARTICLE {document}", "jp": "QUESTION {question}\nFICHE {document}"},
            max_model_tokens=10,
            max_output_tokens=3,
            count_tokens=lambda text: len(text.split()),
        )

        self.assertEqual(report["model_jobs"], 1)
        self.assertEqual(report["incompatible_jobs"], 1)
        self.assertFalse(report["compatible"])
        self.assertEqual(report["maximum_total_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
