import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "74_g7_graded_jp_contract.py"
SPEC = importlib.util.spec_from_file_location("g7_graded_jp_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_score_labels_keeps_fixed_k_denominator_for_non_judgeable_slots():
    assert MODULE.score_labels(["A"] * 4 + ["B"] * 2 + ["C"] * 4, k=10) == 0.5
    assert MODULE.score_labels(["A", "non_jugeable"], k=10) == 0.1


def test_score_labels_rejects_unknown_labels_and_overfilled_rankings():
    with pytest.raises(ValueError, match="unknown label"):
        MODULE.score_labels(["Z"], k=10)
    with pytest.raises(ValueError, match="fit a positive fixed K"):
        MODULE.score_labels(["A", "B"], k=1)


def test_validate_judgment_rejects_generic_a_justification():
    valid, reason = MODULE.validate_judgment(
        {
            "classe": "A",
            "justification": "La décision applique directement la règle permettant de répondre.",
        }
    )
    assert not valid
    assert reason == "generic_justification"


def test_validate_judgment_accepts_concrete_legal_reason():
    valid, reason = MODULE.validate_judgment(
        {
            "classe": "A",
            "justification": (
                "La Cour exige la présence du ministère public même lorsque la juridiction "
                "statue seulement sur l'action civile, ce qui répond à la condition de "
                "régularité posée."
            ),
        }
    )
    assert valid
    assert reason == ""


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"classe": "Z", "justification": "Une règle précise est citée."}, "invalid_label"),
        ({"classe": "B", "justification": ""}, "missing_justification"),
        (
            {
                "classe": "C",
                "justification": "La fiche traite le sujet mais ne permet pas de répondre.",
                "extra": "interdit",
            },
            "unexpected_fields",
        ),
    ],
)
def test_validate_judgment_enforces_minimal_output_contract(payload, reason):
    assert MODULE.validate_judgment(payload) == (False, reason)
