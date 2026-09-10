"""Shared contract for E016 graded jurisprudence judgments."""

from __future__ import annotations


VALID_LABELS = ("A", "B", "C", "D", "E", "non_jugeable")
LABEL_GAIN = {
    "A": 1.0,
    "B": 0.5,
    "C": 0.0,
    "D": 0.0,
    "E": 0.0,
    "non_jugeable": 0.0,
}
EXPECTED_FIELDS = {"classe", "justification"}
GENERIC_PATTERNS = (
    "applique directement la règle permettant de répondre",
    "applique directement la regle permettant de repondre",
    "est directement pertinente",
    "apporte une nuance utile",
    "traite le même sujet",
    "traite le meme sujet",
)


def is_generic_justification(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return not normalized or any(pattern in normalized for pattern in GENERIC_PATTERNS)


def validate_judgment(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "not_an_object"
    if set(payload) - EXPECTED_FIELDS:
        return False, "unexpected_fields"
    if payload.get("classe") not in VALID_LABELS:
        return False, "invalid_label"
    justification = str(payload.get("justification") or "").strip()
    if not justification:
        return False, "missing_justification"
    if is_generic_justification(justification):
        return False, "generic_justification"
    return True, ""


def score_labels(labels: list[str], *, k: int = 10) -> float:
    if k <= 0 or len(labels) > k:
        raise ValueError("labels must fit a positive fixed K")
    unknown = sorted(set(labels) - set(VALID_LABELS))
    if unknown:
        raise ValueError(f"unknown label(s): {unknown}")
    return sum(LABEL_GAIN[label] for label in labels) / k
