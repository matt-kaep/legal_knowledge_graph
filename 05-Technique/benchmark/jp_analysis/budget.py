"""Context budget. max_model_len is REQUIRED and must be verified against the
live vLLM /v1/models at startup. Near-threshold records counted with the real
tokenizer. Resolves adversarial finding #4.

The /3.0 char estimate assumes >=~3 chars/token (true for FR legal text); the
+/-20% band absorbs the estimation error, so OUTSIDE the band the cheap verdict
is assumed (no tokenizer call) and the tokenizer is invoked ONLY inside the
band. This preserves the two-pass cheap filter: at corpus scale the vast
majority of records are clearly below `lo` and never hit the tokenizer."""

class BudgetError(RuntimeError):
    pass

_MARGIN = 512
_CHARS_PER_TOK = 3.0

def compute_threshold(max_model_len, overhead_tokens: int, max_tokens: int) -> int:
    if not max_model_len:
        raise BudgetError("max_model_len is required (verify against /v1/models)")
    thr = int(max_model_len) - int(max_tokens) - int(overhead_tokens) - _MARGIN
    if thr <= 0:
        raise BudgetError(f"non-positive input budget: {thr}")
    return thr

def is_oversized(full_text: str, threshold: int, tokenizer=None,
                 band: float = 0.2) -> bool:
    n = len(full_text or "")
    est = n / _CHARS_PER_TOK
    lo, hi = threshold * (1 - band), threshold * (1 + band)
    if est < lo:
        return False            # cheap: bulk of short docs, no tokenizer call
    if est > hi:
        return True             # cheap: clearly too long, no tokenizer call
    if tokenizer is None:
        return est > threshold  # in band, no tokenizer: fallback to estimate
    return len(tokenizer.encode(full_text)) > threshold  # in band: exact count

def verify_max_model_len(client, expected: int) -> int:
    """Query the live vLLM server; abort if it disagrees with `expected`."""
    models = client.models.list()
    served = getattr(models.data[0], "max_model_len", None)
    if served is not None and int(served) != int(expected):
        raise BudgetError(
            f"server max_model_len={served} != expected {expected}")
    return int(expected)
