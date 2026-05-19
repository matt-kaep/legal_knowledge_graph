"""Classify exceptions into terminal (write failed_terminal record, never retry)
vs retryable (quarantine, retry next run). Resolves adversarial finding #2."""
from enum import Enum
from parsing import ParseError

class ErrorClass(str, Enum):
    TERMINAL = "terminal"
    RETRYABLE = "retryable"

def classify_error(exc: Exception) -> ErrorClass:
    # Parse/validation/content errors: data is bad, retrying won't help.
    if isinstance(exc, (ParseError, ValueError)):
        return ErrorClass.TERMINAL
    # HTTP status if present (openai.APIStatusError exposes .status_code)
    status = getattr(exc, "status_code", None)
    if status is not None:
        if 500 <= int(status) <= 599:
            return ErrorClass.RETRYABLE
        return ErrorClass.TERMINAL  # 4xx incl. 400 context overflow
    # Infra: timeouts / connection drops are retryable.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ErrorClass.RETRYABLE
    # Unknown -> retryable (safer: re-tried, never silently lost).
    return ErrorClass.RETRYABLE
