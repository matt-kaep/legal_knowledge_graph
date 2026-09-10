"""Pure, shared rendering for the frozen B2/E029 reranking prompt."""

from __future__ import annotations

import json
from typing import Any


def render_reranking_prompt(prompt_template: str, job: dict[str, Any]) -> str:
    """Render only the candidate fields visible to the reranking model."""
    visible_candidates = [
        {"item_id": candidate["item_id"], "text": candidate["text"]}
        for candidate in job["candidates"]
    ]
    return (
        f"{prompt_template.rstrip()}\n\nQuestion :\n{job['question']}\n\n"
        f"Références à ordonner :\n{json.dumps(visible_candidates, ensure_ascii=False, indent=2)}\n"
    )
