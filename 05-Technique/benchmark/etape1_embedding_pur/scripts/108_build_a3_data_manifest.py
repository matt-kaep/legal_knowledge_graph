#!/usr/bin/env python3
"""Build a reproducible, non-redistributive data manifest for an A3 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A3 = ROOT / "configs/benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json"
DEFAULT_CAMPAIGN = ROOT / "configs/confirmatory_campaign_b1_a3_r1.json"
DEFAULT_OUTPUT = Path("results/benchmark-a3-b1/data-manifest-a3.json")
RECOVERY = (
    "Obtain or reconstruct from the authorized source, preserve the relative path, "
    "then run the A3 preflight."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_data_path(data_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else data_root / path


def _portable_code_path(path: Path, repo_root: Path) -> str:
    """Keep the manifest relocatable when its code files live in the repository."""
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _campaign_data_inputs(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return (role, relative path, expected sha256) triples from B1's input contract."""
    entries: list[tuple[str, str, str]] = []
    for section in ("datasets", "fold_inputs", "candidate_inputs", "graph_inputs"):
        for name, item in payload.get(section, {}).items():
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                continue
            entries.append((f"{section}.{name}", str(item["path"]), str(item["sha256"])))
    return entries


def build_data_manifest(
    *,
    a3_path: Path,
    campaign_path: Path,
    data_root: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify all B1 data inputs and emit a stable, deduplicated public inventory."""
    a3_path = a3_path.resolve()
    campaign_path = campaign_path.resolve()
    data_root = data_root.resolve()
    repo_root = (repo_root or ROOT.parents[2]).resolve()
    a3 = _read_json(a3_path)
    campaign = _read_json(campaign_path)
    expected_a3 = str(campaign.get("a3", {}).get("sha256", ""))
    actual_a3 = sha256(a3_path)
    if actual_a3 != expected_a3:
        raise ValueError(
            f"A3 manifest hash mismatch: expected {expected_a3}, got {actual_a3}"
        )

    grouped: dict[str, dict[str, Any]] = {}
    roles: dict[str, list[str]] = defaultdict(list)
    for role, raw_path, expected in _campaign_data_inputs(campaign):
        existing = grouped.get(raw_path)
        if existing is not None and existing["expected_sha256"] != expected:
            raise ValueError(f"Conflicting expected hashes for {raw_path}")
        grouped.setdefault(raw_path, {"expected_sha256": expected})
        roles[raw_path].append(role)

    artifacts: list[dict[str, Any]] = []
    for raw_path in sorted(grouped):
        path = _resolve_data_path(data_root, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing campaign data input: {path}")
        actual = sha256(path)
        expected = grouped[raw_path]["expected_sha256"]
        if actual != expected:
            raise ValueError(
                f"Campaign data hash mismatch for {raw_path}: expected {expected}, got {actual}"
            )
        artifacts.append(
            {
                "path": raw_path,
                "roles": sorted(roles[raw_path]),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": expected,
                "hash_matches": True,
                "redistribution": "not_cleared",
                "recovery": RECOVERY,
            }
        )

    return {
        "schema_version": 1,
        "manifest_type": "a3_b1_data_inventory",
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_path": _portable_code_path(campaign_path, repo_root),
        "campaign_manifest_sha256": sha256(campaign_path),
        "a3_manifest_id": a3.get("manifest_id"),
        "a3_manifest_path": _portable_code_path(a3_path, repo_root),
        "a3_manifest_sha256": actual_a3,
        "provenance": (
            "Every artifact is an immutable B1 campaign input and was rehashed "
            "against its sealed expected SHA-256."
        ),
        "redistribution": {
            "status": "not_cleared",
            "license": (
                "Do not redistribute benchmark questions, legal text, decisions, "
                "embeddings, or graph inputs until source-specific terms are verified."
            ),
            "recovery": RECOVERY,
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3", type=Path, default=DEFAULT_A3)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LKG_DATA_ROOT", ROOT.parents[2])),
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT.parents[2])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output}")
    payload = build_data_manifest(
        a3_path=args.a3,
        campaign_path=args.campaign,
        data_root=args.data_root,
        repo_root=args.repo_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "artifacts": payload["artifact_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
