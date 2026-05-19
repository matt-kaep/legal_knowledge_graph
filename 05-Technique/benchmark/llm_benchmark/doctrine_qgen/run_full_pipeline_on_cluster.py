#!/usr/bin/env python3
"""
run_full_pipeline_on_cluster.py — orchestrateur Python du pipeline doctrine_qgen.

Étapes :
    1. Vérifier / installer les dépendances Python (requirements.txt)
    2. Phase 0 : parse PDFs → JSON sectionné (idempotent)
    3. Phase 1 : génération vLLM, modèle A puis modèle B
    4. Phase 2 : validation extraction-only (déterministe)
    5. Phase 3 : anonymisation pour blind review
    6. Résumé final

Usage :
    cd doctrine_qgen
    python3 run_full_pipeline_on_cluster.py                      # tout
    python3 run_full_pipeline_on_cluster.py --skip-install       # skip pip install
    python3 run_full_pipeline_on_cluster.py --only-phase 0       # une phase
    python3 run_full_pipeline_on_cluster.py --models gemma4-31B  # un seul modèle
    python3 run_full_pipeline_on_cluster.py --max-len 12288 --gpu-util 0.88
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
os.chdir(HERE)

DEFAULT_MODELS = ["gemma4-31B", "gemma4-26B-A4B"]

# Mapping nom-pip → nom-import (pour vérifier la présence sans installer si déjà là)
REQUIRED_PACKAGES = {
    "vllm":            "vllm",
    "pdfplumber":      "pdfplumber",
    "pypdf":           "pypdf",
    "jsonschema":      "jsonschema",
    "requests":        "requests",
    "unidecode":       "unidecode",
    "openai":          "openai",
    "huggingface_hub": "huggingface_hub",
    "tqdm":            "tqdm",
}


# ═══════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════

def banner(title: str) -> None:
    print()
    print("═" * 16 + f" {title} " + "═" * 16)


def run_step(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        print(f"[FAIL] commande sortie code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def run_step_soft(cmd: list[str]) -> int:
    """Comme run_step mais NE tue PAS le pipeline : retourne le code.

    Utilisé pour phase 1 par modèle : un modèle qui ne charge pas
    (checkpoint incompatible vLLM) ne doit pas faire perdre les modèles
    déjà générés ni empêcher phases 2-3.
    """
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        print(f"[WARN] {cmd[2] if len(cmd) > 2 else cmd[0]} sortie code "
              f"{res.returncode} — on continue", file=sys.stderr)
    return res.returncode


# ═══════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════

def check_missing_packages() -> list[str]:
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def install_requirements(skip_vllm: bool = False) -> None:
    """Installe les deps depuis requirements.txt. vLLM est lourd → optionnel.

    skip_vllm=True : utile si vLLM est déjà installé via un autre canal
    (conda env du cluster, build custom CUDA, etc.)
    """
    req_file = HERE / "requirements.txt"
    if not req_file.exists():
        print("[ERROR] requirements.txt absent", file=sys.stderr)
        sys.exit(1)

    if skip_vllm:
        # pip install ligne par ligne sauf vllm
        lines = [
            ln.strip() for ln in req_file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#") and not ln.startswith("vllm")
        ]
        if not lines:
            return
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *lines]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet",
               "-r", str(req_file)]

    print(f"[deps] pip install ({'sans vllm' if skip_vllm else 'tout'})")
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        print("[FAIL] installation pip échouée", file=sys.stderr)
        sys.exit(res.returncode)


# ═══════════════════════════════════════════════════════════════════════
# PHASES
# ═══════════════════════════════════════════════════════════════════════

def phase_0_parse(force: bool = False) -> None:
    banner("PHASE 0 — parse PDFs")

    parsed_dir = HERE / "data" / "parsed_doctrine_sections"
    pdf_dir = HERE / "data" / "source_pdfs"
    existing_json = list(parsed_dir.glob("*.json")) if parsed_dir.exists() else []

    # Cas typique cluster : JSON déjà rsync'ed depuis local, pas de PDFs.
    # On skippe gracieusement sauf si --force-phase-0.
    if existing_json and not force:
        print(f"[skip] {len(existing_json)} JSON déjà présents dans "
              f"{parsed_dir.relative_to(HERE)} — phase 0 sautée.")
        print("       (utiliser --force-phase-0 pour re-parser depuis les PDFs)")
        return

    if not pdf_dir.exists() and not existing_json:
        print(f"[FAIL] ni PDFs dans {pdf_dir.relative_to(HERE)}, "
              f"ni JSON dans {parsed_dir.relative_to(HERE)}.", file=sys.stderr)
        print("       Copie soit les PDFs (parsing sur cluster), "
              "soit les JSON déjà parsés (recommandé).", file=sys.stderr)
        sys.exit(1)

    run_step([sys.executable, "phase_0_parse_doctrine_pdfs.py"])


def phase_1_generate(model_alias: str, max_len: int,
                     gpu_util: float, port: int) -> int:
    banner(f"PHASE 1 — generate ({model_alias})")
    return run_step_soft([
        sys.executable, "phase_1_generate_crfpa_questions_via_vllm.py",
        "--model", model_alias,
        "--max-len", str(max_len),
        "--gpu-util", str(gpu_util),
        "--port", str(port),
    ])


def models_with_output() -> list[str]:
    """Alias distincts ayant produit au moins un fichier de questions."""
    per_model = HERE / "results" / "per_model"
    aliases: set[str] = set()
    if per_model.exists():
        for doc_dir in per_model.iterdir():
            if doc_dir.is_dir():
                for f in doc_dir.glob("*_questions.json"):
                    aliases.add(f.stem.replace("_questions", ""))
    return sorted(aliases)


def phase_2_validate() -> None:
    banner("PHASE 2 — validation strict extraction-only")
    run_step([sys.executable, "phase_2_validate_strict_extraction_only.py"])


def phase_3_anonymize() -> None:
    banner("PHASE 3 — anonymisation pour blind review")
    run_step([sys.executable, "phase_3_anonymize_outputs_for_blind_review.py",
              "--all"])


# ═══════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    banner("RÉSUMÉ")

    per_model = HERE / "results" / "per_model"
    valid_dir = HERE / "results" / "validation"
    blind_dir = HERE / "results" / "for_blind_review"

    totals: dict[str, dict[str, int]] = {}
    if per_model.exists():
        for doc_dir in sorted(per_model.iterdir()):
            if not doc_dir.is_dir():
                continue
            for f in doc_dir.glob("*_questions.json"):
                alias = f.stem.replace("_questions", "")
                payload = json.loads(f.read_text())
                t = totals.setdefault(alias, {"docs": 0, "questions": 0})
                t["docs"] += 1
                t["questions"] += payload.get("n_questions_total", 0)

    print("Questions générées :")
    if not totals:
        print("  (aucun output trouvé)")
    for alias, t in totals.items():
        print(f"  {alias:20s} : {t['questions']:4d} questions "
              f"sur {t['docs']} docs")

    print("\nValidation (taux de pass) :")
    if valid_dir.exists():
        reports = sorted(valid_dir.glob("*_validation_report.json"))
        if not reports:
            print("  (aucun rapport)")
        for f in reports:
            r = json.loads(f.read_text())
            c = r["counters"]
            tot = max(1, c["n_questions_total"])
            rate = 100.0 * c["n_passed"] / tot
            print(f"  {f.stem:60s} : {rate:5.1f}% "
                  f"({c['n_passed']}/{c['n_questions_total']})")
    else:
        print("  (dossier validation absent)")

    print(f"\nBlind review prêt dans : {blind_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════════════
# ENV CHECK
# ═══════════════════════════════════════════════════════════════════════

def check_env_vars() -> None:
    if not os.environ.get("HF_TOKEN"):
        print("[WARN] HF_TOKEN non défini — vLLM peut échouer à pull "
              "les modèles AWQ privés/gated")
    if not os.environ.get("HF_HOME"):
        print("[WARN] HF_HOME non défini — cache HF par défaut "
              "(~/.cache/huggingface)")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-install", action="store_true",
                   help="ne pas lancer pip install (deps déjà présentes)")
    p.add_argument("--skip-vllm-install", action="store_true",
                   help="installer toutes les deps sauf vllm")
    p.add_argument("--only-phase", type=int, choices=[0, 1, 2, 3],
                   help="ne lancer qu'une seule phase (0=parse, 1=gen, "
                        "2=validate, 3=anonymize)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"alias modèles vLLM (défaut: {DEFAULT_MODELS})")
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--gpu-util", type=float, default=0.92)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--force-phase-0", action="store_true",
                   help="forcer le re-parsing même si les JSON sont déjà là")
    args = p.parse_args()

    # 1. Dépendances
    if not args.skip_install:
        missing = check_missing_packages()
        if missing:
            print(f"[deps] manquant: {missing}")
            install_requirements(skip_vllm=args.skip_vllm_install)
            still_missing = check_missing_packages()
            if still_missing and not args.skip_vllm_install:
                print(f"[FAIL] toujours manquant après install: {still_missing}",
                      file=sys.stderr)
                sys.exit(1)
        else:
            print("[deps] toutes les dépendances présentes")
    else:
        print("[deps] skip install (--skip-install)")

    # 2. Sanity ENV
    check_env_vars()

    # 3. Phases
    if args.only_phase is None or args.only_phase == 0:
        phase_0_parse(force=args.force_phase_0)
    if args.only_phase is None or args.only_phase == 1:
        ok_models, ko_models = [], []
        for model in args.models:
            rc = phase_1_generate(model, args.max_len, args.gpu_util, args.port)
            (ok_models if rc == 0 else ko_models).append(model)
        print(f"\n[phase 1] modèles OK: {ok_models or '—'}  "
              f"| KO: {ko_models or '—'}")

    if args.only_phase is None or args.only_phase == 2:
        # Valide ce qui existe (modèle KO n'a juste pas d'output à valider).
        phase_2_validate()

    if args.only_phase is None or args.only_phase == 3:
        # La blind review A/B exige ≥2 modèles ; sinon on saute proprement
        # plutôt que de produire un comparatif corrompu (correctness).
        avail = models_with_output()
        if len(avail) >= 2:
            phase_3_anonymize()
        else:
            print(f"\n[phase 3] SAUTÉE — blind review A/B nécessite ≥2 "
                  f"modèles, {len(avail)} disponible(s) ({avail or '—'}). "
                  f"Résous le 2e modèle puis relance --only-phase 3.")

    # 4. Résumé
    print_summary()


if __name__ == "__main__":
    main()
