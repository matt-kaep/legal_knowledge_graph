"""Phase 1 — Génère des questions CRFPA via vLLM (mode A/B sur 2 modèles).

Pour un modèle donné, lance vLLM en local, parcourt les sections L1 de
chaque doc parsé, et appelle l'API chat/completions avec un response_format
JSON schema strict. Output : results/per_model/<doc_id>/<alias>_questions.json.

Inspiré de ../run_all_models.py pour les patterns vLLM (start/wait/kill).

Usage :
    python3 phase_1_generate_crfpa_questions_via_vllm.py --model gemma4-31B
    python3 phase_1_generate_crfpa_questions_via_vllm.py --model gemma4-26B-A4B
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
PARSED_DIR = HERE / "data" / "parsed_doctrine_sections"
RESULTS_DIR = HERE / "results" / "per_model"
LOG_DIR = HERE / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_PATH = HERE / "prompts" / "extract_crfpa_questions_v1.txt"
SCHEMA_PATH = HERE / "schemas" / "crfpa_question_format_option_c.json"

VLLM_PID_F = LOG_DIR / "vllm_qgen.pid"

# ── Modèles A/B (les 2 visés uniquement) ─────────────────────────────────
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gemma4-31B":     ("QuantTrio/gemma-4-31B-it-AWQ",         "Gemma 4 dense 31B — AWQ 4-bit"),
    "gemma4-26B-A4B": ("cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", "Gemma 4 MoE 26B/4B — AWQ 4-bit"),
}

# Révision HF épinglée par modèle (None = main).
# gemma4-26B-A4B : le repo a été ré-uploadé le 2026-05-01 (re-quantif →
# nommage de poids d'experts MoE que vLLM 0.19 ne sait plus charger,
# KeyError 'layers.0.experts.0.down_proj.weight_packed'). On repointe
# sur le dernier upload AVANT le run OK du 28/04 (commit du 2026-04-12).
MODEL_REVISION: dict[str, str] = {
    "gemma4-26B-A4B": "519bdca117c8",
}


# ═══════════════════════════════════════════════════════════════════════
# vLLM lifecycle
# ═══════════════════════════════════════════════════════════════════════

def kill_vllm() -> None:
    if not VLLM_PID_F.exists():
        return
    try:
        pid = int(VLLM_PID_F.read_text())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(5)
    except (ProcessLookupError, ValueError, PermissionError):
        pass
    VLLM_PID_F.unlink(missing_ok=True)


def start_vllm(model_id: str, log_path: Path, port: int, max_len: int,
               gpu_util: float, num_gpus: int,
               revision: str | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--tensor-parallel-size", str(num_gpus),
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(gpu_util),
        "--port", str(port),
    ]
    if revision:
        cmd += ["--revision", revision]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                             preexec_fn=os.setsid)
    VLLM_PID_F.write_text(str(proc.pid))
    return proc


def wait_vllm(proc: subprocess.Popen, port: int, timeout_s: int = 900) -> bool:
    health = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def detect_num_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        return max(1, len([l for l in out.strip().split("\n") if l]))
    except Exception:
        return 1


# ═══════════════════════════════════════════════════════════════════════
# Génération
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(template: str, doc_id: str, unit: dict) -> str:
    """Construit le prompt pour une unité de génération (L1 entier ou L2).

    Une unité a la même forme quelle que soit la granularité :
    {section_id, title, text, articles_in_span, jp_in_span, l2_titles}.
    """
    # Substitution par .replace() et NON .format() : le prompt contient un
    # exemple few-shot avec du JSON littéral ({"code_slug": ...}), que
    # str.format() prendrait pour des champs à formater (KeyError).
    subs = {
        "{doc_id}": doc_id,
        "{l1_title}": unit["title"],
        "{l2_titles_list}": json.dumps(unit.get("l2_titles", []), ensure_ascii=False),
        "{section_text}": unit["text"],
        "{articles_json}": json.dumps(unit.get("articles_in_span", []), ensure_ascii=False),
        "{jp_json}": json.dumps(unit.get("jp_in_span", []), ensure_ascii=False),
    }
    out = template
    for placeholder, value in subs.items():
        out = out.replace(placeholder, value)
    return out


def _est_tokens(s: str) -> int:
    """Estimation conservatrice (≈3 chars/token → surestime → on découpe
    plutôt que de risquer un dépassement ; le garde-fou vllm_400 reste)."""
    return len(s) // 3


def make_generation_units(
    doc: dict, template: str, doc_id: str,
    budget_tokens: int, min_unit_tokens: int = 200,
) -> list[dict]:
    """Découpage HYBRIDE : un L1 qui rentre dans le budget reste entier ;
    un L1 trop gros est remplacé par ses L2 enfants (autonomes, whitelists
    strictes produites par phase_0). Les unités de texte trop courtes
    (bruit parser : titres bibliographiques) sont écartées."""
    units: list[dict] = []
    for l1 in doc["sections_l1"]:
        l1_unit = {
            "section_id": l1["section_id"],
            "title": l1["title"],
            "text": l1["text_l1_with_l2_children"],
            "articles_in_span": l1.get("articles_in_span", []),
            "jp_in_span": l1.get("jp_in_span", []),
            "l2_titles": [c["title"] for c in l1.get("l2_children", [])],
            "offset_start": l1["offset_start"],
            "offset_end": l1["offset_end"],
        }
        prompt = build_prompt(template, doc_id, l1_unit)
        if _est_tokens(prompt) <= budget_tokens:
            units.append(l1_unit)
            continue
        # L1 trop gros → ses L2 enfants autonomes
        children = l1.get("l2_children", [])
        if not children:
            # Pas de L2 pour découper : on garde le L1, le filet vllm_400
            # le marquera proprement plutôt que de le perdre en silence.
            units.append(l1_unit)
            continue
        for c in children:
            units.append({
                "section_id": c["section_id"],
                "title": c["title"],
                "text": c.get("text", ""),
                "articles_in_span": c.get("articles_in_span", []),
                "jp_in_span": c.get("jp_in_span", []),
                "l2_titles": [],
                "offset_start": c["offset_start"],
                "offset_end": c["offset_end"],
            })

    kept = [u for u in units if _est_tokens(u["text"]) >= min_unit_tokens]
    skipped = len(units) - len(kept)
    if skipped:
        print(f"  ({skipped} unité(s) <{min_unit_tokens} tok ignorée(s) "
              f"— bruit parser)")
    return kept


def call_vllm_with_schema(client, model_id: str, prompt: str,
                           schema: dict, temperature: float = 0.2,
                           max_tokens: int = 2048) -> dict | None:
    """Appel chat/completions vLLM avec response_format json_schema strict."""
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "crfpa_questions",
                "schema": schema,
                "strict": True,
            },
        },
    )
    txt = resp.choices[0].message.content
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def run_generation_for_model(alias: str, hf_id: str, args) -> int:
    """Génère pour TOUS les docs parsés. Retourne 0 si OK."""
    import openai
    from openai import OpenAI

    if not PROMPT_PATH.exists():
        print(f"[ERREUR] prompt manquant : {PROMPT_PATH}")
        return 1
    if not SCHEMA_PATH.exists():
        print(f"[ERREUR] schema manquant : {SCHEMA_PATH}")
        return 1

    template = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        print(f"[ERREUR] aucun parsed dans {PARSED_DIR} — lancer phase_0 d'abord")
        return 1

    # Démarre vLLM
    log_path = LOG_DIR / f"vllm_qgen_{alias}.log"
    num_gpus = args.num_gpus or detect_num_gpus()
    revision = MODEL_REVISION.get(alias)
    rev_msg = f"  rev={revision}" if revision else ""
    print(f"  ↳ start vLLM ({hf_id})  GPUs={num_gpus}  "
          f"max_len={args.max_len}{rev_msg}")
    t0 = time.time()
    proc = start_vllm(hf_id, log_path, args.port, args.max_len,
                       args.gpu_util, num_gpus, revision=revision)
    try:
        if not wait_vllm(proc, args.port, timeout_s=args.wait_timeout):
            print(f"  ✗ vLLM KO — voir {log_path}")
            return 2
        print(f"  ↳ vLLM prêt ({int(time.time()-t0)}s)")

        # timeout par requête + AUCUN retry : si vLLM se fige (grammaire
        # xgrammar bloquée, GPU à 0%), réessayer sur le même serveur figé
        # ne fait que doubler le temps perdu.
        client = OpenAI(base_url=f"http://localhost:{args.port}/v1",
                         api_key="EMPTY",
                         timeout=args.req_timeout, max_retries=0)

        # Budget d'entrée = contexte modèle − sortie réservée − marge
        # (template chat, tokens spéciaux). Un L1 dont le prompt estimé
        # dépasse ce budget est découpé en ses L2.
        budget_tokens = args.max_len - args.max_tokens_out - 512
        print(f"  ↳ budget entrée = {budget_tokens} tok "
              f"(max_len {args.max_len} − sortie {args.max_tokens_out} − 512)")

        # Disjoncteur : N timeouts CONSÉCUTIFS = vLLM figé → on arrête
        # proprement ce modèle au lieu de gaspiller req_timeout × N unités.
        consec_timeouts = 0
        vllm_dead = False

        for parsed_file in parsed_files:
            if vllm_dead:
                break
            doc = json.loads(parsed_file.read_text(encoding="utf-8"))
            doc_id = doc["doc_id"]
            out_dir = RESULTS_DIR / doc_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{alias}_questions.json"
            if out_file.exists() and not args.force:
                print(f"  SKIP {doc_id}/{alias} (existe)")
                continue

            units = make_generation_units(
                doc, template, doc_id, budget_tokens)
            print(f"  GEN {doc_id}  ({len(doc['sections_l1'])} L1 → "
                  f"{len(units)} unités à générer)")
            sections_out = []
            for unit in units:
                offs = [unit["offset_start"], unit["offset_end"]]
                prompt = build_prompt(template, doc_id, unit)
                try:
                    # 1er essai
                    result = call_vllm_with_schema(
                        client, hf_id, prompt, schema,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens_out,
                    )
                    # retry une fois si échec
                    if result is None:
                        result = call_vllm_with_schema(
                            client, hf_id, prompt, schema,
                            temperature=0.3,
                            max_tokens=args.max_tokens_out,
                        )
                except openai.BadRequestError as e:
                    # Typiquement : prompt + sortie > contexte max du modèle.
                    # vLLM A RÉPONDU (juste un 400) → reset du disjoncteur.
                    consec_timeouts = 0
                    print(f"    SKIP {unit['section_id']} — vLLM 400 "
                          f"(contexte dépassé ?) : {e}")
                    sections_out.append({
                        "section_id": unit["section_id"],
                        "doc_id": doc_id,
                        "questions": [],
                        "_error": "vllm_400",
                        "_error_message": str(e),
                        "_source_offsets": offs,
                    })
                    continue
                except (openai.APITimeoutError,
                        openai.APIConnectionError) as e:
                    # vLLM n'a PAS répondu (figé / coupé). Unité marquée.
                    consec_timeouts += 1
                    print(f"    SKIP {unit['section_id']} — timeout/conn "
                          f"API ({consec_timeouts}) : {e}")
                    sections_out.append({
                        "section_id": unit["section_id"],
                        "doc_id": doc_id,
                        "questions": [],
                        "_error": "api_timeout",
                        "_error_message": str(e),
                        "_source_offsets": offs,
                    })
                    if consec_timeouts >= args.max_consec_timeouts:
                        print(f"  ⚠ {consec_timeouts} timeouts consécutifs "
                              f"→ vLLM figé, abandon du modèle {alias} "
                              f"(docs restants non générés, reprenables).")
                        vllm_dead = True
                        break
                    continue
                # vLLM a répondu (conforme ou non) → reset du disjoncteur.
                consec_timeouts = 0
                if result is None:
                    sections_out.append({
                        "section_id": unit["section_id"],
                        "doc_id": doc_id,
                        "questions": [],
                        "_error": "schema_validation_failed_twice",
                        "_source_offsets": offs,
                    })
                else:
                    # Force section_id/doc_id + trace les offsets de l'unité
                    # réellement émise (L1 entier ou sous-section L2).
                    result["section_id"] = unit["section_id"]
                    result["doc_id"] = doc_id
                    result["_source_offsets"] = offs
                    sections_out.append(result)

            if vllm_dead:
                # Doc interrompu par le disjoncteur : NE PAS écrire de
                # fichier (sinon out_file.exists() le skipperait à jamais
                # alors qu'un rerun avec vLLM frais le réussirait).
                print(f"    ✗ {doc_id} non écrit (vLLM figé) — "
                      f"sera repris au prochain run.")
                break

            payload = {
                "doc_id": doc_id,
                "model_alias": alias,
                "model_id": hf_id,
                "n_sections": len(sections_out),
                "n_questions_total": sum(
                    len(s.get("questions", [])) for s in sections_out),
                "sections": sections_out,
            }
            out_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"    -> {out_file}  ({payload['n_questions_total']} questions)")
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        return 130
    finally:
        kill_vllm()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=list(MODEL_REGISTRY.keys()),
                    help="Alias du modèle à utiliser")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--max-tokens-out", type=int, default=4096,
                    help="tokens de sortie réservés (questions entières)")
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-gpus", type=int, default=None)
    ap.add_argument("--wait-timeout", type=int, default=900)
    ap.add_argument("--req-timeout", type=float, default=900.0,
                    help="timeout (s) par requête vLLM (défaut 900 = 15 min)")
    ap.add_argument("--max-consec-timeouts", type=int, default=3,
                    help="N timeouts consécutifs = vLLM figé → abandon modèle")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--force", action="store_true",
                    help="Régénérer même si le fichier existe")
    args = ap.parse_args()

    alias = args.model
    hf_id, note = MODEL_REGISTRY[alias]
    print(f"{'='*70}\n[{alias}] {hf_id}\n  {note}\n{'='*70}")

    kill_vllm()
    rc = run_generation_for_model(alias, hf_id, args)
    print(f"\nReturn code : {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
