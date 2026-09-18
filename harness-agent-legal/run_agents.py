#!/usr/bin/env python3
"""Orchestrateur : lance un agent pi par (condition, modalité, question).

Deux conditions :
  - web       : pi + pi-web-access (web_search + built-in), SANS openlegy
  - openlegy  : pi + pi-web-access + openlegy.ts (web + MCP Légifrance)

Lance pi en --mode json, collecte la trace (tool calls, tours, latence) et
extrait la réponse finale de l'assistant (bloc text du message_end).

Usage:
  python3 run_agents.py --condition web|openlegy|all --modality articles|jurisprudence|all
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.tsv"
PROMPT_DIR = HERE
PROMTPS = {"articles": "prompt-articles.txt", "jurisprudence": "prompt-jurisprudence.txt"}
OUT = HERE / "output"
OPENLEGY = HERE / "openlegy.ts"
OPENLEGY_TS = (HERE / "openlegy.ts").resolve()
APPEND_REF_TS = (HERE / "append_reference.ts").resolve()
ENDPOINT = "https://mcp.openlegi.fr/legifrance/mcp"

TOKEN = os.environ.get("OPENLEGY_TOKEN", "")

CONDITIONS = {
    # condition -> list of extra extension args (openlegy.ts); pi-web-access is global via npm install
    "web": [],
    "openlegy": [str(OPENLEGY_TS)],
}


def resolve_lkg_repo() -> Path:
    """Resolve the repository without embedding a developer-specific path."""
    configured = os.environ.get("LKG_REPO")
    return Path(configured).expanduser().resolve() if configured else HERE.parent


def load_questions():
    rows = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        qid, _, enonce = line.partition("\t")
        rows.append({"qid": qid.strip(), "question": enonce.strip()})
    return rows


def build_prompt(prompt_file: str, question: str, csv_path=None) -> str:
    template = (PROMPT_DIR / prompt_file).read_text(encoding="utf-8")
    if csv_path:
        # facilité : l'outil append_reference est disponible ; la consigne principale
        # est déjà dans le prompt fichier (budget de recherche strict + réponse partielle).
        pass
    return f"{template.rstrip()}\n\nQuestion :\n{question}\n"


def collect_final_text(events):
    """Extrait le texte final de l'assistant à partir des events json lus ligne à ligne."""
    # on reconstruit le tout dernier message_end role=assistant
    final_blocks = []
    for e in events:
        if e.get("type") != "message_end":
            continue
        m = e.get("message", {})
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                final_blocks.append(c.get("text", ""))
    return "".join(final_blocks)


def parse_references(text: str):
    """Extrait references[] depuis la réponse (JSON pur ou entouré de texte/backticks)."""
    if not text:
        return []
    # retire fences éventuelles
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.strip()
        if s.startswith("json"):
            s = s[4:].lstrip()
    # cherche le premier objet JSON { ... "references" : [...] }
    import re
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        payload = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    refs = payload.get("references")
    if not isinstance(refs, list):
        return []
    return [str(r) for r in refs if isinstance(r, str) and r]


def summarize_events(events):
    tool_calls = []
    turn_count = 0
    for e in events:
        t = e.get("type")
        if t == "tool_execution_start":
            tool_calls.append({"tool": e.get("toolName"), "args": e.get("args")})
        elif t == "turn_start":
            turn_count += 1
    return {"tool_calls": tool_calls, "turns": turn_count}


def output_paths(out_dir: Path, qid: str, modal: str) -> tuple[Path, Path, Path]:
    """Return short, deterministic artifact paths for one agent invocation.

    Question identifiers are descriptive and can exceed filesystem filename
    limits. The full identifier remains inside every JSON record; only the
    local artifact stem is hashed.
    """
    digest = hashlib.sha256(f"agent-artifact-v1\\0{qid}".encode("utf-8")).hexdigest()
    stem = f"q_{digest}__{modal}"
    return (
        out_dir / f"{stem}.json",
        out_dir / f"{stem}.trace.json",
        out_dir / f"{stem}.csv",
    )


def run_one(condition, qid, modal, question, prompt_file, timeout=300):
    out_dir = OUT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file, trace_file, csv_file = output_paths(out_dir, qid, modal)
    if out_file.exists():
        return {"skipped": True, "path": str(out_file)}

    # Préparation du CSV de sortie (vide au démarrage, rempli par append_reference)
    csv_file.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["OPENLEGY_TOKEN"] = TOKEN
    env["OPENLEGY_ENDPOINT"] = ENDPOINT
    env["APPEND_REFS_PATH"] = str(csv_file)

    extras = [str(APPEND_REF_TS)]
    if condition == "openlegy":
        extras.append(str(OPENLEGY_TS))
    # pi-web-access est global (installé via npm) — pas besoin de -e

    cmd = ["pi", "--mode", "json", "--no-session", "-a"]
    for ext in extras:
        cmd += ["-e", ext]
    prompt = build_prompt(PROMTPS[modal], question, csv_path=csv_file)
    cmd += [prompt]

    started = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env, stdin=subprocess.DEVNULL)
    except OSError as e:
        record = {"condition": condition, "qid": qid, "modalite": modal,
                  "status": "error", "error": f"spawn failed: {e}"}
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "error"}

    import select
    stdout_fd = proc.stdout.fileno()
    lines_buf = []
    timed_out = False

    def _drain():
        while True:
            r, _, _ = select.select([stdout_fd], [], [], 0)
            if not r:
                break
            line = proc.stdout.readline()
            if line == "":
                break
            lines_buf.append(line)

    while True:
        if time.monotonic() - started > timeout:
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            break
        _drain()
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    _drain()
    elapsed = time.monotonic() - started
    stderr_tail = ""
    try:
        stderr_tail = proc.stderr.read()[-2000:]
    except Exception:
        pass

    events = []
    for line in lines_buf:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    trace = summarize_events(events)

    # ── Récupération des références : source PRINCIPALE = CSV append-only ──
    # (écrit par l'agent via append_reference à chaque identification, donc
    #  préservé même si le process est tué par timeout/erreur)
    csv_refs = []
    if csv_file.exists():
        for line in csv_file.read_text(encoding="utf-8").splitlines():
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    order = int(parts[0])
                except ValueError:
                    order = len(csv_refs) + 1
                csv_refs.append({"ordre": order, "reference": parts[1].strip()})
        csv_refs.sort(key=lambda x: x["ordre"])
    # normalise : positions 1..10, vides remplies par ""
    slots = ["" for _ in range(10)]
    for ref in csv_refs:
        idx = ref["ordre"] - 1
        if 0 <= idx < 10 and slots[idx] == "":
            slots[idx] = ref["reference"]
    refs = [r for r in slots if r]  # pour compat format JSON (liste non vide)

    # fallback : si le CSV est vide (agent n'a pas utilisé append_reference),
    # on récupère la réponse JSON du flux (ancien comportement)
    if not refs:
        final_text = collect_final_text(events)
        refs = parse_references(final_text)

    # trace technique
    (trace_file).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    if timed_out:
        status = "ok" if refs else "timeout"
        record = {
            "condition": condition, "qid": qid, "modalite": modal,
            "status": status, "elapsed_seconds": round(elapsed, 3),
            "timed_out": True, "error": f"timeout after {timeout}s",
            "returncode": proc.returncode, "turns": trace["turns"],
            "tool_calls": trace["tool_calls"], "references": refs,
            "slots_10": slots, "raw_text": "",
        }
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": status, "turns": trace["turns"], "tool_calls": len(trace["tool_calls"]),
                "timed_out": True, "refs": len(refs)}

    if proc.returncode not in (0, None) and not events and not refs:
        record = {"condition": condition, "qid": qid, "modalite": modal,
                  "status": "error", "returncode": proc.returncode,
                  "stderr": stderr_tail, "elapsed_seconds": elapsed}
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "error"}

    record = {
        "condition": condition, "qid": qid, "modalite": modal,
        "status": "ok", "elapsed_seconds": round(elapsed, 3),
        "returncode": proc.returncode, "turns": trace["turns"],
        "tool_calls": trace["tool_calls"],
        "references": refs, "slots_10": slots,
        "raw_text": "",
    }
    out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "turns": trace["turns"], "tool_calls": len(trace["tool_calls"]), "n_refs_csv": len(csv_refs)}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="all", choices=["web", "openlegy", "all"])
    ap.add_argument("--modality", default="all", choices=["articles", "jurisprudence", "all"])
    ap.add_argument("--limit", type=int, default=None, help="nb de questions à traiter (pour pilote)")
    ap.add_argument("--timeout", type=int, default=300, help="timeout (s) par run pi")
    ap.add_argument("--start", type=int, default=0, help="indice de départ dans questions.tsv (ex: 3 traite les 7 suivantes)")
    args = ap.parse_args()

    if not TOKEN and args.condition in ("openlegy", "all"):
        print("WARNING: OPENLEGY_TOKEN vide — la condition openlegy échouera sur les appels MCP.")

    conditions = ["web", "openlegy"] if args.condition == "all" else [args.condition]
    modals = ["articles", "jurisprudence"] if args.modality == "all" else [args.modality]

    questions = load_questions()
    if args.start:
        questions = questions[args.start:]
    if args.limit:
        questions = questions[:args.limit]

    summary = []
    for cond in conditions:
        for q in questions:
            for modal in modals:
                r = run_one(cond, q["qid"], modal, q["question"], PROMTPS[modal], timeout=args.timeout)
                summary.append({"condition": cond, "qid": q["qid"], "modalite": modal, **r})
                print(f"[{cond}/{modal}] {q['qid'][:50]} -> {r.get('status')} "
                      f"tours={r.get('turns')} tools={r.get('tool_calls')}", flush=True)
                (HERE / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTerminé. Résumé :", HERE / "run_summary.json", flush=True)


if __name__ == "__main__":
    main()
