"""Phase 0 — Parse les PDFs Jurisclasseurs en JSON sectionné.

Sortie : data/parsed_doctrine_sections/<doc_id>.json (1 par PDF)

Logique reprise de ../parse_doctrine/parse_audit.py :
  - extract_text via pdfplumber (fallback pypdf)
  - normalisation espaces / exposants
  - détection sections L1 (I. ...) et L2 (A. ...)
  - extraction articles via extract_pairs_v5 (de iterate_regex.py)
  - extraction jurisprudence (Cass / CEDH / CC / CJUE / pourvoi)

Post-processing nouveau : groupage hiérarchique L1 → L2 children, et
attribution des articles/JP à leur span L1 (par offset).

Idempotent : ne reparse pas si le JSON existe déjà, sauf --force.

Usage :
    python3 phase_0_parse_doctrine_pdfs.py [--force] [--pdf-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
DATA_PDFS = HERE / "data" / "source_pdfs"
DATA_OUT = HERE / "data" / "parsed_doctrine_sections"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# iterate_regex.py est livré dans le bundle (autonomie cluster)
sys.path.insert(0, str(HERE))

import pdfplumber  # noqa: E402
from iterate_regex import extract_pairs_v5  # noqa: E402

# Liste par défaut des 5 PDFs subset (override via --pdf-dir avec listing complet)
DEFAULT_PDFS = [
    "Art. 11 - Fasc. 20 _ Secret de l_instruction.pdf",
    "Art. 114 à 121 - Fasc. 20 _ Interrogatoires et confrontations.pdf",
    "Art. 100 à 100-7 - Fasc. 20 _ Interceptions des correspondances émises par la voie des communications électroniques.pdf",
    "Art. 10 - Fasc. 20 _ ACTION CIVILE. – Prescription. – Mesures d_instruction.pdf",
    "App. Art. 11 - Fasc. 20 _ Protection de la présomption d’innocence.pdf",
]


# ── PDF → texte ───────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> tuple[str, int]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            chunks = [(p.extract_text() or "") for p in pdf.pages]
            return "\n".join(chunks), n_pages
    except Exception as e:
        print(f"  [pdfplumber failed: {e}, fallback pypdf]")
        from pypdf import PdfReader
        r = PdfReader(str(pdf_path))
        chunks = [p.extract_text() or "" for p in r.pages]
        return "\n".join(chunks), len(r.pages)


# ── Normalisation ─────────────────────────────────────────────────────────

_SPACE_FIXES = {
    " ": " ", " ": " ", " ": " ",
    "​": "", "﻿": "",
}
_SUPERSCRIPT = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}


def normalize_text(t: str) -> str:
    for k, v in _SPACE_FIXES.items():
        t = t.replace(k, v)
    for k, _ in _SUPERSCRIPT.items():
        t = t.replace(k, " ")
    return t


def strip_footnote_markers(t: str) -> str:
    for k in _SUPERSCRIPT:
        t = t.replace(k, "")
    return t


# ── doc_id slug ───────────────────────────────────────────────────────────

def make_doc_id(filename: str) -> str:
    base = Path(filename).stem
    nfkd = unicodedata.normalize("NFKD", base)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", no_accents).strip("_").lower()
    return slug


def make_doc_title(filename: str) -> str:
    """Titre lisible dérivé du nom de fichier."""
    stem = Path(filename).stem
    # "Art. 11 - Fasc. 20 _ Secret de l_instruction" → invert
    parts = stem.split(" _ ", 1)
    if len(parts) == 2:
        prefix, label = parts
        label = label.replace("_", "'")
        return f"{label} ({prefix})"
    return stem.replace("_", "'")


# ── Sections (L1 / L2) ────────────────────────────────────────────────────

SECTION_L1_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<num>(?:[IVX]{1,5}))\s*\.\s*"
    r"(?:[—–\-]\s*)?"
    r"(?P<title>[A-ZÉÈÀÂÊÎÔÛÇa-zéèàâêîôûç][^\n]{4,140})$"
)

SECTION_L2_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<num>[A-H])\s*\.\s*"
    r"(?:[—–\-]\s*)?"
    r"(?P<title>[A-ZÉÈÀÂÊÎÔÛÇa-zéèàâêîôûç][^\n]{4,140})$"
)

_BIBLIO_HINTS = re.compile(
    r"\b(JCP|Gaz\.\s*Pal|D\.\s*\d{4}|Bull\.|JurisData|Rec\.|Rev\.|"
    r"chron\.|obs\.|note\s+[A-Z]|comm\.|RTD|RSC|"
    r"éd\.|ed\.|p\.\s*\d|n°\s*\d{2,})",
    re.IGNORECASE,
)


def _is_biblio_line(title: str) -> bool:
    if _BIBLIO_HINTS.search(title):
        return True
    if re.search(r"\b[A-Z]\.\s*[A-ZÉÈÀ][a-zéèàâêîôû]{2,}", title):
        return True
    words = title.split()
    if len(words) <= 3 and all(re.match(r"^[A-ZÉÈÀ][a-zéèàâêîôû\-']+$", w) for w in words):
        return True
    return False


def detect_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    seen_offsets: set[int] = set()

    for m in SECTION_L1_RE.finditer(text):
        title = m.group("title").strip().rstrip(".,;:")
        if _is_biblio_line(title) or len(title) < 8:
            continue
        sections.append({
            "level": 1, "num": m.group("num"),
            "title": title[:140], "offset": m.start(),
        })
        seen_offsets.add(m.start())

    for m in SECTION_L2_RE.finditer(text):
        if m.start() in seen_offsets:
            continue
        title = m.group("title").strip().rstrip(".,;:")
        if _is_biblio_line(title) or len(title) < 8:
            continue
        sections.append({
            "level": 2, "num": m.group("num"),
            "title": title[:140], "offset": m.start(),
        })

    sections.sort(key=lambda s: s["offset"])
    return sections


# ── Jurisprudence ─────────────────────────────────────────────────────────

POURVOI_RE = re.compile(
    r"(?:pourvoi|n°|n[°º]|no)\s*[:.]?\s*"
    r"(?P<num>\d{2}[\-\.]\d{2,3}[\.\-]\d{2,4})",
    re.IGNORECASE,
)
CASS_RE = re.compile(
    r"\b(?:Cass\.?|Civ\.?|Crim\.?|Com\.?|Soc\.?|Ass\.?\s*pl[ée]n\.?|Ch\.?\s*mixte)"
    r"(?:\s*,?\s*(?:1re|1ère|2e|3e|crim\.?|civ\.?|com\.?|soc\.?))?"
    r"\s*,?\s*"
    r"(?P<date>\d{1,2}\s+"
    r"(?P<month>janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc)[a-zéèêû.]*"
    r"\s+(?P<year>\d{4}))"
    r"(?:[^.\n]{0,80}?n°?\s*(?P<pourvoi>\d{2}[\-\.]\d{2,3}[\.\-]\d{2,4}))?",
    re.IGNORECASE,
)
CEDH_RE = re.compile(
    r"\bCEDH\s*,?\s*(?:Gr\.?\s*Ch\.?\s*,?\s*)?"
    r"\d{1,2}\s+"
    r"(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc)[a-zéèêû.]*"
    r"\s+\d{4}"
    r"(?:[^.\n]{0,150}?req\.?\s*n°?\s*\d+/\d+)?",
    re.IGNORECASE,
)
CC_RE = re.compile(
    r"\bCons\.?\s*const\.?\s*,?\s*"
    r"(?:(?:déc|décision)\.?\s*n°?\s*)?\d{4}[\-]\d{2,4}\s*(?:DC|QPC)?"
    r"(?:\s*du\s+\d{1,2}\s+\w+\s+\d{4})?",
    re.IGNORECASE,
)
CJUE_RE = re.compile(
    r"\b(?:CJUE|CJCE)\s*,?\s*"
    r"\d{1,2}\s+"
    r"(?:janv|févr|mars|avr|mai|juin|juill|août|sept|oct|nov|déc)[a-zéèêû.]*"
    r"\s+\d{4}"
    r"(?:[^.\n]{0,150}?(?:aff\.?\s*(?:C-)?\d+/\d+)?)?",
    re.IGNORECASE,
)

_FR_MONTHS = {
    "janv": 1, "févr": 2, "fevr": 2, "mars": 3, "avr": 4, "mai": 5,
    "juin": 6, "juill": 7, "août": 8, "aout": 8, "sept": 9, "oct": 10,
    "nov": 11, "déc": 12, "dec": 12,
}


def _parse_iso_date(day: str, month_token: str, year: str) -> str | None:
    key = month_token.lower().rstrip(".").rstrip("u").rstrip("é")[:4]
    # Tente la clé directe puis 3 lettres
    for k in (month_token.lower().rstrip("."), month_token.lower()[:4],
              month_token.lower()[:3]):
        if k in _FR_MONTHS:
            return f"{int(year):04d}-{_FR_MONTHS[k]:02d}-{int(day):02d}"
    return None


def extract_jurisprudence_with_meta(text: str) -> list[dict]:
    """Renvoie une liste plate {short_ref, pourvoi?, chamber?, date?, offset, type}."""
    out: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    # Cass — on tente de capturer chambre + date + pourvoi
    for m in CASS_RE.finditer(text):
        span = (m.start(), m.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        ref = re.sub(r"\s+", " ", m.group(0).strip())[:120]
        # chambre
        chamber = None
        head = m.group(0).lower()
        for ch_key, ch_val in (("crim", "crim"), ("civ", "civ"),
                                ("com", "com"), ("soc", "soc"),
                                ("plén", "ass_plen"), ("plen", "ass_plen"),
                                ("mixte", "mixte")):
            if ch_key in head:
                chamber = ch_val
                break
        # date ISO
        iso_date = None
        try:
            day_m = re.search(r"(\d{1,2})\s+(\w+)[a-zéèêû.]*\s+(\d{4})", m.group(0))
            if day_m:
                iso_date = _parse_iso_date(day_m.group(1), day_m.group(2), day_m.group(3))
        except Exception:
            pass
        item = {
            "type": "cass",
            "short_ref": ref,
            "offset": m.start(),
        }
        if chamber:
            item["chamber"] = chamber
        if iso_date:
            item["date"] = iso_date
        try:
            pv = m.groupdict().get("pourvoi")
            if pv:
                item["pourvoi"] = pv
        except (IndexError, KeyError):
            pass
        out.append(item)

    for m in CEDH_RE.finditer(text):
        span = (m.start(), m.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append({
            "type": "cedh",
            "short_ref": re.sub(r"\s+", " ", m.group(0).strip())[:120],
            "offset": m.start(),
        })
    for m in CC_RE.finditer(text):
        span = (m.start(), m.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append({
            "type": "cc",
            "short_ref": re.sub(r"\s+", " ", m.group(0).strip())[:120],
            "offset": m.start(),
        })
    for m in CJUE_RE.finditer(text):
        span = (m.start(), m.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append({
            "type": "cjue",
            "short_ref": re.sub(r"\s+", " ", m.group(0).strip())[:120],
            "offset": m.start(),
        })
    # Pourvois orphelins (numéro seul, hors d'un span déjà capturé)
    for m in POURVOI_RE.finditer(text):
        if any(s[0] <= m.start() <= s[1] for s in seen_spans):
            continue
        out.append({
            "type": "pourvoi_only",
            "short_ref": f"pourvoi n° {m.group('num')}",
            "pourvoi": m.group("num"),
            "offset": m.start(),
        })
    return out


# ── Articles avec offset ──────────────────────────────────────────────────
# extract_pairs_v5 retourne un set sans offsets. Pour pouvoir attribuer
# chaque article à un span L1, on relance une regex grossière pour récupérer
# les offsets, puis on filtre l'union avec extract_pairs_v5 (qui valide la
# normalisation code+article).

ARTICLE_OFFSET_RE = re.compile(
    r"(?:articles?|arts?\.?)\s+"
    r"(?P<art>(?:[LRDAE]\.?\s*)?\d[\d\-\.]*"
    r"(?:\s*(?:bis|ter|quater|quinquies|sexies))?)"
    r"(?:\s+(?:du|de\s+la|de\s+l['’])\s+(?:nouveau\s+|ancien\s+)?"
    r"(?:code|loi|décret|convention|règlement)[^.;\n]{0,80})?",
    re.IGNORECASE,
)


def extract_articles_with_offsets(text: str, valid_pairs: set[str]) -> list[dict]:
    """Renvoie [{code_slug, article_num, offset}]. Filtre par valid_pairs."""
    # valid_pairs = set "code_slug:article_num" venant de extract_pairs_v5
    # On ne peut pas mapper exactement chaque match au code_slug correct sans
    # rejouer toute la logique de v5. Compromis pragmatique : pour chaque
    # paire validée, on cherche TOUTES les occurrences du article_num dans
    # le texte et on prend leurs offsets. Le filtrage par span L1 sera fait
    # ensuite : un article apparaît dans tous les spans L1 où son numéro
    # textuel apparaît au moins une fois APRÈS une mention de son code.
    # Pour rester simple et robuste, on associe chaque pair_key à toutes
    # les positions du article_num trouvées. C'est conservateur (on peut
    # avoir doublons inter-spans) mais n'introduit pas de faux articles.
    out: list[dict] = []
    for pk in valid_pairs:
        if ":" not in pk:
            continue
        code_slug, art = pk.split(":", 1)
        # Cherche le numéro brut dans le texte
        # On évite les surmatchs en exigeant un word-boundary à gauche/droite
        num_core = re.escape(art)
        # remplace - par patterns avec espaces possibles
        try:
            for m in re.finditer(rf"\b{num_core}\b", text):
                out.append({
                    "code_slug": code_slug,
                    "article_num": art,
                    "offset": m.start(),
                })
        except re.error:
            continue
    return out


# ── Groupage hiérarchique ─────────────────────────────────────────────────

def _collect_refs(
    start: int,
    end: int,
    articles_with_offset: list[dict],
    jp_with_meta: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Articles + JP dont l'offset ∈ [start, end), dédupliqués.

    Même logique pour un span L1 ou un span L2 — d'où l'extraction en
    helper : la whitelist d'une sous-section L2 est strictement celle de
    son propre span, pas celle du L1 parent.
    """
    seen_pk: set[str] = set()
    articles_in_span: list[dict] = []
    for a in articles_with_offset:
        if start <= a["offset"] < end:
            pk = f"{a['code_slug']}:{a['article_num']}"
            if pk in seen_pk:
                continue
            seen_pk.add(pk)
            articles_in_span.append({
                "code_slug": a["code_slug"],
                "article_num": a["article_num"],
            })

    seen_refs: set[str] = set()
    jp_in_span: list[dict] = []
    for jp in jp_with_meta:
        if start <= jp["offset"] < end:
            key = jp.get("pourvoi") or jp["short_ref"]
            if key in seen_refs:
                continue
            seen_refs.add(key)
            entry = {"short_ref": jp["short_ref"]}
            for k in ("pourvoi", "chamber", "date"):
                if k in jp:
                    entry[k] = jp[k]
            jp_in_span.append(entry)

    return articles_in_span, jp_in_span


def group_sections_hierarchical(
    sections: list[dict],
    text: str,
    articles_with_offset: list[dict],
    jp_with_meta: list[dict],
) -> list[dict]:
    """Construit sections_l1 ; chaque L2 enfant est AUTONOME.

    Un L2 enfant porte son propre `text` et ses propres
    `articles_in_span` / `jp_in_span` (scopés à son span d'offsets), pour
    que phase_1 puisse générer dessus en respectant l'extraction-only
    stricte sans réutiliser la whitelist (plus large) du L1 parent.
    """
    # Les bornes d'un L1 = depuis son offset jusqu'au prochain L1 (ou EOF)
    l1_indices = [i for i, s in enumerate(sections) if s["level"] == 1]
    out: list[dict] = []
    for idx_pos, i in enumerate(l1_indices):
        s = sections[i]
        start = s["offset"]
        end = (sections[l1_indices[idx_pos + 1]]["offset"]
               if idx_pos + 1 < len(l1_indices) else len(text))

        # L2 enfants : sections L2 dont offset ∈ [start, end), autonomes
        l2_children = []
        for j in range(i + 1, len(sections)):
            s2 = sections[j]
            if s2["offset"] >= end:
                break
            if s2["level"] == 2:
                # offset_end du L2 = prochain L2/L1 ou end
                next_off = end
                for k in range(j + 1, len(sections)):
                    if sections[k]["offset"] < end:
                        next_off = sections[k]["offset"]
                        break
                l2_start = s2["offset"]
                l2_arts, l2_jp = _collect_refs(
                    l2_start, next_off, articles_with_offset, jp_with_meta)
                l2_children.append({
                    "section_id": f"L1_{idx_pos+1:03d}_L2_{s2['num']}",
                    "title": f"{s2['num']}. — {s2['title']}",
                    "offset_start": l2_start,
                    "offset_end": next_off,
                    "text": text[l2_start:next_off],
                    "articles_in_span": l2_arts,
                    "jp_in_span": l2_jp,
                })

        articles_in_span, jp_in_span = _collect_refs(
            start, end, articles_with_offset, jp_with_meta)

        out.append({
            "section_id": f"L1_{idx_pos+1:03d}",
            "title": f"{s['num']}. — {s['title']}",
            "offset_start": start,
            "offset_end": end,
            "text_l1_with_l2_children": text[start:end],
            "l2_children": l2_children,
            "articles_in_span": articles_in_span,
            "jp_in_span": jp_in_span,
        })
    return out


# ── Entrée principale ─────────────────────────────────────────────────────

def parse_one_pdf(pdf_path: Path) -> dict:
    raw_text, n_pages = extract_text(pdf_path)
    text = normalize_text(raw_text)
    text_for_regex = strip_footnote_markers(text)

    sections = detect_sections(text)
    valid_pairs = extract_pairs_v5(text_for_regex)
    articles_offsets = extract_articles_with_offsets(text_for_regex, valid_pairs)
    jp_meta = extract_jurisprudence_with_meta(text_for_regex)

    sections_l1 = group_sections_hierarchical(sections, text, articles_offsets, jp_meta)

    return {
        "doc_id": make_doc_id(pdf_path.name),
        "doc_title": make_doc_title(pdf_path.name),
        "source_pdf_filename": pdf_path.name,
        "n_pages": n_pages,
        "n_chars": len(text),
        "sections_l1": sections_l1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=DATA_PDFS,
                    help=f"Dossier des PDFs (défaut : {DATA_PDFS})")
    ap.add_argument("--force", action="store_true",
                    help="Re-parse même si JSON déjà présent")
    args = ap.parse_args()

    pdf_dir: Path = args.pdf_dir
    if not pdf_dir.exists():
        print(f"[ERREUR] PDF dir introuvable : {pdf_dir}")
        return 1

    # Liste tous les PDFs du dossier (pas seulement les 5 du subset, pour permettre extension)
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf")])
    if not pdfs:
        print(f"[WARN] aucun PDF dans {pdf_dir}")
        return 1

    print(f"PDF dir   : {pdf_dir}")
    print(f"Output    : {DATA_OUT}")
    print(f"PDFs      : {len(pdfs)}")
    print()

    n_done, n_skipped, n_failed = 0, 0, 0
    for pdf in pdfs:
        doc_id = make_doc_id(pdf.name)
        out_json = DATA_OUT / f"{doc_id}.json"
        if out_json.exists() and not args.force:
            print(f"  SKIP {doc_id} (déjà parsé)")
            n_skipped += 1
            continue
        print(f"  PARSE {doc_id}")
        try:
            payload = parse_one_pdf(pdf)
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            n_l1 = len(payload["sections_l1"])
            n_arts = sum(len(s["articles_in_span"]) for s in payload["sections_l1"])
            n_jp = sum(len(s["jp_in_span"]) for s in payload["sections_l1"])
            print(f"    OK : {n_l1} L1  ·  {n_arts} articles  ·  {n_jp} jp")
            n_done += 1
        except Exception as e:
            print(f"    FAIL : {e}")
            n_failed += 1

    print(f"\nDone={n_done}  Skipped={n_skipped}  Failed={n_failed}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
