"""Harnais d'itération locale pour améliorer le regex d'extraction d'articles.

Entrées
-------
- cluster_data/regex_validation/sample_100.jsonl : 100 arrêts avec `text`
- regex_validation_FN_20260423-1459.csv         : pair_keys trouvés par LLM
                                                   mais manqués par regex baseline

Objectif : mesurer combien de FN sont récupérés par des variantes successives
du regex, sans inflation de la précision.

Usage
-----
$ python iterate_regex.py                 # compare baseline + V1 + V2 + V3
$ python iterate_regex.py --version v2    # ne teste qu'une version
$ python iterate_regex.py --show-missed 20  # liste 20 FN encore manqués

Les variantes sont définies dans `extractors.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).parent
SAMPLE  = ROOT / "cluster_data" / "regex_validation" / "sample_100.jsonl"
FN_CSV  = ROOT / "regex_validation_FN_20260423-1459.csv"

# ═══════════════════════════════════════════════════════════════════════
# NORMALISATION (identique à enrichissement_base_complete.ipynb)
# ═══════════════════════════════════════════════════════════════════════

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def normalize_code(name: str) -> str:
    s = name.lower().strip()
    s = _strip_accents(s)
    s = re.sub(r"[’'ʼ']", "_", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def normalize_article(prefix: str, number: str) -> str:
    prefix = prefix.upper().strip() if prefix else ""
    num = re.sub(r"[\s.]+", "", number.strip())
    if prefix:
        return f"{prefix}{num}"
    return num

def make_pair_key(code_slug: str, article_norm: str) -> str:
    return f"{code_slug}:{article_norm}"

# ═══════════════════════════════════════════════════════════════════════
# LISTE DE CODES (identique)
# ═══════════════════════════════════════════════════════════════════════

CODES_OFFICIELS = [
    "Code civil",
    "Code de commerce",
    "Code de déontologie des architectes",
    "Code de justice administrative",
    "Code de justice militaire (nouveau)",
    "Code de l'action sociale et des familles",
    "Code de l'entrée et du séjour des étrangers et du droit d'asile",
    "Code de l'environnement",
    "Code de l'expropriation pour cause d'utilité publique",
    "Code de l'organisation judiciaire",
    "Code de l'urbanisme",
    "Code de l'énergie",
    "Code de la commande publique",
    "Code de la consommation",
    "Code de la construction et de l'habitation",
    "Code de la défense",
    "Code de la famille et de l'aide sociale",
    "Code de la justice pénale des mineurs",
    "Code de la mutualité",
    "Code de la propriété intellectuelle",
    "Code de la route",
    "Code de la santé publique",
    "Code de la sécurité intérieure",
    "Code de la sécurité sociale",
    "Code de la voirie routière",
    "Code de procédure civile",
    "Code de procédure pénale",
    "Code des assurances",
    "Code des communes",
    "Code des douanes",
    "Code des impositions sur les biens et services",
    "Code des juridictions financières",
    "Code des pensions civiles et militaires de retraite",
    "Code des pensions militaires d'invalidité et des victimes de guerre",
    "Code des postes et des communications électroniques",
    "Code des procédures civiles d'exécution",
    "Code des relations entre le public et l'administration",
    "Code du cinéma et de l'image animée",
    "Code du patrimoine",
    "Code du service national",
    "Code du sport",
    "Code du travail",
    "Code forestier (nouveau)",
    "Code général de la fonction publique",
    "Code général de la propriété des personnes publiques",
    "Code général des collectivités territoriales",
    "Code général des impôts",
    "Code général des impôts, annexe IV",
    "Code minier (nouveau)",
    "Code monétaire et financier",
    "Code pénal",
    "Code pénitentiaire",
    "Code rural et de la pêche maritime",
    "Code électoral",
    "Livre des procédures fiscales",
]

CODES_HISTORIQUES = {
    "Nouveau code de procédure civile": "Code de procédure civile",
    "Nouveau Code de procédure civile": "Code de procédure civile",
    "ancien code de procédure civile":  "Code de procédure civile",
    "Code d'instruction criminelle":    "Code d'instruction criminelle",
    "Code Napoléon":                    "Code civil",
    "Code rural":                       "Code rural et de la pêche maritime",
    "Code forestier":                   "Code forestier (nouveau)",
    "Code minier":                      "Code minier (nouveau)",
    "Code de justice militaire":        "Code de justice militaire (nouveau)",
    # Acronymes fréquents
    "CESEDA": "Code de l'entrée et du séjour des étrangers et du droit d'asile",
    "CGI":    "Code général des impôts",
    "CPC":    "Code de procédure civile",
    "CPP":    "Code de procédure pénale",
    "COJ":    "Code de l'organisation judiciaire",
    "CPCE":   "Code des procédures civiles d'exécution",
    "CSS":    "Code de la sécurité sociale",
    "CSP":    "Code de la santé publique",
    "CJPM":   "Code de la justice pénale des mineurs",
    # Typos fréquents (manque "de")
    "code procédure civile": "Code de procédure civile",
    "code procédure pénale": "Code de procédure pénale",
}

_ACCENT_CLASS = {
    "a": "[aàâä]", "A": "[AÀÂÄ]",
    "e": "[eéèêë]", "E": "[EÉÈÊË]",
    "i": "[iîï]",  "I": "[IÎÏ]",
    "o": "[oôö]",  "O": "[OÔÖ]",
    "u": "[uùûü]", "U": "[UÙÛÜ]",
    "c": "[cç]",   "C": "[CÇ]",
    # accent → non-accent (car l'utilisateur peut écrire sans accent)
    "é": "[eéèêë]", "È": "[EÉÈÊË]",
    "è": "[eéèêë]", "É": "[EÉÈÊË]",
    "ê": "[eéèêë]", "Ê": "[EÉÈÊË]",
    "à": "[aàâä]",  "À": "[AÀÂÄ]",
    "â": "[aàâä]",  "Â": "[AÀÂÄ]",
    "ô": "[oôö]",   "Ô": "[OÔÖ]",
    "î": "[iîï]",   "Î": "[IÎÏ]",
    "ç": "[cç]",    "Ç": "[CÇ]",
    "ù": "[uùûü]",  "Ù": "[UÙÛÜ]",
}


def _escape_code_name(name: str) -> str:
    """Construit une regex qui matche le nom d'un code en étant tolérant
    aux accents (vieux arrêts ALL CAPS sans accents) et aux différentes
    apostrophes Unicode.
    """
    parts = []
    for ch in name:
        if ch in _ACCENT_CLASS:
            parts.append(_ACCENT_CLASS[ch])
        elif ch in "'’ʼ'":
            parts.append("[’'ʼ']")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)

def build_code_patterns():
    official = {}
    for code in sorted(CODES_OFFICIELS, key=len, reverse=True):
        official[code] = re.compile(r"\b" + _escape_code_name(code) + r"\b",
                                    re.IGNORECASE)
    variantes = []
    for var_name, normalise in CODES_HISTORIQUES.items():
        pat = re.compile(r"\b" + _escape_code_name(var_name) + r"\b",
                         re.IGNORECASE)
        variantes.append((pat, normalise))
    variantes.sort(key=lambda x: len(x[0].pattern), reverse=True)
    return official, variantes

CODE_PATTERNS_OFFICIAL, CODE_PATTERNS_VARIANTES = build_code_patterns()

# ═══════════════════════════════════════════════════════════════════════
# EXTRACTEURS : BASELINE (v0)
# ═══════════════════════════════════════════════════════════════════════

ARTICLE_NUM_RE = r"([LRDAE]\.?\s*\d[\d\-\.]*\d?|\d{1,5}(?:\-\d+)*)"
ALINEA_RE      = r"(?:\s+(?:al(?:inéa)?|AL)\.?\s*(\d+))?"


def _normalize_pair_article(article_raw: str) -> str:
    """Normalise un numéro d'article. Gère :
    - préfixe L/R/D/A/E suivi ou non d'un point
    - suffixes latins 'bis/ter/quater/quinquies/...' (gardés en minuscules)
    - lettre suffixe majuscule après un latin (ex: '1649 quinquies B' → '1649quinquiesB')
    - espaces et tirets normalisés
    """
    s = article_raw.strip()
    # Détecte un latin suffix (+ lettre éventuelle)
    latin_m = re.search(r"\b(bis|ter|quater|quinquies|sexies|septies)"
                        r"(?:\s+([A-Z]))?\s*$", s, re.IGNORECASE)
    latin_suffix = ""
    if latin_m:
        latin_suffix = latin_m.group(1).lower()
        if latin_m.group(2):
            latin_suffix += latin_m.group(2).upper()
        s = s[:latin_m.start()].rstrip()
    pm = re.match(r"^([LRDAE])\.?\s*(.+)$", s, re.IGNORECASE)
    if pm:
        base = normalize_article(pm.group(1), pm.group(2))
    else:
        base = normalize_article("", s)
    return base + latin_suffix


def extract_pairs_v0(text: str) -> set[str]:
    """Baseline : reproduction exacte du regex d'enrichissement."""
    pairs = set()
    all_patterns = (
        [(pat, name) for name, pat in CODE_PATTERNS_OFFICIAL.items()]
        + [(pat, norm) for pat, norm in CODE_PATTERNS_VARIANTES]
    )
    for code_pat, code_name in all_patterns:
        p1 = re.compile(
            r"(?:articles?|arts?\.?)\s+" + ARTICLE_NUM_RE + ALINEA_RE
            + r"\s+du\s+" + code_pat.pattern,
            re.IGNORECASE,
        )
        for m in p1.finditer(text):
            pairs.add(make_pair_key(normalize_code(code_name),
                                    _normalize_pair_article(m.group(1))))
        p2 = re.compile(code_pat.pattern + r"\s+" + ARTICLE_NUM_RE + ALINEA_RE,
                        re.IGNORECASE)
        for m in p2.finditer(text):
            pairs.add(make_pair_key(normalize_code(code_name),
                                    _normalize_pair_article(m.group(1))))
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# EXTRACTEUR V1 : LISTING + SUFFIXES
# ═══════════════════════════════════════════════════════════════════════
#
# Nouvelle logique :
#   (1) Listing "articles X, Y et Z du CODE"
#       → on capture TOUS les numéros avant "du CODE".
#   (2) Suffixes optionnels après chaque numéro : ", alinéa N" / ", § N" /
#       ", paragraphe N" / "bis", "ter" — on les ignore pour le pair_key.
#   (3) Form "article X ... du CODE" avec matières intercalaires
#       (peut être long : on reste à <80 chars entre num et "du CODE").

# Un "numéro d'article" potentiellement annoté
# Inclut : num classiques ; articles "nommés" (préliminaire, liminaire) ;
# Annexes ; articles ordinaux (premier, 1er, 2e).
_NUM_TOKEN = (
    r"(?:"
    r"préliminaire|liminaire|"               # articles nommés
    r"[LRDAE]\.?\s*\d[\d\-\.]*\d?|"          # préfixe + numérique
    r"\d[\d\-\.]*\d?"                        # numérique nu
    r")"
    r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies)"
    r"(?:\s+[A-Z])?)?"                       # + lettre majuscule optionnelle
                                             #   (ex: "1649 quinquies B")
)

# Suffixe après le numéro. Plusieurs formes possibles, dans n'importe
# quel ordre, répétables. On les consomme sans les réémettre.
#   ", alinéa 2" / ", § 3" / ", paragraphe 1" / "alinéa 1er"
#   ", 3. alinéa" (inversé, majuscules)
#   ", II" (romain)
#   ", 5°" (degré / sous-section)
#   " (ancien)" / " (nouveau)"
_SUFFIX_ONE = (
    r"(?:"
        r"\s*,\s*(?:al(?:inéa)?s?\.?|§|paragraphes?)\s*"
        r"(?:[\d\w°]+|(?:\d+(?:er|e|ère|ième))|premier|dernier)"
    r"|"
        r"\s*,\s*[IVX]+\b"
    r"|"
        r"\s*,\s*\d+\s*\.\s*(?:alinéa|ALINEA)"
    r"|"
        r"\s*,\s*\d+°"
    r"|"
        r"\s*\(\s*(?:ancien|nouveau)\s*\)"
    r"|"
        r"\s+(?:ancien|nouveau)\b"            # sans parenthèses
    r"|"
        r"\s+et\s+suivants?\b"                # terminateur "et suivants"
    r")"
)
_SUFFIX = rf"(?:{_SUFFIX_ONE})*"


_LIST_ITEM_EXTRACT = re.compile(
    r"(?:^|(?<=[,\s]))"
    r"(?:"
        r"(?:préliminaire|liminaire)|"
        r"(?:[LRDAE]\.?\s*)?\d[\d\-\.]*\d?"
        r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies)"
        r"(?:\s+[A-Z])?)?"                        # + lettre optionnelle
    r")"
    r"(?=\s*(?:,|\s+et\s+|\s+[àa]\s+|$|\s*\.|\s+\w))",
    re.IGNORECASE,
)


def _emit_numbers_from_listing(listing_raw: str) -> list[str]:
    """Extrait tous les numéros d'articles d'un listing 'X, Y et Z alinéa...'.

    Stratégie : utilise finditer pour matcher itérativement les tokens
    d'articles, en **ignorant** les suffixes "alinéa/§/paragraphe/romain"
    qui sont internes au LIST_ITEM pattern global.
    """
    # Retire d'abord les blocs suffixes pour éviter qu'ils soient pris
    # pour des tokens (ex: ", 3. ALINEA" ne doit pas produire "3").
    cleaned = re.sub(
        r",\s*(?:al(?:inéa)?s?\.?|§|paragraphes?)\s*"
        r"(?:[\d\w°]+|(?:\d+(?:er|e|ère|ième))|premier|dernier)",
        " ",
        listing_raw,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r",\s*[IVX]+\b", " ", cleaned)
    cleaned = re.sub(r",\s*\d+\s*\.\s*(?:alinéa|ALINEA)", " ", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*\d+°", " ", cleaned)
    cleaned = re.sub(r"\(\s*(?:ancien|nouveau)\s*\)", " ", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:ancien|nouveau)\b", " ", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+et\s+suivants?\b", " ", cleaned,
                     flags=re.IGNORECASE)
    nums = []
    for m in _LIST_ITEM_EXTRACT.finditer(cleaned):
        tok = m.group(0).strip(" .;:")
        if tok:
            nums.append(tok)
    return nums


def extract_pairs_v1(text: str) -> set[str]:
    """V1 — listing + suffixes + intercalaires courts.

    Forme 1 : "article(s) <listing> du CODE"
    Forme 2 : "CODE <listing>" (rare en pratique)
    """
    pairs = set()
    all_patterns = (
        [(pat, name) for name, pat in CODE_PATTERNS_OFFICIAL.items()]
        + [(pat, norm) for pat, norm in CODE_PATTERNS_VARIANTES]
    )

    # Regex générique qui capture un listing devant "du CODE".
    # Le listing est : suite de (num+suffixe) séparés par ", " ou " et ".
    LIST_ITEM = _NUM_TOKEN + _SUFFIX
    # Séparateurs autorisés : ',' / ' et ' / ' à ' / espace seul si le
    # token suivant ressemble à un article (L./R./D./A./E. ou digit).
    # Ce dernier couvre les listings malformés avec virgule oubliée
    # (ex. "L. 433-1, L. 433-2 R. 432-2, R. 433-1...").
    _SEP = (r"(?:\s*,\s*"
            r"|\s+et\s+(?:les\s+articles?\s+|l['’]\s*articles?\s+)?"
            r"|\s+[àa]\s+"
            r"|\s+(?=[LRDAE]\.?\s*\d))")
    LISTING   = rf"(?:{LIST_ITEM})(?:{_SEP}{LIST_ITEM})*"

    for code_pat, code_name in all_patterns:
        # Forme 1 : "articles <listing> [intercalaire court] [,] du CODE"
        # Intercalaire toléré jusqu'à 30 chars mais interdit d'y trouver
        # un autre "code/loi/décret/convention" pour éviter la contamination
        # "1134 du code civil et L.1243-8 du code du travail".
        p1 = re.compile(
            rf"(?:articles?|arts?\.?)\s+({LISTING})"
            rf"(?:\s+(?!(?:du|de\s+la|de\s+l['’])\s+"
            rf"(?:nouveau\s+|ancien\s+|présent\s+|present\s+|"
            rf"même\s+|meme\s+|dit\s+|ledit\s+)?"
            rf"(?:code|loi|décret|decret|convention|règlement|reglement))"
            rf"[^,;\n]{{0,40}}?)?"
            rf"(?:\s*,)?\s+du\s+(?:(?:nouveau|ancien)\s+)?"
            + code_pat.pattern,
            re.IGNORECASE,
        )
        for m in p1.finditer(text):
            for num in _emit_numbers_from_listing(m.group(1)):
                pairs.add(make_pair_key(normalize_code(code_name),
                                        _normalize_pair_article(num)))

        # Forme 2 : "CODE <num>"
        p2 = re.compile(code_pat.pattern + r"\s+" + ARTICLE_NUM_RE + ALINEA_RE,
                        re.IGNORECASE)
        for m in p2.finditer(text):
            pairs.add(make_pair_key(normalize_code(code_name),
                                    _normalize_pair_article(m.group(1))))

        # Forme 3 : structure inversée "CODE … (en ses | notamment)?
        #                                  articles <listing>"
        # CESEDA style : "Vu le Code de l'Entrée… notamment en ses
        # articles L. 741-1 et suivants".
        # On autorise jusqu'à 80 chars entre le code et "articles" pour
        # englober "notamment en ses", ", en ses", ", dans ses", etc.
        p3 = re.compile(
            code_pat.pattern
            + r"(?:[,\s](?:notamment|en\s+ses|dans\s+ses|pris\s+en\s+ses"
              r"|lu\s+ensemble)?[^.;]{0,80}?)?"
            + rf"\s+(?:articles?|arts?\.?)\s+({LISTING})"
              r"(?![^,;\n]{0,40}\s+du\s+(?:nouveau\s+|ancien\s+)?code)",
            re.IGNORECASE,
        )
        for m in p3.finditer(text):
            for num in _emit_numbers_from_listing(m.group(1)):
                pairs.add(make_pair_key(normalize_code(code_name),
                                        _normalize_pair_article(num)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# EXTRACTEUR V2 : V1 + VARIANTES ORTHOGRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════
# Déjà gérées par _escape_code_name (apostrophes Unicode) et
# CODES_HISTORIQUES (nouveau/ancien CPC). On laisse un hook pour ajouter
# des variantes de casse/accent si besoin.

def extract_pairs_v2(text: str) -> set[str]:
    return extract_pairs_v1(text)


# ═══════════════════════════════════════════════════════════════════════
# EXTRACTEUR V3 : V2 + ANAPHORE DE CODE
# ═══════════════════════════════════════════════════════════════════════
#
# Heuristique : quand un paragraphe cite "le code X" ou "du code X", puis
# mentionne "article N" sans code ensuite dans une fenêtre courte
# (même paragraphe ou < 500 chars), on propage le code.

# Marqueurs d'anaphore explicite : "du même code", "dudit code", "de ce
# code", "du code précité", "ledit code". Le regex capture le num d'article
# AVANT le marqueur.
_ANAPHORA_RE = re.compile(
    r"\b(?:articles?|arts?\.?)\s+(" + _NUM_TOKEN + _SUFFIX + r")\s+"
    r"(?:du\s+(?:même|dit)\s+code"
    r"|du\s+code\s+(?:précité|susvisé)"
    r"|de\s+ce\s+code"
    r"|dudit\s+code"
    r"|ledit\s+code)",
    re.IGNORECASE,
)


def _find_last_code_before(offset: int, text: str,
                           window: int = 800) -> str | None:
    """Cherche le dernier code officiel mentionné dans les `window` chars
    avant `offset`.
    """
    start = max(0, offset - window)
    chunk = text[start:offset]
    best = None
    best_pos = -1
    for code_pat, code_name in (
        [(p, n) for n, p in CODE_PATTERNS_OFFICIAL.items()]
        + [(p, n) for p, n in CODE_PATTERNS_VARIANTES]
    ):
        for m in code_pat.finditer(chunk):
            if m.start() > best_pos:
                best_pos = m.start()
                best = code_name
    return best


def extract_pairs_v3(text: str) -> set[str]:
    """V3 = V2 + anaphore **explicite uniquement**.

    On ne propage le code que si l'anaphore est marquée textuellement :
    "du même code", "dudit code", "de ce code", "du code précité/susvisé".
    Cette restriction évite les faux positifs massifs quand un article
    ubiquitaire (ex. '700 du code de procédure civile') apparaît en amont
    et serait propagé à tort.
    """
    pairs = extract_pairs_v2(text)
    for m in _ANAPHORA_RE.finditer(text):
        listing = m.group(1)
        code_name = _find_last_code_before(m.start(), text)
        if not code_name:
            continue
        for num in _emit_numbers_from_listing(listing):
            pairs.add(make_pair_key(normalize_code(code_name),
                                    _normalize_pair_article(num)))
    return pairs


EXTRACTORS = {
    "v0": extract_pairs_v0,
    "v1": extract_pairs_v1,
    "v2": extract_pairs_v2,
    "v3": extract_pairs_v3,
}


# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════

def load_sample() -> dict[str, dict]:
    recs = {}
    with open(SAMPLE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


_LLM_NOISY_SUFFIX = re.compile(
    r",(?:§|paragraphe|alinéa|ancien|II+|IV|VI+)"
    r"[\w\s°]*\)?",
    re.IGNORECASE,
)

def _normalize_llm_pair_key(pk: str) -> str:
    """Aligne la convention LLM sur celle du regex pour comparaison équitable.

    - Retire préfixes 'nouveau_' / 'ancien_' du code slug (le regex les
      mappe vers la version en vigueur via CODES_HISTORIQUES).
    - Retire suffixes bruyants dans la partie article :
        ',alinéa2', ',paragraphe9', ',II', ',§3,c)', 'ancien'.
    - Retire tiret traînant ('R4323-' → 'R4323').
    """
    if ":" not in pk:
        return pk
    code, art = pk.split(":", 1)
    # préfixes historiques
    for prefix in ("nouveau_", "ancien_"):
        if code.startswith(prefix):
            code = code[len(prefix):]
    # nettoyage article
    art = _LLM_NOISY_SUFFIX.sub("", art)
    art = art.rstrip("-")
    # strip trailing 'ancien' sans virgule (L135-2ancien → L135-2)
    art = re.sub(r"(ancien|nouveau)$", "", art, flags=re.IGNORECASE)
    return f"{code}:{art}"


def load_fn_targets() -> dict[str, set[str]]:
    """Charge les pair_keys que le LLM a trouvés mais pas le regex baseline.

    Normalise les pair_keys LLM pour retirer les "pseudo-FN" dus à des
    différences de convention de slug (LLM met 'nouveau_', 'L20,II', etc.).

    Ce sont nos *cibles* : ce que la nouvelle regex doit récupérer.
    """
    targets = defaultdict(set)
    with open(FN_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            targets[row["id"]].add(_normalize_llm_pair_key(row["pair_key"]))
    return dict(targets)


def load_regex_baseline(recs: dict[str, dict]) -> dict[str, set[str]]:
    """Pour chaque arrêt, récupère la sortie regex baseline depuis
    `code_article_pairs` déjà calculés en enrichissement.
    """
    out = {}
    for rid, r in recs.items():
        pairs = set(r.get("code_article_pairs") or [])
        out[rid] = pairs
    return out


# ═══════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(version: str,
             recs: dict,
             regex_baseline: dict,
             fn_targets: dict) -> dict:
    """Applique extractor[version] aux 100 arrêts et calcule les métriques
    par rapport à (baseline ∪ FN) = LLM-truth approximée.
    """
    extractor = EXTRACTORS[version]

    # Ensemble cible par arrêt : ce que le LLM a trouvé
    #   = (regex baseline ∩ LLM) ∪ FN  approx  (regex baseline ∪ FN) − FP
    #   → on ne connaît pas les FP, donc on utilise:
    #       LLM_truth ≈ baseline_TP ∪ FN
    #     baseline_TP = baseline_pks − FP
    #     Comme FP n'est pas dispo, on approxime: LLM_truth ≈ baseline ∪ FN
    # Cela surestime légèrement la cible pour les arrêts avec FP, mais l'erreur
    # est bornée (57 FP globaux sur 486 vrais positifs LLM).
    totals = {"fn_recovered": 0, "fn_total": 0,
              "new_pairs": 0, "baseline_size": 0, "new_size": 0}
    per_record = []

    for rid, r in recs.items():
        text = r.get("text") or ""
        baseline = regex_baseline[rid]
        fn = fn_targets.get(rid, set())
        llm_truth = baseline | fn  # approximation
        new_pairs = extractor(text)

        recovered = (new_pairs & fn)          # FN récupérés
        extra     = new_pairs - llm_truth     # potentiels nouveaux FP
        missed    = fn - new_pairs            # FN encore manqués

        totals["fn_total"]      += len(fn)
        totals["fn_recovered"]  += len(recovered)
        totals["new_pairs"]     += len(extra)
        totals["baseline_size"] += len(baseline)
        totals["new_size"]      += len(new_pairs)

        per_record.append({
            "id": rid,
            "jur": r.get("_jurisdiction") or "?",
            "n_baseline": len(baseline),
            "n_new":      len(new_pairs),
            "n_fn":       len(fn),
            "n_recovered": len(recovered),
            "n_extra":    len(extra),
            "missed":     sorted(missed),
            "extra":      sorted(extra),
        })

    return {"totals": totals, "per_record": per_record}


def print_eval(version: str, ev: dict, compare_to: dict | None = None) -> None:
    t = ev["totals"]
    print(f"\n╔{'═'*68}╗")
    print(f"║ {version.upper():<66} ║")
    print(f"╚{'═'*68}╝")
    print(f"  FN récupérés    : {t['fn_recovered']:4d} / {t['fn_total']:4d}"
          f"  ({100*t['fn_recovered']/max(1,t['fn_total']):.1f}%)")
    print(f"  Nouveaux pairs  : {t['new_pairs']:4d}  "
          f"(potentiels FP, à inspecter)")
    print(f"  Taille regex out: {t['new_size']:5d}  "
          f"(baseline {t['baseline_size']})")

    if compare_to is not None:
        d_rec = t['fn_recovered'] - compare_to['totals']['fn_recovered']
        d_ext = t['new_pairs']    - compare_to['totals']['new_pairs']
        sign = lambda x: f"+{x}" if x >= 0 else str(x)
        print(f"  Δ vs précédent  : récup {sign(d_rec)}  · extras {sign(d_ext)}")


def print_missed_samples(ev: dict, recs: dict, n: int = 20) -> None:
    """Affiche quelques FN encore manqués après itération, avec contexte."""
    print(f"\n┌ FN encore manqués (échantillon {n}) ┐")
    shown = 0
    for rec in ev["per_record"]:
        if shown >= n:
            break
        if not rec["missed"]:
            continue
        text = recs[rec["id"]].get("text") or ""
        for pk in rec["missed"][:3]:
            # trouve un contexte
            code_slug, art = pk.split(":", 1)
            code_readable = code_slug.replace("_", " ")
            print(f"\n  [{rec['jur']}] {rec['id']} — {pk}")
            # cherche l'article dans le texte
            # prend le 1er num "mot" du pair_key
            num_search = re.sub(r"^[LRDAE]", "", art)
            needle = re.search(rf"\b{re.escape(num_search)}\b", text)
            if needle:
                s = max(0, needle.start() - 80)
                e = min(len(text), needle.end() + 80)
                snippet = text[s:e].replace("\n", " ")
                print(f"       …{snippet}…")
            shown += 1


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(EXTRACTORS)+["all"],
                    default="all")
    ap.add_argument("--show-missed", type=int, default=0)
    ap.add_argument("--show-extra", type=int, default=0,
                    help="échantillon de nouveaux pairs (potentiels FP)")
    args = ap.parse_args()

    recs           = load_sample()
    regex_baseline = load_regex_baseline(recs)
    fn_targets     = load_fn_targets()

    print(f"Loaded : {len(recs)} arrêts   ·  {sum(len(v) for v in fn_targets.values())} FN LLM")

    versions = list(EXTRACTORS) if args.version == "all" else [args.version]
    prev = None
    for v in versions:
        ev = evaluate(v, recs, regex_baseline, fn_targets)
        print_eval(v, ev, prev)
        prev = ev

    if args.show_missed:
        print_missed_samples(prev, recs, args.show_missed)

    if args.show_extra:
        print(f"\n┌ Nouveaux pairs introduits (échantillon {args.show_extra}) ┐")
        shown = 0
        for rec in prev["per_record"]:
            if shown >= args.show_extra: break
            for pk in rec["extra"][:3]:
                print(f"  [{rec['jur']}] {rec['id']} — {pk}")
                shown += 1
                if shown >= args.show_extra: break


if __name__ == "__main__":
    main()
