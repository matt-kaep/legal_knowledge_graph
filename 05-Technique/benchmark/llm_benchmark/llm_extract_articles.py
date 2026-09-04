"""Extraction normée d'articles de codes juridiques via LLM.

Objectif : comparer regex V3 (baseline symbolique) à plusieurs LLMs open-weights.
Pour que la comparaison soit valide, le prompt impose :
  - la liste exacte des codes autorisés (slug canonique) ;
  - un schéma JSON strict (validé par vLLM via response_format.json_schema) ;
  - un format d'article compact `(L|R|D|A|E)?\\d[\\d\\-]*(bis|ter|…)?[A-Z]?`.

Le post-traitement (normalize_code / normalize_article) est identique à celui
du regex → on finit par comparer des sets de pair_keys canoniques.

Usage :
    from llm_extract_articles import extract_pairs_llm, LLM_SYSTEM_PROMPT
    pairs = extract_pairs_llm(text, client, model_id="mistralai/Ministral-8B")
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from iterate_regex import (
    CODES_OFFICIELS, CODES_HISTORIQUES,
    normalize_code, normalize_article, make_pair_key,
)

# ═══════════════════════════════════════════════════════════════════════
# TABLE DES CODES AUTORISÉS : slug canonique → nom lisible
# Fournie au LLM pour qu'il choisisse UN slug dans la liste.
# ═══════════════════════════════════════════════════════════════════════

CODE_SLUGS: dict[str, str] = {}
for code in CODES_OFFICIELS:
    CODE_SLUGS[normalize_code(code)] = code

# Plus les variantes historiques (Nouveau CPC → CPC etc.) : on ne les expose
# PAS au LLM (on veut la cible canonique seulement) — mais on les tolère
# dans le post-processing.
HISTORIQUE_MAP: dict[str, str] = {
    normalize_code(k): normalize_code(v)
    for k, v in CODES_HISTORIQUES.items()
}

# Pour prompt : liste triée slug → nom (lisible humain) + exemples
CODE_SLUG_BLOCK = "\n".join(
    f"- {slug}  ({name})"
    for slug, name in sorted(CODE_SLUGS.items())
)


# ═══════════════════════════════════════════════════════════════════════
# PROMPT NORMÉ — identique pour tous les LLMs
# ═══════════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT = f"""Tu es un extracteur d'articles de codes juridiques français.

## TÂCHE
On te donne le texte brut d'un arrêt (Cour de cassation, cour d'appel ou tribunal).
Tu dois extraire TOUS les articles de CODES OFFICIELS cités dans ce texte, en
respectant strictement le format de sortie demandé.

## RÈGLES IMPÉRATIVES

1. **Inclure** uniquement les articles de codes figurant dans la liste ci-dessous.
   **EXCLURE** :
   - les lois et décrets (ex. "loi du 6 juillet 1989", "décret n°2004-...")
   - les conventions collectives
   - les règlements européens et directives
   - les codes étrangers
   - toute référence qui n'est pas un article de code officiel français

2. **Listings** ("articles X, Y et Z" / "articles X à Z") : émettre CHAQUE article.
   Pour un range "X à Z", émettre TOUS les intermédiaires si l'énumération est
   évidente (ex. "L743-6 à L743-8" → L743-6, L743-7, L743-8).

3. **Anaphores** ("du même code", "dudit code", "du code précité") : rattacher
   au dernier code explicitement cité avant l'anaphore.

4. **Suffixes latins** (bis, ter, quater, quinquies, sexies, septies) : les
   conserver dans le numéro d'article, collés (ex. "1649 quinquies B" → "1649quinquiesB").

5. **Ne rien inventer** : si un article n'apparaît pas littéralement, ne pas
   l'émettre.

6. **Dédupliquer** : chaque couple (code, article) ne doit apparaître qu'une fois.

## FORMAT DE L'ARTICLE (chaîne compacte)
- Préfixe optionnel en majuscule : `L`, `R`, `D`, `A`, `E` (collé au numéro)
- Numéro : chiffres avec tirets internes éventuels (pas d'espace, pas de point)
- Suffixe latin éventuel, minuscule, collé (ex. `bis`, `quinquies`)
- Lettre finale éventuelle, majuscule (ex. `B` dans `1649quinquiesB`)

Exemples valides : `1134`, `L742-1`, `R213-12-2`, `1649quinquiesB`, `700`, `L.1243-8` ❌ (supprimer le point).

## LISTE DES CODES AUTORISÉS (slug canonique → nom)

Choisis UNIQUEMENT dans la liste suivante. Si un code cité n'y figure pas,
NE PAS l'émettre.

{CODE_SLUG_BLOCK}

## FORMAT DE SORTIE (JSON strict)

Émets un objet JSON avec un seul champ `pairs` contenant une liste d'objets
{{"code_slug": "...", "article": "..."}}. Pas de champ additionnel, pas de
commentaire, pas de markdown.

## EXEMPLES

Texte : "Vu les articles 1134 du code civil et L. 1243-8 du code du travail ;"
→ {{"pairs": [
     {{"code_slug": "code_civil", "article": "1134"}},
     {{"code_slug": "code_du_travail", "article": "L1243-8"}}
   ]}}

Texte : "Vu l'article L743-6 à L743-8 du code de l'entrée et du séjour des étrangers
et du droit d'asile ; vu l'article L742-1 du même code ;"
→ {{"pairs": [
     {{"code_slug": "code_de_l_entree_et_du_sejour_des_etrangers_et_du_droit_d_asile", "article": "L743-6"}},
     {{"code_slug": "code_de_l_entree_et_du_sejour_des_etrangers_et_du_droit_d_asile", "article": "L743-7"}},
     {{"code_slug": "code_de_l_entree_et_du_sejour_des_etrangers_et_du_droit_d_asile", "article": "L743-8"}},
     {{"code_slug": "code_de_l_entree_et_du_sejour_des_etrangers_et_du_droit_d_asile", "article": "L742-1"}}
   ]}}

Texte : "ARTICLE 1649 QUINQUIES B DU CODE GENERAL DES IMPOTS"
→ {{"pairs": [
     {{"code_slug": "code_general_des_impots", "article": "1649quinquiesB"}}
   ]}}
"""


LLM_USER_TEMPLATE = """Voici le texte de l'arrêt. Extrais tous les articles de codes officiels cités.

<texte>
{text}
</texte>

Réponds avec le JSON strict attendu uniquement."""


# ═══════════════════════════════════════════════════════════════════════
# SCHÉMA JSON (pour vLLM response_format)
# ═══════════════════════════════════════════════════════════════════════

LLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code_slug": {"type": "string"},
                    "article":   {"type": "string"},
                },
                "required": ["code_slug", "article"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


# ═══════════════════════════════════════════════════════════════════════
# POST-PROCESSING : parse + normalise → set[pair_key]
# ═══════════════════════════════════════════════════════════════════════

_ART_FORMAT_RE = re.compile(
    r"^(?P<prefix>[LRDAE])?"
    r"(?P<num>\d[\d\-]*\d?|\d)"
    r"(?P<suffix>bis|ter|quater|quinquies|sexies|septies)?"
    r"(?P<letter>[A-Z])?$"
)

def _canonicalize_article(raw: str) -> str | None:
    """Nettoie un article émis par LLM et renvoie la forme canonique
    (cf. normalize_article côté regex). None si parsing impossible."""
    if not raw:
        return None
    s = raw.strip()
    # Supprime espaces, points, apostrophes
    s = re.sub(r"[.\s]", "", s)
    # Supprime "art" / "article" éventuels laissés en tête
    s = re.sub(r"^(?:article|art)\.?", "", s, flags=re.I)
    # Majuscule du préfixe
    m = re.match(r"^([lrdae])", s)
    if m:
        s = s[0].upper() + s[1:]
    # Suffixe latin éventuel : normaliser minuscule
    s = re.sub(
        r"(bis|ter|quater|quinquies|sexies|septies)",
        lambda m: m.group(1).lower(),
        s, flags=re.I,
    )
    if _ART_FORMAT_RE.match(s):
        return s
    # Fallback : tentative via normalize_article avec préfixe extrait
    m2 = re.match(r"^([LRDAE])?(\d[\d\-]*)(.*)$", s)
    if m2:
        prefix = m2.group(1) or ""
        num    = m2.group(2)
        rest   = m2.group(3) or ""
        out = normalize_article(prefix, num)
        if rest:
            out += rest.lower()
        return out
    return None


def _canonicalize_slug(raw: str) -> str | None:
    """Renvoie le slug canonique ou None si non reconnu."""
    if not raw:
        return None
    s = normalize_code(raw)
    if s in CODE_SLUGS:
        return s
    # Variante historique → canonique
    if s in HISTORIQUE_MAP:
        return HISTORIQUE_MAP[s]
    # Match partiel (le LLM peut avoir légèrement renommé)
    for slug in CODE_SLUGS:
        if s == slug:
            return slug
    return None


def parse_llm_output(raw: str) -> set[str]:
    """Parse la sortie JSON du LLM et renvoie un set de pair_keys
    canoniques. Tolérant aux petits écarts (JSON en markdown, slug inconnu).
    """
    if not raw:
        return set()
    # Récupère le premier objet JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return set()
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set()
    pairs = data.get("pairs") or []
    out: set[str] = set()
    for item in pairs:
        if not isinstance(item, dict):
            continue
        slug = _canonicalize_slug(item.get("code_slug", ""))
        art  = _canonicalize_article(item.get("article", ""))
        if slug and art:
            out.add(make_pair_key(slug, art))
    return out


# ═══════════════════════════════════════════════════════════════════════
# APPEL vLLM (interface OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════

def call_llm(
    text: str,
    client: Any,
    model_id: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Renvoie (raw_json_string, meta). meta contient latency_s, tokens_used,
    finish_reason, error."""
    t0 = time.time()
    meta = {"latency_s": None, "tokens_used": None,
            "finish_reason": None, "error": None}
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": LLM_USER_TEMPLATE.format(text=text)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ArticleExtraction",
                    "schema": LLM_OUTPUT_SCHEMA,
                    "strict": True,
                },
            },
        )
        raw = resp.choices[0].message.content or ""
        meta["latency_s"]     = round(time.time() - t0, 3)
        meta["tokens_used"]   = resp.usage.completion_tokens if resp.usage else None
        meta["finish_reason"] = resp.choices[0].finish_reason
        if meta["finish_reason"] == "length":
            meta["error"] = f"Coupé par max_tokens={max_tokens}"
        return raw, meta
    except Exception as e:
        meta["latency_s"] = round(time.time() - t0, 3)
        meta["error"]     = f"{type(e).__name__}: {e}"
        return "", meta


def extract_pairs_llm(
    text: str,
    client: Any,
    model_id: str,
    max_tokens: int = 2048,
) -> tuple[set[str], dict]:
    """Pipeline complet : appel LLM + parse → set[pair_key] + méta."""
    raw, meta = call_llm(text, client, model_id, max_tokens=max_tokens)
    meta["raw_response"] = raw
    pairs = parse_llm_output(raw)
    meta["n_pairs"] = len(pairs)
    return pairs, meta
