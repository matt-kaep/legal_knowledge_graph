"""
Préparation des données pour le cluster GPU.

À exécuter LOCALEMENT (avec accès aux fichiers Judilibre ~7 GB).
Produit un dossier cluster_data/ auto-contenu (~quelques MB) à transférer sur le cluster.

Usage :
    cd 05-Technique/benchmark
    python prepare_cluster_data.py

Produit :
    cluster_data/m6_gold_cases.json   — 5 cas gold M6 avec décisions complètes
    cluster_data/m1_full.json         — ~2670 questions M1 depuis Les-Audits-Affaires
                                        (le notebook échantillonne N au hasard à l'exécution)
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# Regex : capture la référence d'article seule (sans nom du code, trop variable en FR).
# Exemples : "article L. 442-1", "art. 1641", "article 302 septies A", "article L2141-2"
_ARTICLE_RE = re.compile(
    r"(?:article|art\.)\s+"
    r"(?:L\.?\s*)?"                                                   # préfixe L. optionnel
    r"\d+(?:[-–]\d+)*"                                                # numéro : 1641, 122-5, L2141-2
    r"(?:\s+(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?"  # suffixes latins
    r"(?:\s+[A-Z](?=\s|$|[,.;)]))?",                                  # lettre seule "A" avec lookahead
    flags=re.IGNORECASE,
)


def _extract_articles(text: str) -> list[str]:
    """Extrait les références d'articles citées dans un texte juridique."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _ARTICLE_RE.finditer(text):
        ref = " ".join(m.group(0).split())  # normalise les espaces
        key = ref.lower()
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return out

OUTDIR = Path(__file__).parent / "cluster_data"
OUTDIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# M6 — Export des cas gold (nécessite les fichiers Judilibre locaux)
# ══════════════════════════════════════════════════════════════════════


def export_m6_gold_cases() -> Path:
    """Charge les 5 cas gold M6 et les exporte en JSON auto-contenu."""
    print("Chargement des cas M6 gold (peut prendre ~1-2 min — lecture Judilibre)…")
    from m6_mvp.gold_cases import get_all_cases

    cases = get_all_cases()
    out = []
    for c in cases:
        d = c.decision
        s = d.structure

        out.append({
            "id": c.id,
            "specialisation": c.specialisation.value,
            "difficulty": c.difficulty.value,
            "case": {
                "case_summary": c.case.case_summary,
                "client_position": c.case.client_position.value,
                "key_facts": c.case.key_facts,
            },
            "question": c.question,
            "decision": {
                "id": d.id,
                "juridiction": d.juridiction.value,
                "chambre": d.chambre,
                "date": str(d.date),
                "numero": d.numero,
                "full_text": d.full_text,
                "structure": {
                    "faits": s.faits if s else None,
                    "moyens": s.moyens if s else None,
                    "motifs": s.motifs if s else None,
                    "dispositif": s.dispositif if s else None,
                } if s else None,
            },
            "gold": {
                "camp_in_decision": c.gold.camp_in_decision,
                "sens_arret": c.gold.sens_arret.value,
                "is_favorable": c.gold.is_favorable,
                "dispositif_summary": c.gold.dispositif_summary,
                "relevance": c.gold.relevance,
                "principles_extracted": c.gold.principles_extracted,
                "transfer_reasoning": c.gold.transfer_reasoning,
            },
            "notes": c.notes,
        })

    path = OUTDIR / "m6_gold_cases.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ {len(out)} cas exportés → {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


# ══════════════════════════════════════════════════════════════════════
# M1 — Export d'un échantillon Les-Audits-Affaires
# ══════════════════════════════════════════════════════════════════════

_M1_GOLD_OVERRIDES: dict[str, dict] = {}  # {id: {"gold_articles": [...], "gold_keywords": [...]}}


def export_m1_full(n: int | None = None) -> Path:
    """Charge Les-Audits-Affaires et exporte les questions en JSON.

    n=None  → export COMPLET (~2670 cas) vers m1_full.json
              Le notebook échantillonnera N au hasard à l'exécution.
    n=<int> → échantillon stratifié de N cas vers m1_sample.json (legacy).
    """
    print(f"Chargement Les-Audits-Affaires (HuggingFace, ~50 MB)…")
    try:
        import pandas as pd

        df = pd.read_parquet(
            "hf://datasets/legmlai/les-audits-affaires/data/train-00000-of-00001.parquet"
        )
        print(f"  Dataset chargé : {len(df)} lignes, colonnes : {list(df.columns)}")

        # LegMLAI : la "réponse" est éclatée en 5 dimensions
        answer_dims = [
            "action_requise",
            "delai_legal",
            "documents_obligatoires",
            "impact_financier",
            "consequences_non_conformite",
        ]
        cat_col = "scenario_type" if "scenario_type" in df.columns else None

        if n is None:
            # Export complet — on garde toutes les lignes dans l'ordre d'origine.
            # Le notebook appliquera son propre seed pour l'échantillonnage.
            sample = df.reset_index(drop=True)
            outfile = "m1_full.json"
        else:
            # Échantillon stratifié si scenario_type dispo
            if cat_col and df[cat_col].nunique() >= 5:
                sample = df.groupby(cat_col, group_keys=False).apply(
                    lambda x: x.sample(min(2, len(x)), random_state=42)
                ).head(n)
            else:
                sample = df.sample(n=min(n, len(df)), random_state=42)
            sample = sample.reset_index(drop=True)
            outfile = "m1_sample.json"

        # Padding de l'id sur 5 chiffres pour supporter jusqu'à 99 999 cas
        id_width = 5 if n is None else 3

        out = []
        for i, row in sample.iterrows():
            question_text = str(row["question"])
            answer_parts = [
                f"[{dim.upper()}] {row[dim]}"
                for dim in answer_dims if dim in df.columns and pd.notna(row[dim])
            ]
            answer_text = "\n\n".join(answer_parts)
            category = str(row[cat_col]) if cat_col else "inconnu"
            gold_articles = _extract_articles(answer_text)

            record = {
                "id": f"M1-S-{i+1:0{id_width}d}",
                "specialisation": category,
                "question": question_text,
                "gold_answer": answer_text,
                "gold_articles": gold_articles,
                "gold_keywords": _extract_keywords(answer_text),
            }
            if record["id"] in _M1_GOLD_OVERRIDES:
                record.update(_M1_GOLD_OVERRIDES[record["id"]])
            out.append(record)

    except Exception as e:
        print(f"  ⚠ Impossible de charger Les-Audits depuis HF : {e}")
        print("  → Génération d'un échantillon statique de secours…")
        out = _static_m1_fallback()
        outfile = "m1_sample.json"

    path = OUTDIR / outfile
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ {len(out)} questions exportées → {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


# Alias legacy (ancien nom)
export_m1_sample = export_m1_full


def _extract_keywords(text: str, max_kw: int = 5) -> list[str]:
    """Extrait grossièrement les mots-clés juridiques d'un texte gold."""
    stopwords = {"le", "la", "les", "de", "du", "des", "en", "et", "ou", "un", "une",
                 "il", "elle", "est", "que", "qui", "sur", "par", "à", "au", "avec",
                 "pour", "pas", "ne", "se", "si", "ce", "cette", "car", "donc",
                 "parce", "doit", "être", "sont", "peut", "cette", "selon", "dans"}
    # Retire les marqueurs de section [ACTION_REQUISE] et les citations d'articles
    cleaned = re.sub(r"\[[A-Z_]+\]", " ", text)
    cleaned = _ARTICLE_RE.sub(" ", cleaned)
    words = [w.strip(".,;:()\"'") for w in cleaned.lower().split() if len(w) > 5]
    kw = [w for w in words if w not in stopwords and not w.startswith("[")]
    # On dé-duplique en gardant l'ordre
    seen: set[str] = set()
    unique = []
    for w in kw:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_kw]


def _static_m1_fallback() -> list[dict]:
    """Questions M1 statiques si Les-Audits-Affaires n'est pas disponible."""
    return [
        {
            "id": "M1-S-001",
            "specialisation": "Droit Social",
            "question": "Quelles sont les conditions de validité d'une clause de non-concurrence dans un contrat de travail ?",
            "gold_answer": (
                "La clause de non-concurrence est valide si elle est limitée dans le temps et dans l'espace, "
                "qu'elle protège les intérêts légitimes de l'entreprise et qu'elle comporte une contrepartie "
                "financière pour le salarié. À défaut d'une seule de ces conditions, la clause est nulle."
            ),
            "gold_articles": ["L. 1121-1 Code du travail"],
            "gold_keywords": ["non-concurrence", "contrepartie", "limitée", "espace", "temps"],
        },
        {
            "id": "M1-S-002",
            "specialisation": "Droit Civil",
            "question": "Quelles sont les conditions d'application de la garantie des vices cachés ?",
            "gold_answer": (
                "L'acheteur peut invoquer la garantie des vices cachés (art. 1641 C. civ.) si le vice était "
                "antérieur à la vente, caché (non apparent), rédhibitoire (rendant la chose inutilisable ou "
                "en diminuant fortement l'usage). L'action se prescrit en 2 ans à compter de la découverte du vice."
            ),
            "gold_articles": ["1641 Code civil", "1644 Code civil", "1648 Code civil"],
            "gold_keywords": ["vice", "caché", "antérieur", "rédhibitoire", "prescription"],
        },
        {
            "id": "M1-S-003",
            "specialisation": "Droit Pénal",
            "question": "Quelles sont les conditions légales de la légitime défense en droit pénal français ?",
            "gold_answer": (
                "La légitime défense (art. 122-5 C. pén.) suppose une agression injustifiée, actuelle ou imminente, "
                "et une riposte nécessaire et proportionnée. L'excès de riposte exclut la cause d'irresponsabilité."
            ),
            "gold_articles": ["122-5 Code pénal", "122-6 Code pénal"],
            "gold_keywords": ["légitime défense", "proportionnée", "agression", "imminente", "nécessaire"],
        },
        {
            "id": "M1-S-004",
            "specialisation": "Droit Commercial",
            "question": "Quand y a-t-il rupture brutale de relations commerciales établies au sens de l'article L. 442-1, II du Code de commerce ?",
            "gold_answer": (
                "La rupture brutale suppose l'existence d'une relation commerciale stable, ancienne et significative, "
                "une rupture sans préavis écrit suffisant eu égard à la durée de la relation. La durée minimale du "
                "préavis s'apprécie selon l'usage du secteur, sans pouvoir dépasser 18 mois."
            ),
            "gold_articles": ["L. 442-1 II Code de commerce"],
            "gold_keywords": ["rupture brutale", "relations commerciales", "préavis", "durée"],
        },
        {
            "id": "M1-S-005",
            "specialisation": "Droit de la Famille",
            "question": "Comment est fixée la prestation compensatoire en cas de divorce ?",
            "gold_answer": (
                "La prestation compensatoire (art. 270-271 C. civ.) vise à compenser la disparité créée par la rupture "
                "dans les conditions de vie. Le juge tient compte des besoins du créancier et des ressources du débiteur, "
                "notamment les revenus, le patrimoine, la durée du mariage et les sacrifices professionnels."
            ),
            "gold_articles": ["270 Code civil", "271 Code civil"],
            "gold_keywords": ["prestation compensatoire", "disparité", "ressources", "mariage"],
        },
        {
            "id": "M1-S-006",
            "specialisation": "Droit Fiscal",
            "question": "Quelles sont les conditions de déductibilité de la TVA sur les dépenses d'entreprise ?",
            "gold_answer": (
                "La TVA est déductible si la dépense est affectée à une activité économique soumise à TVA, si la taxe "
                "est mentionnée sur une facture régulière, et si la déduction intervient dans les délais légaux. "
                "Les dépenses à caractère mixte font l'objet d'une proratisation."
            ),
            "gold_articles": ["271 Code général des impôts", "205 Annexe II CGI"],
            "gold_keywords": ["TVA", "déductibilité", "assujetti", "facture", "prorata"],
        },
        {
            "id": "M1-S-007",
            "specialisation": "Droit Social",
            "question": "Un employeur peut-il modifier unilatéralement les conditions de travail d'un salarié ?",
            "gold_answer": (
                "L'employeur peut modifier les conditions de travail (organisation, horaires) mais ne peut pas modifier "
                "le contrat de travail sans l'accord du salarié. La distinction est essentielle : "
                "seul le changement d'un élément essentiel du contrat (rémunération, qualification, lieu de travail "
                "hors secteur géographique) nécessite l'accord exprès du salarié."
            ),
            "gold_articles": ["L. 1221-1 Code du travail", "L. 1232-1 Code du travail"],
            "gold_keywords": ["modification contrat", "unilatéral", "accord salarié", "conditions travail"],
        },
        {
            "id": "M1-S-008",
            "specialisation": "Droit Civil",
            "question": "Quelles sont les conditions de la responsabilité pour trouble anormal du voisinage ?",
            "gold_answer": (
                "Le trouble anormal du voisinage est un régime de responsabilité sans faute (art. 1253 C. civ.). "
                "Il suppose un trouble dépassant les inconvénients normaux du voisinage, en raison de son intensité, "
                "de sa durée ou de sa répétition. La preuve du trouble incombe à la victime."
            ),
            "gold_articles": ["1253 Code civil"],
            "gold_keywords": ["trouble anormal", "voisinage", "responsabilité", "sans faute", "intensité"],
        },
        {
            "id": "M1-S-009",
            "specialisation": "Droit Commercial",
            "question": "Quelles sont les obligations du dirigeant de société en matière de déclaration de cessation de paiements ?",
            "gold_answer": (
                "Le dirigeant doit déclarer la cessation des paiements dans les 45 jours au greffe du tribunal de commerce "
                "en vue d'une procédure de redressement ou liquidation judiciaire (art. L. 631-4 C. com.). "
                "Le défaut de déclaration dans les délais peut engager sa responsabilité personnelle pour insuffisance d'actif."
            ),
            "gold_articles": ["L. 631-4 Code de commerce", "L. 651-2 Code de commerce"],
            "gold_keywords": ["cessation paiements", "dirigeant", "45 jours", "redressement", "liquidation"],
        },
        {
            "id": "M1-S-010",
            "specialisation": "Droit Pénal",
            "question": "Quels sont les éléments constitutifs du délit d'abus de biens sociaux ?",
            "gold_answer": (
                "L'abus de biens sociaux (art. L. 241-3 ou L. 242-6 C. com.) suppose un usage des biens ou du crédit "
                "de la société, contraire à l'intérêt social, à des fins personnelles ou pour favoriser une autre société "
                "dans laquelle le dirigeant a un intérêt, et une mauvaise foi. La prescription est de 6 ans."
            ),
            "gold_articles": ["L. 241-3 Code de commerce", "L. 242-6 Code de commerce"],
            "gold_keywords": ["abus biens sociaux", "dirigeant", "intérêt personnel", "mauvaise foi"],
        },
    ]


# ══════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 60)
    print("Préparation des données benchmark pour le cluster")
    print("=" * 60)

    m6_path = export_m6_gold_cases()
    m1_path = export_m1_full(n=None)  # export complet : le notebook échantillonne ensuite

    print()
    print("=" * 60)
    print("TERMINÉ. Transférez le dossier cluster_data/ sur le cluster :")
    print(f"  scp -r {OUTDIR} user@cluster:/path/to/benchmark/")
    print("=" * 60)

    # Vérification
    for path in [m6_path, m1_path]:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {path.name}: {len(data)} enregistrements, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
