"""
Schémas Pydantic pour le benchmark KG juridique FR.

Couvre les structures communes à tous les modules (M1 → M6) et les structures
spécifiques au Module 6 (interprétation d'arrêt contextualisée).

Référence : 05-Technique/methodologies/Benchmark-KG-Juridique-FR-Design.md
"""

from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════
# Énumérations partagées
# ═════════════════════════════════════════════════════════════════════


class Juridiction(str, Enum):
    """Juridictions françaises principales + tag 'synthétique' pour MVP."""
    CC = "CC"          # Cour de cassation
    CE = "CE"          # Conseil d'État
    CCONST = "CCONST"  # Conseil constitutionnel
    CA = "CA"          # Cour d'appel
    TJ = "TJ"          # Tribunal judiciaire
    TC = "TC"          # Tribunal de commerce
    CPH = "CPH"        # Conseil de prud'hommes
    TA = "TA"          # Tribunal administratif
    CAA = "CAA"        # Cour administrative d'appel
    AUTRE = "AUTRE"
    SYNTH = "SYNTH"    # Arrêt synthétique pour MVP


class JPRank(str, Enum):
    """Rang hiérarchique de la JP dans le raisonnement."""
    PRINCIPE = "principe"           # pose la règle (CC typiquement)
    INTERPRETATION = "interpretation"  # précise (CA typiquement)
    ESPECE = "espece"               # applique aux faits (TJ/TC/premières instances)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    TRAP = "trap"  # spécial M6 — décision piège


class ClientPosition(str, Enum):
    DEMANDE = "demande"
    DEFENSE = "defense"


class SensArret(str, Enum):
    """Dispositif globale de la décision."""
    CASSATION = "cassation"
    CASSATION_PARTIELLE = "cassation_partielle"
    REJET = "rejet"
    CONFIRMATION = "confirmation"     # CA
    INFIRMATION = "infirmation"       # CA
    INFIRMATION_PARTIELLE = "infirmation_partielle"
    ACCUEIL = "accueil"               # 1re instance
    DEBOUTE = "deboute"               # 1re instance
    NON_LIEU = "non_lieu"
    AUTRE = "autre"


class Specialisation(str, Enum):
    """Spécialisations juridiques (miroir des 5 catégories Hector)."""
    DROIT_PENAL = "Droit Pénal"
    DROIT_SOCIAL = "Droit Social"
    DROIT_FAMILLE = "Droit de la Famille"
    DROIT_COMMERCIAL = "Droit Commercial"
    DROIT_CIVIL = "Droit Civil"
    DROIT_FISCAL = "Droit Fiscal"
    DROIT_ADMINISTRATIF = "Droit Administratif"
    AUTRE = "Autre"


# ═════════════════════════════════════════════════════════════════════
# Entités — articles et décisions
# ═════════════════════════════════════════════════════════════════════


class ArticleReference(BaseModel):
    """Référence à un article de loi."""
    code: str = Field(..., description="Nom du code (ex: 'Code du travail')")
    number: str = Field(..., description="Numéro (ex: 'L. 1234-1')")
    version_date: date | None = Field(
        None, description="Version applicable de l'article (pour temporalité)"
    )
    relevance: Literal["central", "supporting"] = "central"
    why: str | None = None


class DecisionStructure(BaseModel):
    """Sections structurelles d'une décision juridique."""
    faits: str | None = None
    moyens: str | None = None
    motifs: str | None = None
    dispositif: str | None = None


class Decision(BaseModel):
    """Décision de justice (arrêt, jugement)."""
    id: str = Field(..., description="ECLI ou ID Judilibre ou ID synthétique")
    juridiction: Juridiction
    chambre: str | None = None
    date: date
    numero: str | None = Field(None, description="Numéro de pourvoi/affaire")
    full_text: str = Field(..., description="Texte complet de la décision")
    structure: DecisionStructure | None = None
    is_synthetic: bool = False


class JPReference(BaseModel):
    """Référence à une décision, enrichie de son rang dans le raisonnement."""
    decision_id: str
    juridiction: Juridiction
    date: date
    rank: JPRank
    why: str = Field(..., description="Pourquoi cette décision est citée")


# ═════════════════════════════════════════════════════════════════════
# Contexte d'un cas
# ═════════════════════════════════════════════════════════════════════


class CaseContext(BaseModel):
    """Contexte d'un dossier client (niveau 2)."""
    specialisation: Specialisation
    case_summary: str = Field(..., description="Résumé du dossier")
    client_position: ClientPosition
    key_facts: list[str] = Field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════
# Module 6 — Interprétation d'arrêt contextualisée
# ═════════════════════════════════════════════════════════════════════


class M6GoldAnnotation(BaseModel):
    """
    Annotation gold standard pour Module 6.
    Les 7 dimensions évaluées pour une paire (dossier, décision).
    """

    camp_in_decision: str = Field(
        ..., description="Qui représente la position du client dans cette décision ?"
    )
    sens_arret: SensArret
    is_favorable: bool = Field(..., description="Décision favorable au client actuel ?")
    dispositif_summary: str = Field(..., description="Résumé (2-4 phrases) de ce que décide la cour")
    relevance: float = Field(..., ge=0, le=1, description="Score de pertinence 0-1")
    principles_extracted: list[str] = Field(
        ..., description="Principes de droit utilisables dans le dossier"
    )
    transfer_reasoning: str = Field(
        ..., description="Justification du transfert au dossier actuel (1 paragraphe)"
    )


class M6TestCase(BaseModel):
    """Un cas complet de test du Module 6."""

    id: str
    specialisation: Specialisation
    difficulty: Difficulty
    case: CaseContext
    question: str = Field(..., description="Question implicite (en général 'est-ce utile pour moi ?')")
    decision: Decision
    gold: M6GoldAnnotation
    notes: str | None = None


# ═════════════════════════════════════════════════════════════════════
# Structures de sortie (ce que le LLM doit produire, à comparer au gold)
# ═════════════════════════════════════════════════════════════════════


class M6ModelOutput(BaseModel):
    """Ce qu'un LLM testé doit produire pour M6.
    Même schéma que M6GoldAnnotation pour permettre la comparaison directe."""

    camp_in_decision: str
    sens_arret: SensArret
    is_favorable: bool
    dispositif_summary: str
    relevance: float
    principles_extracted: list[str]
    transfer_reasoning: str


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def test_case_summary_line(tc: M6TestCase) -> str:
    """Résumé une ligne pour affichage rapide."""
    return (
        f"[{tc.id}] {tc.specialisation.value} ({tc.difficulty.value}) — "
        f"{tc.decision.juridiction.value} {tc.decision.date} — "
        f"favorable={tc.gold.is_favorable}"
    )
