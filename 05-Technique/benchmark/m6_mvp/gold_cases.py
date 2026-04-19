"""
5 cas gold standard pour le Module 6 — Interprétation d'arrêt contextualisée.

Couverture :
    - Droit Social (cas 1)     — décision favorable, niveau facile
    - Droit Civil-Baux (cas 2) — décision favorable, niveau moyen
    - Droit de la Famille (3)  — décision défavorable, niveau moyen
    - Droit Commercial (4)     — cas piège (semble favorable, ne l'est pas)
    - Droit Pénal (5)          — décision ambiguë, niveau difficile

Les décisions sont des VRAIES décisions Judilibre (Cour de cassation).
Les dossiers clients sont fictifs mais inspirés de vrais patterns jurisprudentiels.

Décisions utilisées :
    1. Cass. soc., 15 janv. 2020, n° 18-14.177 (liberté d'expression salarié)
    2. Cass. 3e civ., 8 mars 2018, n° 17-10.315 (trouble anormal voisinage)
    3. Cass. 1re civ., 2 sept. 2020, n° 19-11.928 (prestation compensatoire)
    4. Cass. com., 17 janv. 2018, n° 16-22.253 (rupture brutale — piège procédural)
    5. Cass. crim., 30 janv. 2018, n° 17-81.706 (légitime défense)
"""

from __future__ import annotations

from datetime import date

from schema import (
    CaseContext,
    ClientPosition,
    Decision,
    DecisionStructure,
    Difficulty,
    Juridiction,
    M6GoldAnnotation,
    M6TestCase,
    SensArret,
    Specialisation,
)


def _load_real_decision(
    number: str,
    keywords: list[str],
    date_min: str,
    date_max: str,
    chamber: str | None = None,
) -> dict:
    """Charge une décision réelle depuis les fichiers Judilibre locaux."""
    from loaders.judilibre import search_decisions

    kw = {"chamber": chamber} if chamber else {}
    results = search_decisions(
        "CC",
        keywords=keywords,
        date_min=date_min,
        date_max=date_max,
        max_results=10,
        enriched=True,
        **kw,
    )
    for r in results:
        if r["number"] == number:
            return r
    raise ValueError(f"Décision {number} introuvable dans Judilibre")


def _make_decision(record: dict) -> Decision:
    """Convertit un enregistrement Judilibre en Decision Pydantic."""
    from loaders.judilibre import to_decision
    return to_decision(record)


# ═══════════════════════════════════════════════════════════════════════
# Chargement des décisions réelles (lazy, au premier import)
# ═══════════════════════════════════════════════════════════════════════


def _build_cases() -> list[M6TestCase]:
    """Construit les 5 cas gold avec les vraies décisions Judilibre."""

    # ─── CAS 1 — Droit Social — Licenciement faute grave (email critique) ───
    # Cass. soc., 15 janv. 2020, n° 18-14.177
    dec1 = _make_decision(_load_real_decision(
        "18-14.177",
        ["liberté", "expression", "licenciement", "faute"],
        "2020-01-15", "2020-01-15",
        chamber="sociale",
    ))

    case_1 = M6TestCase(
        id="M6-JP-001",
        specialisation=Specialisation.DROIT_SOCIAL,
        difficulty=Difficulty.EASY,
        case=CaseContext(
            specialisation=Specialisation.DROIT_SOCIAL,
            case_summary=(
                "Salarié commercial de 8 ans d'ancienneté, licencié pour faute grave "
                "après avoir envoyé un email critique sur le management à ses collègues "
                "de travail. L'email pointait des décisions de réorganisation internes. "
                "Aucun avertissement préalable. Pas d'antécédent disciplinaire."
            ),
            client_position=ClientPosition.DEFENSE,
            key_facts=[
                "ancienneté 8 ans",
                "email interne critique (pas public)",
                "absence d'avertissement préalable",
                "aucun antécédent disciplinaire",
                "propos non diffamatoires",
            ],
        ),
        question="Le licenciement pour faute grave est-il valable dans ce contexte ?",
        decision=dec1,
        gold=M6GoldAnnotation(
            camp_in_decision=(
                "M. L..., salarié responsable commercial, demandeur au pourvoi — "
                "licencié pour faute grave pour propos critiques"
            ),
            sens_arret=SensArret.CASSATION,
            is_favorable=True,
            dispositif_summary=(
                "La Cour de cassation casse l'arrêt d'appel qui avait requalifié "
                "le licenciement en cause réelle et sérieuse. Elle vise les articles "
                "L. 1121-1 et L. 1232-1 du code du travail et rappelle que la liberté "
                "d'expression du salarié ne peut être restreinte que de manière justifiée "
                "et proportionnée. Le renvoi est prononcé devant la CA de Montpellier."
            ),
            relevance=0.90,
            principles_extracted=[
                "La liberté d'expression du salarié est protégée (art. L. 1121-1 C. trav.)",
                "Seules des restrictions justifiées par la tâche et proportionnées au but sont admises",
                "Des propos critiques en interne ne suffisent pas à constituer une faute grave",
                "La qualification de faute grave exige de caractériser précisément l'abus",
            ],
            transfer_reasoning=(
                "Cette décision s'applique directement au dossier : le salarié a diffusé un "
                "email critique en interne, sans diffamation apparente. La Cour de cassation "
                "exige que l'employeur caractérise un abus (propos injurieux, diffamatoires ou "
                "excessifs) pour justifier un licenciement, ce qui semble absent ici. Argument "
                "central : le licenciement pour faute grave viole la liberté d'expression du "
                "salarié car les propos ne sont ni injurieux ni diffamatoires."
            ),
        ),
        notes="Cas de référence 'favorable direct' — tout le raisonnement se transfère.",
    )

    # ─── CAS 2 — Droit Civil (Baux) — Trouble anormal du voisinage ─────────
    # Cass. 3e civ., 8 mars 2018, n° 17-10.315
    dec2 = _make_decision(_load_real_decision(
        "17-10.315",
        ["trouble anormal", "voisinage"],
        "2018-03-08", "2018-03-08",
    ))

    case_2 = M6TestCase(
        id="M6-JP-002",
        specialisation=Specialisation.DROIT_CIVIL,
        difficulty=Difficulty.MEDIUM,
        case=CaseContext(
            specialisation=Specialisation.DROIT_CIVIL,
            case_summary=(
                "Copropriétaire d'un appartement se plaint de nuisances sonores provenant "
                "de l'appartement voisin situé à l'étage supérieur : bruits répétés après 23h "
                "(musique forte, talons), relevés par constat d'huissier sur 3 mois. Le voisin "
                "refuse toute médiation. Le demandeur sollicite la cessation des troubles et "
                "des dommages-intérêts."
            ),
            client_position=ClientPosition.DEMANDE,
            key_facts=[
                "bruits répétés après 23h",
                "constat d'huissier sur 3 mois",
                "refus de médiation du voisin",
                "pas de problème isolé, répétition caractérisée",
            ],
        ),
        question=(
            "Le demandeur peut-il obtenir la cessation des troubles et des dommages-intérêts "
            "au titre du trouble anormal du voisinage ?"
        ),
        decision=dec2,
        gold=M6GoldAnnotation(
            camp_in_decision=(
                "Les consorts X..., propriétaires voisins, demandeurs — ils invoquent "
                "le trouble anormal de voisinage lié à la terrasse exhaussée"
            ),
            sens_arret=SensArret.REJET,
            is_favorable=True,
            dispositif_summary=(
                "La Cour de cassation rejette le pourvoi. Elle valide l'appréciation "
                "souveraine de la cour d'appel sur le trouble anormal de voisinage, "
                "notamment la présence d'encombrants et la configuration des lieux. "
                "Les critères d'appréciation (constatations d'huissier, photographies, "
                "configuration des lieux) sont confirmés."
            ),
            relevance=0.85,
            principles_extracted=[
                "Le trouble anormal de voisinage s'apprécie souverainement par les juges du fond",
                "Le constat d'huissier + photographies constituent des preuves recevables",
                "La configuration des lieux et la situation de l'ouvrage sont des critères clés",
                "L'appréciation relève du pouvoir souverain des juges du fond",
            ],
            transfer_reasoning=(
                "Décision pertinente pour le dossier : le demandeur dispose de conditions "
                "probatoires comparables (constat d'huissier sur 3 mois, bruits après 23h). "
                "La Cour confirme que l'appréciation souveraine des juges du fond suffit "
                "dès lors que les preuves (constats, attestations) établissent la réalité "
                "du trouble. Stratégie : produire le constat d'huissier et les attestations "
                "de voisinage pour caractériser la répétition, l'horaire nocturne et l'intensité."
            ),
        ),
        notes=(
            "Cas favorable avec une nuance (la décision traite de terrasse exhaussée, pas "
            "de bruit nocturne — le LLM doit pondérer la pertinence au lieu de déclarer 1.00)."
        ),
    )

    # ─── CAS 3 — Droit de la Famille — Prestation compensatoire ─────────────
    # Cass. 1re civ., 2 sept. 2020, n° 19-11.928
    dec3 = _make_decision(_load_real_decision(
        "19-11.928",
        ["prestation compensatoire", "disparité"],
        "2020-09-02", "2020-09-02",
        chamber="civile",
    ))

    case_3 = M6TestCase(
        id="M6-JP-003",
        specialisation=Specialisation.DROIT_FAMILLE,
        difficulty=Difficulty.MEDIUM,
        case=CaseContext(
            specialisation=Specialisation.DROIT_FAMILLE,
            case_summary=(
                "Mari, cadre supérieur (salaire net 6 500 €/mois), divorce par consentement "
                "altéré après 18 ans de mariage. Épouse, enseignante (3 200 €/mois), a interrompu "
                "sa carrière pendant 6 ans pour élever les 3 enfants. Le mari conteste le "
                "montant de la prestation compensatoire (200 000 €) fixée en 1re instance, "
                "arguant qu'il avait aussi fait des sacrifices professionnels (refus de mutation)."
            ),
            client_position=ClientPosition.DEFENSE,
            key_facts=[
                "mariage 18 ans",
                "disparité de revenus 6 500 vs 3 200",
                "interruption de carrière de l'épouse 6 ans",
                "mari invoque refus de mutation comme 'sacrifice'",
                "montant initial 200 000 €",
            ],
        ),
        question=(
            "Le mari peut-il obtenir la réduction de la prestation compensatoire en invoquant "
            "ses propres sacrifices professionnels ?"
        ),
        decision=dec3,
        gold=M6GoldAnnotation(
            camp_in_decision=(
                "Mme H..., épouse, demanderesse au pourvoi (conteste le montant insuffisant "
                "de la PC) — le mari M. R... est défendeur"
            ),
            sens_arret=SensArret.REJET,
            is_favorable=False,
            dispositif_summary=(
                "La Cour de cassation rejette le pourvoi. Elle rappelle les articles 270 et "
                "271 du code civil : la PC compense la disparité créée par la rupture dans "
                "les conditions de vie. Le juge la fixe selon les besoins du créancier et "
                "les ressources du débiteur, au moment du divorce et dans un avenir prévisible. "
                "L'appréciation souveraine de la CA est validée."
            ),
            relevance=0.80,
            principles_extracted=[
                "La PC compense la disparité créée par la rupture dans les conditions de vie (art. 270 C. civ.)",
                "Elle s'apprécie au regard des besoins du créancier et ressources du débiteur (art. 271)",
                "Le juge tient compte de la situation au moment du divorce et de son évolution prévisible",
                "L'appréciation du montant relève du pouvoir souverain des juges du fond",
            ],
            transfer_reasoning=(
                "DÉCISION DÉFAVORABLE pour le mari client. La Cour de cassation confirme la "
                "méthode d'appréciation de la PC basée sur les articles 270-271 C. civ. Les "
                "principes posés valident l'analyse de la CA qui tient compte de la disparité "
                "objective. Les 'sacrifices' du débiteur ne sont pas un critère autonome de "
                "réduction. Stratégie : ne pas invoquer cet argument, plutôt contester sur "
                "les éléments patrimoniaux et la capacité future de l'épouse à augmenter ses "
                "revenus."
            ),
        ),
        notes="Cas défavorable explicite. Le LLM doit reconnaître que transfert = risque.",
    )

    # ─── CAS 4 — Droit Commercial — CAS PIÈGE (cassation procédurale) ──────
    # Cass. com., 17 janv. 2018, n° 16-22.253
    dec4 = _make_decision(_load_real_decision(
        "16-22.253",
        ["rupture brutale", "relations commerciales"],
        "2018-01-17", "2018-01-17",
        chamber="commerciale",
    ))

    case_4 = M6TestCase(
        id="M6-JP-004",
        specialisation=Specialisation.DROIT_COMMERCIAL,
        difficulty=Difficulty.TRAP,
        case=CaseContext(
            specialisation=Specialisation.DROIT_COMMERCIAL,
            case_summary=(
                "Distributeur de produits cosmétiques, partenaire de la société X depuis 12 ans "
                "sous contrat renouvelé chaque année, voit brusquement son contrat résilié sans "
                "préavis. Il assigne X pour rupture brutale des relations commerciales établies "
                "(art. L. 442-1, II C. com.) et réclame 800 000 € au titre du préjudice."
            ),
            client_position=ClientPosition.DEMANDE,
            key_facts=[
                "relation commerciale 12 ans",
                "contrat renouvelé annuellement",
                "rupture sans préavis",
                "préjudice évalué 800 000 €",
            ],
        ),
        question=(
            "Le distributeur peut-il obtenir réparation sur le fondement de la rupture brutale ?"
        ),
        decision=dec4,
        gold=M6GoldAnnotation(
            camp_in_decision=(
                "Société DCO (distributeur de gravure/marquage antivol), victime de "
                "la rupture — les sociétés Jeannin sont auteures de la résiliation"
            ),
            sens_arret=SensArret.CASSATION,
            is_favorable=False,
            dispositif_summary=(
                "La Cour de cassation casse l'arrêt de la CA de Paris, mais sur un moyen "
                "PROCÉDURAL (violation de l'article 4 du CPC). La CA avait limité les "
                "dommages-intérêts en modifiant les termes du litige. La cassation porte sur "
                "le quantum, PAS sur le principe de la rupture brutale qui reste acquis. "
                "Aucun principe de fond utile n'est dégagé."
            ),
            relevance=0.20,
            principles_extracted=[
                "Procédural : le juge ne peut modifier l'objet du litige (art. 4 CPC)",
                "Aucun principe de fond sur la rupture brutale n'est posé ici",
            ],
            transfer_reasoning=(
                "ATTENTION CAS PIÈGE. La décision est une CASSATION, ce qui peut laisser "
                "croire qu'elle est favorable au distributeur. En réalité : (1) la cassation "
                "porte sur une violation de l'article 4 CPC (ultra petita), pas sur le fond, "
                "(2) le moyen porte sur le calcul du préjudice, pas sur la qualification de "
                "rupture brutale, (3) la qualification de rupture reste acquise. "
                "POUR NOTRE DOSSIER : cette décision n'apporte rien de substantiel. Ne pas "
                "l'invoquer comme JP favorable. Chercher plutôt une décision traitant au fond "
                "du calcul du préavis pour une relation de 12 ans."
            ),
        ),
        notes=(
            "CAS PIÈGE CRITIQUE. Un LLM naïf qui voit 'cassation' + faits proches va répondre "
            "is_favorable=True et relevance=0.8+. Le bon comportement est is_favorable=False "
            "et relevance<=0.3 car la décision n'apporte aucun principe de fond."
        ),
    )

    # ─── CAS 5 — Droit Pénal — Légitime défense (décision ambiguë) ─────────
    # Cass. crim., 30 janv. 2018, n° 17-81.706
    dec5 = _make_decision(_load_real_decision(
        "17-81.706",
        ["légitime défense", "nécessité"],
        "2018-01-30", "2018-01-30",
        chamber="criminelle",
    ))

    case_5 = M6TestCase(
        id="M6-JP-005",
        specialisation=Specialisation.DROIT_PENAL,
        difficulty=Difficulty.HARD,
        case=CaseContext(
            specialisation=Specialisation.DROIT_PENAL,
            case_summary=(
                "Commerçant agressé de nuit dans son magasin par deux individus armés. Il a "
                "riposté en tirant un coup de feu (arme détenue légalement) blessant l'un d'eux. "
                "Il est poursuivi pour violences volontaires. Il invoque la légitime défense. "
                "L'individu blessé était désarmé au moment du tir (il venait de jeter son couteau "
                "pour prendre la fuite)."
            ),
            client_position=ClientPosition.DEFENSE,
            key_facts=[
                "agression nocturne par deux individus armés",
                "riposte par coup de feu — 1 blessé",
                "agresseur désarmé au moment précis du tir (fuite amorcée)",
                "arme du commerçant légalement détenue",
            ],
        ),
        question="Le commerçant peut-il bénéficier de la légitime défense ?",
        decision=dec5,
        gold=M6GoldAnnotation(
            camp_in_decision=(
                "M. Michel-Ange X..., auteur des violences avec arme (manivelle), "
                "poursuivi pour violences — il invoque la légitime défense"
            ),
            sens_arret=SensArret.REJET,
            is_favorable=False,
            dispositif_summary=(
                "La Cour de cassation rejette le pourvoi. Elle confirme la condamnation "
                "pour violences avec arme. La légitime défense est écartée car la riposte "
                "(manivelle) était disproportionnée par rapport à l'agression initiale "
                "(coups de pied et de poing). La condition de proportionnalité de la "
                "défense fait défaut."
            ),
            relevance=0.92,
            principles_extracted=[
                "La légitime défense suppose une réponse proportionnée à l'agression",
                "L'utilisation d'une arme contre une agression à mains nues est disproportionnée",
                "La proportionnalité s'apprécie au moment précis de la riposte",
                "Distinction entre défense légitime et riposte excessive",
            ],
            transfer_reasoning=(
                "DÉCISION DÉFAVORABLE à notre client dans sa configuration actuelle. Les "
                "principes sont proches : notre client a utilisé une arme à feu alors que "
                "l'agresseur avait abandonné son arme. La Cour confirme l'exigence de "
                "proportionnalité. STRATÉGIE : (1) distinguer les faits en insistant sur "
                "l'agression initiale armée (couteau vs mains nues dans l'arrêt), ce qui "
                "pourrait modifier l'appréciation de la proportionnalité ; (2) invoquer "
                "l'état de choc et la simultanéité (légitime défense putative) ; (3) à "
                "défaut, plaider les circonstances atténuantes plutôt que l'irresponsabilité."
            ),
        ),
        notes=(
            "Cas difficile : l'arrêt traite d'une agression à mains nues (pied/poing) vs "
            "arme (manivelle), tandis que notre dossier concerne une agression armée initiale "
            "(couteau). Le LLM doit identifier cette distinction et nuancer sa réponse."
        ),
    )

    return [case_1, case_2, case_3, case_4, case_5]


# Chargement lazy des cas
_CASES: list[M6TestCase] | None = None


def get_all_cases() -> list[M6TestCase]:
    """Retourne les 5 cas gold standard (chargement lazy)."""
    global _CASES
    if _CASES is None:
        _CASES = _build_cases()
    return _CASES


# Pour compatibilité avec le code existant
ALL_CASES = property(lambda self: get_all_cases())


if __name__ == "__main__":
    cases = get_all_cases()
    for c in cases:
        print(
            f"{c.id} {c.specialisation.value} ({c.difficulty.value}) → "
            f"favorable={c.gold.is_favorable} | {c.decision.juridiction.value} "
            f"{c.decision.date} n°{c.decision.numero}"
        )
