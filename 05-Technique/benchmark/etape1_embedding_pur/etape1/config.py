"""Étape 1 — chemins, constantes, mapping code_slug → nom LEGI officiel."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Inputs read-only (relatifs depuis ROOT)
BENCH = ROOT.parents[0]  # 05-Technique/benchmark/
BUNDLE = BENCH / "baseline_b2" / "penal_bundle"
GRAPH_NPZ = BUNDLE / "graph_penal.npz"
JP_INDEX = BUNDLE / "jp_index_penal.parquet"
RUBRICS = BUNDLE / "rubrics_penal.json"

# LEGI SQLite (produit par Task 3)
LEGI_DIR = DATA / "legi"
LEGI_DIR.mkdir(exist_ok=True)
LEGI_SQLITE = LEGI_DIR / "legi.sqlite"

# Sorties
ARTICLES_PARQUET = DATA / "articles_penal.parquet"
ARTICLES_COVERAGE = DATA / "articles_coverage.json"
ARTICLES_ORDER = DATA / "articles_order.npy"
PAIRKEY_TO_GRAPHCOL = DATA / "pairkey_to_graphcol.npy"
JP_ORDER = DATA / "jp_order.npy"
JP_TO_GRAPHROW = DATA / "jp_to_graphrow.npy"
EMB_ARTICLES = DATA / "emb_articles.npy"
EMB_JP = DATA / "emb_jp.npy"
TOKEN_STATS = DATA / "token_stats.json"
RECALL_CURVES = DATA / "recall_curves.csv"
RECALL_KSTAR = DATA / "recall_kstar.json"

# Sorties — full corpus (tous codes)
ARTICLES_PARQUET_ALL    = DATA / "articles_all.parquet"
ARTICLES_COVERAGE_ALL   = DATA / "articles_all_coverage.json"
ARTICLES_ORDER_ALL      = DATA / "articles_order_all.npy"
PAIRKEY_TO_GRAPHCOL_ALL = DATA / "pairkey_to_graphcol_all.npy"
EMB_ARTICLES_ALL        = DATA / "emb_articles_all.npy"
RECALL_CURVES_ALL       = DATA / "recall_curves_articles_all.csv"
RECALL_KSTAR_ALL        = DATA / "recall_kstar_articles_all.json"

# Sorties — JP summaries (depuis OVH PostgreSQL, step1_raw.synthese_pour_avocat)
JP_SUMMARIES_PARQUET    = DATA / "jp_summaries_penal.parquet"
JP_SUMMARY_ORDER        = DATA / "jp_summary_order.npy"
JP_SUMMARY_TO_GRAPHROW  = DATA / "jp_summary_to_graphrow.npy"
EMB_JP_SYNTHESE         = DATA / "emb_jp_synthese.npy"
JP_SUMMARIES_COVERAGE   = DATA / "jp_summaries_coverage.json"

# Credentials DB OVH (.env.local racine 05-Technique/)
ENV_LOCAL = BENCH.parent / ".env.local"

# Modèle
MODEL_ID = "BAAI/bge-m3"
EMB_DIM = 1024
MAX_CTX = 8192          # contexte théorique du modèle (8k)
# Seuil pratique sur Mac MPS : au-delà, l'attention quadratique fait OOM.
# Les textes plus longs sont chunkés+mean-poolés (préservation, pas troncature).
BATCH_MAX_LEN = 2048

# 4 codes pénaux : code_slug → nom officiel LEGI
PENAL_CODES: dict[str, str] = {
    "code_penal":                              "Code pénal",
    "code_de_procedure_penale":                "Code de procédure pénale",
    "code_de_la_route":                        "Code de la route",
    "code_de_la_justice_penale_des_mineurs":   "Code de la justice pénale des mineurs",
}

# Mapping complet : tous les code_slug du graph_penal.npz → titre LEGI officiel.
# Établi par inspection statique de :
#   - set(z["article_codes"]) du graphe (56 slugs)
#   - SELECT DISTINCT titre FROM textes_versions WHERE etat='VIGUEUR' AND titre LIKE 'Code %'
# `code_d_instruction_criminelle` est laissé à None : code abrogé, pas de version
# en VIGUEUR dans LEGI (145 articles non résolvables).
ALL_CODES: dict[str, str | None] = {
    "code_civil":                                       "Code civil",
    "code_d_instruction_criminelle":                    None,
    "code_de_commerce":                                 "Code de commerce",
    "code_de_deontologie_des_architectes":              "Code de déontologie des architectes",
    "code_de_justice_administrative":                   "Code de justice administrative",
    "code_de_justice_militaire_nouveau":                "Code de justice militaire.",
    "code_de_l_action_sociale_et_des_familles":         "Code de l'action sociale et des familles",
    "code_de_l_energie":                                "Code de l'énergie",
    "code_de_l_entree_et_du_sejour_des_etrangers_et_du_droit_d_asile": "Code de l'entrée et du séjour des étrangers et du droit d'asile.",
    "code_de_l_environnement":                          "Code de l'environnement",
    "code_de_l_expropriation_pour_cause_d_utilite_publique": "Code de l'expropriation pour cause d'utilité publique",
    "code_de_l_organisation_judiciaire":                "Code de l'organisation judiciaire",
    "code_de_l_urbanisme":                              "Code de l'urbanisme",
    "code_de_la_commande_publique":                     "Code de la commande publique",
    "code_de_la_construction_et_de_l_habitation":       "Code de la construction et de l'habitation.",
    "code_de_la_consommation":                          "Code de la consommation",
    "code_de_la_defense":                               "Code de la défense.",
    "code_de_la_famille_et_de_l_aide_sociale":          "Code de la famille et de l'aide sociale.",
    "code_de_la_justice_penale_des_mineurs":            "Code de la justice pénale des mineurs",
    "code_de_la_mutualite":                             "Code de la mutualité",
    "code_de_la_propriete_intellectuelle":              "Code de la propriété intellectuelle",
    "code_de_la_route":                                 "Code de la route.",
    "code_de_la_sante_publique":                        "Code de la santé publique",
    "code_de_la_securite_interieure":                   "Code de la sécurité intérieure",
    "code_de_la_securite_sociale":                      "Code de la sécurité sociale.",
    "code_de_la_voirie_routiere":                       "Code de la voirie routière",
    "code_de_procedure_civile":                         "Code de procédure civile",
    "code_de_procedure_penale":                         "Code de procédure pénale",
    "code_des_assurances":                              "Code des assurances",
    "code_des_communes":                                "Code des communes",
    "code_des_douanes":                                 "Code des douanes",
    "code_des_impositions_sur_les_biens_et_services":   "Code des impositions sur les biens et services",
    "code_des_juridictions_financieres":                "Code des juridictions financières",
    "code_des_pensions_civiles_et_militaires_de_retraite": "Code des pensions civiles et militaires de retraite",
    "code_des_pensions_militaires_d_invalidite_et_des_victimes_de_guerre": "Code des pensions militaires d'invalidité et des victimes de guerre.",
    "code_des_postes_et_des_communications_electroniques": "Code des postes et des communications électroniques",
    "code_des_procedures_civiles_d_execution":          "Code des procédures civiles d'exécution",
    "code_des_relations_entre_le_public_et_l_administration": "Code des relations entre le public et l'administration",
    "code_du_cinema_et_de_l_image_animee":              "Code du cinéma et de l'image animée",
    "code_du_patrimoine":                               "Code du patrimoine",
    "code_du_service_national":                         "Code du service national",
    "code_du_sport":                                    "Code du sport.",
    "code_du_travail":                                  "Code du travail",
    "code_electoral":                                   "Code électoral",
    "code_forestier_nouveau":                           "Code forestier (nouveau)",
    "code_general_de_la_fonction_publique":             "Code général de la fonction publique",
    "code_general_de_la_propriete_des_personnes_publiques": "Code général de la propriété des personnes publiques.",
    "code_general_des_collectivites_territoriales":     "Code général des collectivités territoriales",
    "code_general_des_impots":                          "Code général des impôts, CGI.",
    "code_general_des_impots_annexe_iv":                "Code général des impôts, annexe IV, CGIANIV.",
    "code_minier_nouveau":                              "Code minier (nouveau)",
    "code_monetaire_et_financier":                      "Code monétaire et financier",
    "code_penal":                                       "Code pénal",
    "code_penitentiaire":                               "Code pénitentiaire",
    "code_rural_et_de_la_peche_maritime":               "Code rural et de la pêche maritime",
    "livre_des_procedures_fiscales":                    "Livre des procédures fiscales",
}

# Eval
KS = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
KSTAR_THRESHOLD = 0.5

# Rubrics étendues — multi-branches
RUBRICS_AFFAIRES = BENCH / "data" / "rubrics" / "cnb-affaires-2025-consolidated.json"

# Mapping branche → ensemble de code_slug utilisés en stratégie 2 (filtered retrieval)
# Pour chaque branche : pool restreint aux articles de ces codes.
BRANCHES: dict[str, list[str]] = {
    "penal":    list(PENAL_CODES.keys()),
    "affaires": ["code_civil", "code_de_commerce", "code_monetaire_et_financier"],
}
