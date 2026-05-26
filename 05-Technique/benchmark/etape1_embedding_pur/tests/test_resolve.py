import sqlite3
import pytest
from etape1.resolve import resolve_pair_keys, coverage_report


@pytest.fixture
def mini_db(tmp_path):
    """Mini SQLite avec 3 articles factices, schéma proche de legi.py."""
    db = tmp_path / "mini.sqlite"
    with sqlite3.connect(db) as cx:
        # Schéma simplifié mais fidèle au vrai legi.py :
        #   articles(id, num, bloc_textuel, etat, cid)
        #   textes_versions(id, titre, etat, cid)
        # Jointure : articles.cid = textes_versions.id
        cx.executescript("""
        CREATE TABLE articles(id TEXT PRIMARY KEY, num TEXT,
                              bloc_textuel TEXT, etat TEXT, cid TEXT);
        CREATE TABLE textes_versions(id TEXT, titre TEXT, etat TEXT, cid TEXT);
        INSERT INTO textes_versions VALUES ('T1', 'Code pénal', 'VIGUEUR', 'T1');
        INSERT INTO articles VALUES ('A1', '222-23',           'Acte de pénétration…',  'VIGUEUR', 'T1');
        INSERT INTO articles VALUES ('A2', 'L. 743-7',         'Procédure spéciale…',   'VIGUEUR', 'T1');
        INSERT INTO articles VALUES ('A3', '1649 quinquies B', 'Disposition fiscale…',  'VIGUEUR', 'T1');
        """)
    return db


def test_resolve_plain(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:222-23"], {"code_penal": "Code pénal"})
    assert out["code_penal:222-23"]["texte"].startswith("Acte de pénétration")
    assert out["code_penal:222-23"]["matched_num"] == "222-23"


def test_resolve_letter_prefix_variant(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:L743-7"], {"code_penal": "Code pénal"})
    assert out["code_penal:L743-7"]["texte"].startswith("Procédure")
    assert out["code_penal:L743-7"]["matched_num"] == "L. 743-7"


def test_resolve_latin_suffix(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:1649quinquiesB"], {"code_penal": "Code pénal"})
    assert out["code_penal:1649quinquiesB"]["texte"].startswith("Disposition")


def test_resolve_missing(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:999-99"], {"code_penal": "Code pénal"})
    assert out["code_penal:999-99"]["texte"] is None
    assert out["code_penal:999-99"]["matched_num"] is None


def test_coverage_report_counts(mini_db):
    pks = ["code_penal:222-23", "code_penal:L743-7", "code_penal:999-99"]
    gold = {"code_penal:222-23"}
    out = resolve_pair_keys(mini_db, pks, {"code_penal": "Code pénal"})
    rep = coverage_report(out, gold_pair_keys=gold)
    assert rep["n_total"] == 3
    assert rep["n_resolved"] == 2
    assert rep["resolution_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert rep["n_gold_total"] == 1
    assert rep["n_gold_resolved"] == 1
    assert rep["gold_resolution_rate"] == 1.0
