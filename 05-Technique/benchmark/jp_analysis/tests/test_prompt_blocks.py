from prompts.step1 import step1_shared as sh
from prompts.step1.step1_cassation import PREAMBULE_CASSATION
from prompts.step1.step1_cour_appel import PREAMBULE_COUR_APPEL
from prompts.step1.step1_tribunal import PREAMBULE_TRIBUNAL


def test_preambules_are_distinct_and_nonempty():
    ps = [PREAMBULE_CASSATION, PREAMBULE_COUR_APPEL, PREAMBULE_TRIBUNAL]
    assert all(len(p) > 400 for p in ps)
    assert len(set(ps)) == 3


def test_preambule_markers():
    assert "Cour de cassation" in PREAMBULE_CASSATION
    assert "Cour d'appel" in PREAMBULE_COUR_APPEL
    assert "première instance" in PREAMBULE_TRIBUNAL.lower()


def test_shared_blocks_present():
    assert "préservation factuelle" in sh.BLOC_FACTUEL_PARTAGE.lower()
    # verbatim invariant for attendu_cle kept exactly (D2)
    assert "Reproduction littérale" in sh.BLOC_FORMAT_SORTIE_PARTAGE
    assert "synthese_pour_avocat" in sh.BLOC_FORMAT_SORTIE_PARTAGE


def test_taxonomy_block_injected():
    assert "Autre:" in sh.BLOC_TAXONOMIE_THEMES
    assert "Taxonomie des thèmes" in sh.BLOC_TAXONOMIE_THEMES
