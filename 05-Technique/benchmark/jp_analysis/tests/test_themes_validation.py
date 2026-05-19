from themes_validation import canonicalize_themes

PEN = "Droit pénal — fond"
SUB = "atteintes aux biens et appropriations frauduleuses"

def test_exact_pair_accepted():
    pairs, valid, anomalies = canonicalize_themes([{"branche": PEN, "sous_branche": SUB}])
    assert pairs == [{"branche": PEN, "sous_branche": SUB}]
    assert valid is True and anomalies == []

def test_accent_case_variant_canonicalized():
    pairs, valid, anomalies = canonicalize_themes(
        [{"branche": "droit penal - fond", "sous_branche": SUB.upper()}])
    assert pairs == [{"branche": PEN, "sous_branche": SUB}]
    assert valid is True and anomalies == []

def test_well_formed_autre_accepted():
    t = [{"branche": "Autre:droit minier", "sous_branche": "Autre:redevances"}]
    pairs, valid, anomalies = canonicalize_themes(t)
    assert pairs == t and valid is True and anomalies == []

def test_incoherent_pair_dropped_and_flagged():
    t = [{"branche": "Droit du travail", "sous_branche": "stupéfiants"}]
    pairs, valid, anomalies = canonicalize_themes(t)
    assert pairs == []
    assert valid is False
    assert anomalies and anomalies[0]["raw"] == t[0]

def test_mixed_keeps_good_drops_bad():
    t = [{"branche": PEN, "sous_branche": SUB},
         {"branche": "xxx", "sous_branche": "yyy"}]
    pairs, valid, anomalies = canonicalize_themes(t)
    assert {"branche": PEN, "sous_branche": SUB} in pairs
    assert len(pairs) == 1 and valid is False and len(anomalies) == 1
