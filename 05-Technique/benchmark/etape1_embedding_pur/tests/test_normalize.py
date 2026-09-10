from etape1.normalize import parse_pair_key, legi_num_candidates


def test_parse_simple():
    assert parse_pair_key("code_penal:222-23") == ("code_penal", "222-23")


def test_parse_letter_prefix():
    assert parse_pair_key("code_de_procedure_penale:L743-7") == (
        "code_de_procedure_penale", "L743-7"
    )


def test_parse_strips_whitespace():
    assert parse_pair_key("  code_penal:222-23  ") == ("code_penal", "222-23")


def test_parse_rejects_malformed():
    import pytest
    with pytest.raises(ValueError):
        parse_pair_key("no_colon_here")


def test_candidates_plain_numeric():
    cands = legi_num_candidates("222-23")
    assert "222-23" in cands


def test_candidates_letter_prefix_expands():
    cands = legi_num_candidates("L743-7")
    assert "L743-7"   in cands
    assert "L. 743-7" in cands
    assert "L 743-7"  in cands


def test_candidates_latin_suffix():
    cands = legi_num_candidates("1649quinquiesB")
    assert "1649 quinquies B" in cands
    assert "1649quinquiesB"   in cands


def test_candidates_bis_lowercase():
    cands = legi_num_candidates("L122-1bis")
    assert "L. 122-1 bis" in cands
    assert "L122-1bis"   in cands
