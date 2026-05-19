import pytest
from prompts.step1 import step1_shared as sh
from prompts.step1.step1_routing import route
from prompts.step1.build_prompt import build_system_prompt

def test_routing():
    assert route("CC")[1] == "cassation"
    assert route("CA")[1] == "cour_appel"
    assert route("TJ")[1] == "tribunal"

def test_routing_unknown_raises():
    with pytest.raises(ValueError):
        route("XX")

def test_build_contains_all_blocks():
    sys_prompt, variant = build_system_prompt("CA")
    assert variant == "cour_appel"
    assert "Cour d'appel" in sys_prompt
    assert "# Règles" in sys_prompt
    assert "préservation factuelle" in sys_prompt.lower()
    assert "Taxonomie des thèmes" in sys_prompt

def test_build_is_exact_concatenation_no_formatting():
    # Anti-#18859 guard: assembly must be pure concatenation. If build used
    # str.format()/%/f-string on the block bodies this equality would fail
    # (or raise). Verbatim Hector blocks contain no literal {}, so the real
    # invariant is "the output is byte-exact concatenation of the parts".
    preambule, _ = route("CC")
    expected = (
        preambule + "\n\n# Règles\n\n"
        + sh.BLOC_FACTUEL_PARTAGE + "\n\n"
        + sh.BLOC_FORMAT_SORTIE_PARTAGE + "\n\n"
        + sh.BLOC_TAXONOMIE_THEMES
    )
    assert build_system_prompt("CC")[0] == expected

def test_build_preserves_literal_braces(monkeypatch):
    # Real #18859 regression guard: a block containing literal JSON braces
    # must pass through build untouched (str.format() would raise KeyError).
    sentinel = 'BLOC_AVEC_ACCOLADES {"k": "v", "n": 1}'
    monkeypatch.setattr(sh, "BLOC_FACTUEL_PARTAGE", sentinel)
    out, _ = build_system_prompt("TJ")
    assert sentinel in out

def test_build_all_three_variants_have_taxonomy():
    for j in ("CC", "CA", "TJ"):
        sp, _ = build_system_prompt(j)
        assert "Autre:" in sp
