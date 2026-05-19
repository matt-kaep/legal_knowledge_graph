import pytest
from prompts.step1.step1_routing import route
from prompts.step1.build_prompt import build_system_prompt

def test_routing():
    assert route("CC")[1] == "cassation"
    assert route("CA")[1] == "cour_appel"
    assert route("TJ")[1] == "tribunal"

def test_routing_unknown_raises():
    with pytest.raises(ValueError):
        route("XX")

def test_build_contains_all_blocks_and_no_format_keyerror():
    sys_prompt, variant = build_system_prompt("CA")
    assert variant == "cour_appel"
    assert "Cour d'appel" in sys_prompt
    assert "# Règles" in sys_prompt
    assert "préservation factuelle" in sys_prompt.lower()
    assert "Taxonomie des thèmes" in sys_prompt
    assert "{" in sys_prompt and "}" in sys_prompt

def test_build_all_three_variants_have_taxonomy():
    for j in ("CC", "CA", "TJ"):
        sp, _ = build_system_prompt(j)
        assert "Autre:" in sp
