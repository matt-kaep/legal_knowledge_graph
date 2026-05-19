from prompts.step1 import themes_taxonomy as tx


def test_18_branches_and_canonical_pairs():
    assert len(tx.TAXONOMY) == 18
    assert all(isinstance(v, list) and v for v in tx.TAXONOMY.values())
    assert ("Droit immobilier, baux et construction",
            "baux commerciaux et indemnité d'éviction") in tx.PAIRS
    assert ("Droit des obligations et des contrats",
            "responsabilité contractuelle et dommages-intérêts") in tx.PAIRS
    assert ("Droit des sociétés et des affaires",
            "baux commerciaux et indemnité d'éviction") not in tx.PAIRS
    assert isinstance(tx.TAXONOMY_VERSION, str) and tx.TAXONOMY_VERSION


def test_render_for_prompt_lists_all_pairs():
    rendered = tx.render_for_prompt()
    for branche, subs in tx.TAXONOMY.items():
        assert branche in rendered
        for s in subs:
            assert s in rendered
    assert "Autre:" in rendered
