import pytest
from pydantic import ValidationError
from jp_analysis.schema import Step1Output, SCHEMA_VERSION, json_schema

VALID = {
    "contexte": "Cour de cassation, chambre criminelle, pourvoi en matière de vol.",
    "arguments_parties": [{"partie": "demandeur", "argument": "a", "reponse_juge": "b"}],
    "fondements_retenus": "article 311-1 du code pénal",
    "dispositif": "rejette le pourvoi",
    "attendu_cle": "x" * 250,
    "cited_articles": ["article 311-1 code pénal"],
    "solution_resume": "rejet",
    "dispositif_summary": "pourvoi rejeté",
    "synthese_pour_avocat": "y" * 300,
    "dispositif_nature": "REJETTE",
    "themes": [{"branche": "Droit pénal — fond",
                "sous_branche": "atteintes aux biens et appropriations frauduleuses"}],
}

def test_valid_record_accepted():
    m = Step1Output.model_validate(VALID)
    assert m.dispositif_nature == "REJETTE"
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION

def test_missing_required_field_rejected():
    bad = dict(VALID); del bad["dispositif"]
    with pytest.raises(ValidationError):
        Step1Output.model_validate(bad)

def test_extra_key_rejected():
    bad = dict(VALID); bad["unexpected"] = 1
    with pytest.raises(ValidationError):
        Step1Output.model_validate(bad)

def test_json_schema_has_all_fields_and_no_additional_props():
    js = json_schema()
    assert js["additionalProperties"] is False
    for f in VALID:
        assert f in js["properties"]
    assert set(js["required"]) == set(VALID.keys())
