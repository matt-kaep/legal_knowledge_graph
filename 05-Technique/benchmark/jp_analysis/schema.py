"""Step1Output — schéma de sortie strict (10 champs Hector + themes)."""
from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0.0"

class ArgumentPartie(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partie: str
    argument: str
    reponse_juge: str

class Theme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    branche: str
    sous_branche: str

class Step1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contexte: str
    arguments_parties: list[ArgumentPartie]
    fondements_retenus: str
    dispositif: str
    attendu_cle: str
    cited_articles: list[str]
    solution_resume: str
    dispositif_summary: str
    synthese_pour_avocat: str
    dispositif_nature: str
    themes: list[Theme]

def json_schema() -> dict:
    """JSON Schema for vLLM guided decoding (strict, no extra keys)."""
    js = Step1Output.model_json_schema()
    js["additionalProperties"] = False
    return js
