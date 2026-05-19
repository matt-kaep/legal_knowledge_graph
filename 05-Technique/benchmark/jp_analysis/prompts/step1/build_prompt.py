from prompts.step1.step1_routing import route
from prompts.step1 import step1_shared as sh

def build_system_prompt(juris: str):
    """Assemble by concatenation ONLY — never str.format()/%/f-string on the
    block bodies (they contain literal JSON braces). Lesson: doctrine_qgen #18859."""
    preambule, variant = route(juris)
    system = (
        preambule
        + "\n\n# Règles\n\n"
        + sh.BLOC_FACTUEL_PARTAGE
        + "\n\n"
        + sh.BLOC_FORMAT_SORTIE_PARTAGE
        + "\n\n"
        + sh.BLOC_TAXONOMIE_THEMES
    )
    return system, variant
