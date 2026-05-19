from prompts.step1.step1_cassation import PREAMBULE_CASSATION
from prompts.step1.step1_cour_appel import PREAMBULE_COUR_APPEL
from prompts.step1.step1_tribunal import PREAMBULE_TRIBUNAL

_ROUTES = {
    "CC": (PREAMBULE_CASSATION, "cassation"),
    "CA": (PREAMBULE_COUR_APPEL, "cour_appel"),
    "TJ": (PREAMBULE_TRIBUNAL, "tribunal"),
}

def route(juris: str):
    try:
        return _ROUTES[juris]
    except KeyError:
        raise ValueError(f"unknown juris {juris!r} (expected CC|CA|TJ)")
