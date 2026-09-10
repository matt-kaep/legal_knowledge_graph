from etape1 import config


def test_inputs_exist():
    assert config.GRAPH_NPZ.exists(), config.GRAPH_NPZ
    assert config.JP_INDEX.exists(), config.JP_INDEX
    assert config.RUBRICS.exists(), config.RUBRICS


def test_penal_codes():
    assert len(config.PENAL_CODES) == 4
    assert "code_penal" in config.PENAL_CODES
    assert config.PENAL_CODES["code_de_procedure_penale"] == "Code de procédure pénale"
