from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "sbatch_b2_e030_judge_v3.sh"
SMOKE = ROOT / "scripts" / "sbatch_b2_e030_smoke_v3.sh"


def test_v3_full_launcher_fails_closed_and_records_service_identity():
    source = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "port_must_be_unbound",
        "/v1/models",
        "served-model-name",
        "MODEL_REVISION",
        "E030_ALLOW_FULL_RETRY",
        "technical_failure_receipt.json",
        "listener",
    ):
        assert required in source
    assert "PORT=\"${E030_VLLM_PORT:-8030}\"" not in source


def test_v3_smoke_launcher_checks_the_same_identity_without_judging():
    source = SMOKE.read_text(encoding="utf-8")

    for required in ("/v1/models", "MODEL_REVISION", "smoke_receipt.json", "port_must_be_unbound"):
        assert required in source
    assert "110_run_b2_e030_llm_judge.py" not in source
