from __future__ import annotations

import os
import subprocess
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_telecom_reproducibility.sh"


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_submit_uses_remote_home_defaults_for_data_and_python(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    remote_home = tmp_path / "remote-home"
    (remote_home / "repo_data").mkdir(parents=True)
    python_path = remote_home / "work/.venv-benchmark/bin/python"
    python_path.parent.mkdir(parents=True)
    _executable(python_path, "#!/usr/bin/env bash\nexit 0\n")
    _executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "while [[ $1 == -o ]]; do shift 2; done\n"
        "shift\n"
        "remote_command=\"$*\"\n"
        "exec bash -c \"$remote_command\"\n",
    )
    _executable(
        fake_bin / "sbatch",
        "#!/usr/bin/env bash\n"
        "for argument in \"$@\"; do\n"
        "  if [[ $argument == --export=ALL,* ]]; then\n"
        "    IFS=, read -r _ repo_var data_var python_var <<< \"$argument\"\n"
        "    LKG_REPO=${repo_var#LKG_REPO=}\n"
        "    LKG_DATA_ROOT=${data_var#LKG_DATA_ROOT=}\n"
        "    LKG_PYTHON=${python_var#LKG_PYTHON=}\n"
        "  fi\n"
        "done\n"
        "printf '%s|%s|%s|%s\\n' \"$LKG_REPO\" \"$LKG_DATA_ROOT\" \"$LKG_PYTHON\" \"${*: -1}\"\n",
    )
    (remote_home / "repo").mkdir()
    environment = {
        **os.environ,
        "HOME": str(remote_home),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REMOTE_HOST": "fake-host",
        "REMOTE_REPO": "repo",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER), "submit-ppr-audit"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == (
        f"{remote_home}/repo|{remote_home}/repo_data|"
        f"{remote_home}/work/.venv-benchmark/bin/python|"
        f"{remote_home}/repo/05-Technique/benchmark/etape1_embedding_pur/"
        "scripts/sbatch_ppr_final_audit.sh"
    )
