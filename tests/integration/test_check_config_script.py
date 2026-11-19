"""End-to-end tests for scripts/check_config.py (pre-launch config validation).

Runs the script as a subprocess (same pattern as the fit_fast_tokenizer tool
test): composes real hydra configs, so this pins the user-facing contract —
exit 0 + merged yaml on success, exit 1 + hydra error on failure, override
passthrough, --out persistence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_config.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_experiment_path_composes_and_prints_merged_config():
    result = _run("configs/experiment/qwen_oft.yaml")
    assert result.returncode == 0, result.stderr
    assert "[check_config] architecture -> vla_system" in result.stderr
    assert "variant: oftp" in result.stdout
    assert "qwen_oft_robotwin" in result.stdout
    # base blocks inherited from train.yaml
    assert "learning_rate: 0.0001" in result.stdout


def test_out_flag_persists_loadable_yaml():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "merged.yaml"
        result = _run("configs/experiment/wam.yaml", "--out", str(out))
        assert result.returncode == 0, result.stderr
        assert out.exists()
        merged = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert merged["model"]["architecture"]["framework"] == "dual_system"
        assert merged["project"]["name"] == "wam_robotwin"


def test_override_passthrough_lands_in_merged_output():
    result = _run("configs/experiment/qwen_pi.yaml", "training.batch_size=4")
    assert result.returncode == 0, result.stderr
    assert "batch_size: 4" in result.stdout
    assert "variant: pi" in result.stdout


def test_unknown_experiment_fails_with_nonzero_exit():
    result = _run("+experiment=no_such_experiment")
    assert result.returncode == 1
    assert "FAILED" in result.stderr


def test_path_outside_the_workflow_is_rejected():
    result = _run("/tmp/not_a_launch_config.yaml")
    assert result.returncode != 0
    assert "is not a launch config" in (result.stdout + result.stderr)
