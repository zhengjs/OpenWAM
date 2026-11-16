"""``tools/ingest_lance.py`` dry-run on a synthetic storage root.

The lance WRITE path needs the lance package (absent on robot-only hosts);
the dry-run covers anchor matching, ego-frame GT projection, instruction
folding, and dataset_statistics.json emission — everything except the final
``lance.write_table``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.conftest_driving import build_synthetic_clip

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def storage(tmp_path_factory):
    return build_synthetic_clip(tmp_path_factory.mktemp("ingest"))


def run_ingest(root, *extra):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "ingest_lance.py"),
         "--storage-root", str(root), "--dataset-name", "dsA",
         "--cams", "cam_front", "--t-obs", "4", "--t-pred", "5",
         "--dt-pred-us", "200000", "--anchor-stride-us", "1000000",
         "--rot6d-source", "yaw_quat", "--dry-run", *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_dry_run_writes_stats_and_reports_samples(storage):
    result = run_ingest(storage["root"])
    assert result.returncode == 0, result.stderr
    assert "[dry-run] 5 samples" in result.stdout  # 5s clip @ 1s anchor stride

    stats = json.loads((storage["root"] / "dataset_statistics.json").read_text())
    # speed + yaw_rate + rot6d(6) after the quat conversion
    assert stats["D_state"] == 8
    assert (stats["T_hist"], stats["T_pred"], stats["D_action"]) == (1, 5, 3)
    for key in ("min", "max", "q01", "q99", "mean", "std"):
        assert len(stats["ego_state"][key]) == 8
        assert len(stats["trajectory"][key]) == 3


def test_no_lance_write_in_dry_run(storage):
    run_ingest(storage["root"])
    assert not (storage["root"] / "lance_indices").exists()
