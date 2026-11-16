"""Shared fixtures for preprocessor tests: a tiny dataset_statistics.json."""

import json

import pytest


@pytest.fixture
def stats_path(tmp_path):
    """min_max stats with D_state=4 / D_action=3."""

    def _write(mode="min_max", d_state=4, d_action=3, ego_vectors=None):
        payload = {
            "mode": mode,
            "ego_state": ego_vectors or {"min": [0.0] * d_state, "max": [1.0] * d_state},
            "trajectory": {"min": [-1.0] * d_action, "max": [1.0] * d_action},
            "T_hist": 2, "D_state": d_state, "T_pred": 3, "D_action": d_action,
        }
        path = tmp_path / "dataset_statistics.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    return _write
