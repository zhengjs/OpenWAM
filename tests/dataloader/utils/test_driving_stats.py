"""``dataset_statistics.json`` loader (spec_data_zh.md §2 driving stats contract).

Covers: happy paths for all three modes with neutral fill of unconsumed
vectors, and the validation errors (unknown mode, missing blocks/dims, wrong
shape, non-finite values, ordering violations).
"""

import json

import pytest

from openwam.dataloader.utils.driving_stats import load_dataset_statistics


def write_stats(tmp_path, payload):
    path = tmp_path / "dataset_statistics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_payload(mode="q99"):
    return {
        "mode": mode,
        "ego_state": {"q01": [0.0] * 6, "q99": [1.0] * 6,
                      "min": [0.0] * 6, "max": [1.0] * 6},
        "trajectory": {"q01": [0.0] * 3, "q99": [1.0] * 3,
                       "min": [0.0] * 3, "max": [1.0] * 3},
        "T_hist": 1, "D_state": 6, "T_pred": 40, "D_action": 3,
    }


@pytest.mark.parametrize("mode", ["q99", "min_max", "mean_std"])
def test_load_all_modes_with_neutral_fill(tmp_path, mode):
    payload = valid_payload(mode)
    # only the mode-relevant vectors are supplied; the rest fill neutrally
    mode_vectors = {"q99": {"q01": [0.0] * 6, "q99": [1.0] * 6},
                    "min_max": {"min": [0.0] * 6, "max": [1.0] * 6},
                    "mean_std": {"mean": [0.0] * 6, "std": [1.0] * 6}}
    payload["ego_state"] = mode_vectors[mode]
    payload["trajectory"] = {k: v[:3] for k, v in mode_vectors[mode].items()}
    stats = load_dataset_statistics(write_stats(tmp_path, payload))

    assert stats["mode"] == mode
    assert stats["ego_state"]["q99"].shape == (6,)
    assert stats["ego_state"]["mean"].tolist() == [0.0] * 6  # neutral fill
    # trajectory min: supplied under min_max, neutral -1 fill otherwise
    expected_min = {"q99": -1.0, "min_max": 0.0, "mean_std": -1.0}[mode]
    assert stats["trajectory"]["min"].tolist() == [expected_min] * 3
    assert (stats["T_hist"], stats["D_state"], stats["T_pred"], stats["D_action"]) == (1, 6, 40, 3)


def test_missing_mode_is_unknown(tmp_path):
    payload = valid_payload()
    payload["mode"] = "q95"
    with pytest.raises(ValueError, match="unknown mode"):
        load_dataset_statistics(write_stats(tmp_path, payload))


def test_missing_block_and_dim_keys(tmp_path):
    payload = valid_payload()
    del payload["trajectory"]
    with pytest.raises(ValueError, match="missing stats block 'trajectory'"):
        load_dataset_statistics(write_stats(tmp_path, payload))

    payload = valid_payload()
    del payload["T_pred"]
    with pytest.raises(ValueError, match="missing 'T_pred'"):
        load_dataset_statistics(write_stats(tmp_path, payload))


def test_wrong_vector_shape(tmp_path):
    payload = valid_payload()
    payload["ego_state"]["q01"] = [0.0] * 5
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        load_dataset_statistics(write_stats(tmp_path, payload))


def test_nonfinite_values(tmp_path):
    payload = valid_payload()
    payload["trajectory"]["q99"] = [float("nan")] * 3
    with pytest.raises(ValueError, match="non-finite"):
        load_dataset_statistics(write_stats(tmp_path, payload))


def test_ordering_violation(tmp_path):
    payload = valid_payload()
    payload["trajectory"]["q01"] = [2.0] * 3
    payload["trajectory"]["q99"] = [1.0] * 3
    with pytest.raises(ValueError, match="q01 <= q99"):
        load_dataset_statistics(write_stats(tmp_path, payload))
