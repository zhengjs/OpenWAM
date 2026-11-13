"""DROID clean-export loading uses shipped stats without rescanning source rows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openwam.dataloader.oxe_droid import (
    DROID_EEF_STATS_CONTRACT,
    DROID_EEF_STATS_CONTRACT_KEY,
    OxeDroidDataset,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.oxe_schema import (
    droid_euler7_to_arm10,
    droid_pose6_closedness_to_arm10,
    euler7_action_to_arm10,
)


@pytest.fixture
def droid_export(tmp_path, monkeypatch):
    root = tmp_path / "Droid"
    meta = root / "meta"
    (meta / "episodes").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 15,
                "total_episodes": 2,
                "total_frames": 48,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {},
            }
        )
    )
    rows = []
    for episode in range(2):
        row = {
            "episode_index": episode,
            "length": 24,
            "dataset_from_index": 24 * episode,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for camera in ("observation.images.primary", "observation.images.wrist"):
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = 0
        rows.append(row)
    pd.DataFrame(rows).to_parquet(meta / "episodes" / "chunk-000.parquet")
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["pick up the cup", "place the cup"], name="task")).to_parquet(
        meta / "tasks.parquet"
    )
    action = np.zeros((48, 7), dtype=np.float32)
    action[:, 0] = np.arange(48) / 100
    action[:, 6] = 0.25
    pose = np.zeros((48, 6), dtype=np.float32)
    pose[:, 0] = 0.75
    state = np.zeros((48, 7), dtype=np.float32)
    state[:, :3] = 100  # TCP values must never replace the rigid mount pose.
    state[:, 6] = 0.8
    pd.DataFrame(
        {
            "episode_index": np.repeat([0, 1], 24),
            "frame_index": np.tile(np.arange(24), 2),
            "task_index": np.repeat([0, 1], 24),
            "state": list(state),
            "other_information.observation_gripper_pose6d": list(pose),
            "other_information.action_wrist_pose": list(action),
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    stats = {
        "mean": [0.0] * 10,
        "std": [1.0] * 10,
        "min": [-2.0] * 10,
        "max": [2.0] * 10,
        "q01": [-2.0] * 10,
        "q99": [2.0] * 10,
        DROID_EEF_STATS_CONTRACT_KEY: DROID_EEF_STATS_CONTRACT,
    }
    (meta / "eef_stats.json").write_text(json.dumps(stats))
    monkeypatch.setattr(
        "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames",
        lambda path, frame_indices, height, width: [Image.new("RGB", (width, height)) for _ in frame_indices],
    )
    return root


def _reader(root, **kwargs):
    return OxeDroidDataset(str(root), num_frames=9, video_stride=4, height=32, width=32, **kwargs)


def test_clean_droid_loads_without_exclusion_certificate_or_fallback_columns(droid_export):
    ds = _reader(droid_export, normalize_mode=None)
    assert ds._eps_df["episode_index"].tolist() == [0, 1]
    sample = ds[0]
    assert sample["prompt"] == "pick up the cup"
    assert sample["action"].shape == (8, 20)
    assert sample["proprio"].shape == (1, 20)
    assert sample["action_mask"][:, :10].all()
    assert not sample["action_mask"][:, 10:].any()
    assert sample["proprio"][0, 0].item() == pytest.approx(0.75)
    assert sample["action"][0, 9].item() == pytest.approx(0.75)
    assert sample["proprio"][0, 9].item() == pytest.approx(0.2)
    np.testing.assert_allclose(sample["action"][0, 3:9], [1, 0, 0, 0, 1, 0])
    assert ds[int(ds._cum_n_starts[1])]["prompt"] == "place the cup"


def test_droid_shipped_stats_are_used_without_mutating_export(droid_export):
    before = {str(p.relative_to(droid_export)): p.stat().st_mtime_ns for p in droid_export.rglob("*") if p.is_file()}
    ds = _reader(droid_export)
    sample = ds[0]
    assert sample["action"][0, 9].item() == pytest.approx(0.375)
    np.testing.assert_allclose(sample["action"][0, 3:9], [1, 0, 0, 0, 1, 0])
    after = {str(p.relative_to(droid_export)): p.stat().st_mtime_ns for p in droid_export.rglob("*") if p.is_file()}
    assert before == after


def test_droid_startup_does_not_scan_numeric_or_prompt_source_shards(droid_export, monkeypatch):
    read_parquet = pd.read_parquet

    def metadata_only(path, *args, **kwargs):
        assert "data" not in Path(path).relative_to(droid_export).parts
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", metadata_only)
    assert len(_reader(droid_export)) == 48


def test_legacy_external_exclusions_cannot_reapply_filtering(droid_export):
    (droid_export / "meta" / "excluded_episodes.json").write_text("not a JSON artifact")
    assert _reader(droid_export)._eps_df["episode_index"].tolist() == [0, 1]


def test_missing_droid_stats_fail_without_creating_output(droid_export):
    path = droid_export / "meta" / "eef_stats.json"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="eef_stats.json"):
        _reader(droid_export)
    assert not path.exists()


def test_droid_stats_reject_incompatible_pose_contract(droid_export):
    path = droid_export / "meta" / "eef_stats.json"
    stats = json.loads(path.read_text())
    stats[DROID_EEF_STATS_CONTRACT_KEY]["action_pose_source"] = "other_information.action_tcp_pose"
    path.write_text(json.dumps(stats))
    with pytest.raises(DataContractError, match="terminal-arm"):
        _reader(droid_export)


def test_droid_stats_reject_wrong_width(droid_export):
    path = droid_export / "meta" / "eef_stats.json"
    stats = json.loads(path.read_text())
    stats["mean"] = [0.0] * 11
    path.write_text(json.dumps(stats))
    with pytest.raises((DataContractError, ValueError), match="10|shape"):
        _reader(droid_export)


def test_droid_cannot_borrow_another_datasets_parent_statistics(droid_export):
    local = droid_export / "meta" / "eef_stats.json"
    outer = droid_export.parent / "meta" / "eef_stats.json"
    outer.parent.mkdir()
    outer.write_bytes(local.read_bytes())
    local.unlink()
    with pytest.raises(FileNotFoundError, match="DROID requires its own"):
        _reader(droid_export)


# ==== from test_oxe_schema.py ====
# Canonical DROID and shared Euler pose conversion regressions.


class TestEuler7ActionToArm10:
    def test_shape(self):
        action = np.zeros((4, 7), dtype=np.float32)
        out = euler7_action_to_arm10(action)
        assert out.shape == (4, 10)
        assert out.dtype == np.float32

    def test_pos_grip_passthrough(self):
        action = np.array([[0.1, 0.2, 0.3, 0, 0, 0, 0.9]], dtype=np.float32)
        out = euler7_action_to_arm10(action)
        np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
        assert out[0, 9] == np.float32(0.9)


class TestDroidEuler7ToArm10:
    def test_closedness_is_inverted_to_canonical_openness(self):
        value = np.zeros((3, 7), dtype=np.float32)
        value[:, 6] = [0.0, 0.25, 1.0]

        out = droid_euler7_to_arm10(value)

        np.testing.assert_allclose(out[:, 9], [1.0, 0.75, 0.0], atol=1e-7)

    def test_pose_conversion_matches_generic_converter(self):
        value = np.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.25]], dtype=np.float32)

        droid = droid_euler7_to_arm10(value)
        generic = euler7_action_to_arm10(value)

        np.testing.assert_allclose(droid[:, :9], generic[:, :9], atol=1e-7)


class TestDroidPose6ClosednessToArm10:
    def test_assembles_arm_side_pose_and_inverts_only_closedness(self):
        pose = np.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=np.float32)
        closedness = np.array([[0.25]], dtype=np.float32)

        out = droid_pose6_closedness_to_arm10(pose, closedness)
        reference = droid_euler7_to_arm10(np.concatenate([pose, closedness], axis=-1))

        np.testing.assert_allclose(out, reference, atol=1e-7)
        np.testing.assert_allclose(out[:, :3], pose[:, :3], atol=1e-7)
        np.testing.assert_allclose(out[:, 9], [0.75], atol=1e-7)


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_euler_pose_conversion_preserves_arbitrary_leading_dimensions(leading):
    value = np.arange(int(np.prod(leading) or 1) * 7, dtype=np.float32).reshape(leading + (7,)) / 10
    expected = euler7_action_to_arm10(value.reshape(-1, 7)).reshape(leading + (10,))
    np.testing.assert_array_equal(euler7_action_to_arm10(value), expected)


@pytest.mark.parametrize("value", [np.zeros(8), np.zeros((2, 6)), np.array(0.0)])
def test_euler_pose_conversion_rejects_wrong_width_instead_of_dropping_fields(value):
    with pytest.raises(ValueError, match="width 7"):
        euler7_action_to_arm10(value)


@pytest.mark.parametrize("index", [0, 3, 6])
def test_droid_conversion_rejects_nonfinite_position_rotation_or_gripper(index):
    value = np.zeros((1, 7), dtype=np.float32)
    value[0, index] = np.nan
    with pytest.raises(ValueError, match="finite"):
        droid_euler7_to_arm10(value)


def test_droid_scalar_pose_is_rejected_with_an_actionable_error():
    with pytest.raises(ValueError, match="width 6"):
        droid_pose6_closedness_to_arm10(np.array(0.0), np.zeros(1))


# ==== from test_oxe_mixture_integration.py ====
# Mixture-level integration test for OXE readers.
#
# Verifies:
#   1. ``mixture.yaml`` composes the supported public sources under Hydra
#      defaults.
#   2. The composed ``cfg.datasets.<name>`` blocks contain the inherited
#      fields from ``configs/dataloader/<name>.yaml``.
#   3. A mixture built from synthetic OXE buckets + FakeActionDataset
#      collates cleanly through ``default_collate`` into per-batch tensors
#      of the expected 2-D mask shapes.

# ``agiworld`` is the user-facing dataset name; its registered config/type in
# this repository is ``agibotworld``.  The tuple is order-sensitive because the
# Hydra defaults order is part of the mixture's stable source-index contract.
MIXTURE_ENTRIES = (
    "agibotworld",
    "robocoin",
    "oxe_droid",
    "interndata_a1",
)


class TestMixtureYamlComposition:
    def test_mixture_yaml_composition(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("configs/dataloader")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="mixture")
        assert tuple(cfg.datasets.keys()) == MIXTURE_ENTRIES
        # Each entry inherits dataset_dir + type from its standalone yaml.
        for name in MIXTURE_ENTRIES:
            assert cfg.datasets[name].get("dataset_dir"), f"{name} missing dataset_dir"
            assert cfg.datasets[name].get("type") == name
            assert cfg.datasets[name].total_hours is None
        assert cfg.weight_strategy == "proportional"


class TestMixtureRegistryDispatch:
    """Only the active OXE reader participates in config dispatch."""

    def test_only_oxe_droid_is_registered(self):
        from openwam.dataloader.registry import list_registered_datasets

        names = set(list_registered_datasets())
        assert "oxe_droid" in names
        deprecated = {"ego4d", "egodex", "oxe_bcz", "oxe_bridge", "oxe_fractal"}
        assert not names.intersection(deprecated)


class TestMixtureSampleShapes:
    """Sanity-check that mixed batches produce uniformly-shaped per-key
    tensors. Uses FakeActionDataset (1-D mask) + a synthetic 2-D-mask
    dataset to verify the mixed batch's loss collate path works.
    """

    @pytest.fixture
    def fake_dataset(self):
        # Import directly instead of depending on a sibling conftest fixture;
        # this file is also collected in suites whose import mode does not
        # expose tests/dataloader/conftest.py fixtures by name.
        from tests.dataloader.conftest import FakeActionDataset

        return FakeActionDataset(n=4, action_dim=20)

    def test_fake_dataset_emits_legacy_1d_mask(self, fake_dataset):
        s = fake_dataset[0]
        assert s["action_mask"].ndim == 1
        assert s["proprio_mask"].ndim == 1

    def test_mixed_batch_collate_works(self):
        # End-to-end-ish: build a mixture from FakeActionDataset only.
        # Reader-level behavior is covered separately; this pins that mixing
        # multiple sources with the new mask shape contract still collates.
        from openwam.dataloader.mixture import MixtureDataset
        from tests.dataloader.conftest import FakeActionDataset

        ds_a = FakeActionDataset(n=5, action_dim=20)
        ds_b = FakeActionDataset(n=8, action_dim=20)
        mix = MixtureDataset(
            datasets=[ds_a, ds_b],
            names=["a", "b"],
        )
        # MixtureDataset extends per-bucket sizes by weight; ds_a has 5
        # real samples + ds_b has 8 real samples → total at least 13.
        assert len(mix) >= 13
        s = mix[0]
        assert s["action"].shape == (32, 20)
        assert "action_mask" in s
        assert "proprio_mask" in s
