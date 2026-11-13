"""Tests for the compact RoboCasa365 state19/action15 reader."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import openwam.dataloader.robocasa365 as rc
from openwam.dataloader.robocasa365 import ACTION_DIM, STATE_DIM, RoboCasa365Dataset

EP_LENGTH = 40
PROMPT = "Open the right drawer."
HEAD = rc.HEAD_CAMERA
WRIST = rc.WRIST_CAMERA
RIGHT = rc.RIGHT_CAMERA


def _stats(dim: int) -> dict:
    return {
        "mean": np.zeros(dim, np.float32),
        "std": np.ones(dim, np.float32),
        "min": -np.ones(dim, np.float32),
        "max": np.ones(dim, np.float32),
        "q01": -np.ones(dim, np.float32),
        "q99": np.ones(dim, np.float32),
    }


def _make_repo(root: Path, tasks=("OpenDrawer",), episodes_per_task: int = 2) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    total_episodes = len(tasks) * episodes_per_task
    info = {
        "codebase_version": "v3.0",
        "robot_type": "PandaOmron",
        "fps": 20,
        "total_episodes": total_episodes,
        "total_frames": total_episodes * EP_LENGTH,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            HEAD: {"dtype": "video", "shape": [256, 256, 3]},
            WRIST: {"dtype": "video", "shape": [256, 256, 3]},
            RIGHT: {"dtype": "video", "shape": [256, 256, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    with (root / "meta" / "robocasa365_normalization_stats.npy").open("wb") as handle:
        np.save(handle, {rc.ACTION_STATS_KEY: _stats(ACTION_DIM), rc.STATE_STATS_KEY: _stats(STATE_DIM)})

    state_rows, action_rows, episode_rows = [], [], []
    offset = episode_index = 0
    for task_index, task in enumerate(tasks):
        for _ in range(episodes_per_task):
            state = np.zeros((EP_LENGTH, STATE_DIM), np.float32)
            state[:, 0] = np.arange(EP_LENGTH) + 100 * episode_index
            state[:, 3] = state[:, 7] = 1.0
            state[:, 9] = 1.0
            state[:, 10:13] = [1.0, 2.0, 3.0]
            state[:, 13] = state[:, 17] = 1.0
            action = np.zeros((EP_LENGTH, ACTION_DIM), np.float32)
            action[:, 0] = state[:, 0] + 0.025
            action[:, 3] = action[:, 7] = 1.0
            action[:, 9] = -1.0
            action[:, 10:13] = [0.1, -0.2, 0.3]
            action[:, 13] = 0.0
            action[:, 14] = np.where(np.arange(EP_LENGTH) % 2 == 0, -1.0, 1.0)
            state_rows.extend(state)
            action_rows.extend(action)
            row = {
                "episode_index": episode_index,
                "dataset_from_index": offset,
                "dataset_to_index": offset + EP_LENGTH,
                "length": EP_LENGTH,
                "tasks": [PROMPT],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "source_prefix": f"pretrain/atomic/{task}/20250820",
                "source_episode_index": episode_index,
            }
            for camera in (HEAD, WRIST, RIGHT):
                row[f"videos/{camera}/chunk_index"] = 0
                row[f"videos/{camera}/file_index"] = 0
                row[f"videos/{camera}/from_timestamp"] = offset / 20
                row[f"videos/{camera}/to_timestamp"] = (offset + EP_LENGTH) / 20
            episode_rows.append(row)
            offset += EP_LENGTH
            episode_index += 1
    pd.DataFrame({"observation.state": state_rows, "action": action_rows}).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet"
    )
    pd.DataFrame(episode_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for camera in (HEAD, WRIST, RIGHT):
        path = root / "videos" / camera / "chunk-000"
        path.mkdir(parents=True)
        (path / "file-000.mp4").write_bytes(b"")
    return root


@contextmanager
def _mock_video():
    def decode(path, frame_indices, height, width):
        if RIGHT in path:
            color = (0, 0, 255)
        elif WRIST in path:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        return [Image.new("RGB", (width, height), color) for _ in frame_indices]

    with patch.object(rc, "decode_video_frames", side_effect=decode):
        yield


def _dataset(tmp_path: Path, **kwargs) -> RoboCasa365Dataset:
    root = _make_repo(tmp_path / "repo")
    kwargs.setdefault("task_name", "OpenDrawer")
    kwargs.setdefault("normalize_mode", None)
    kwargs.setdefault("multiview", False)
    kwargs.setdefault("height", 64)
    kwargs.setdefault("width", 96)
    return RoboCasa365Dataset(str(root), **kwargs)


def test_compact_nonunified_shapes_and_same_row_action(tmp_path):
    with _mock_video():
        dataset = _dataset(tmp_path)
        sample = dataset._build_sample(0, 1)
    assert sample["proprio"].shape == (1, STATE_DIM)
    assert sample["action"].shape == (32, ACTION_DIM)
    # State row 1 is x=1 and action row 1 is target x=1.025. A next-state
    # implementation would have produced/read x around 2 here.
    np.testing.assert_allclose(sample["proprio"][0, 0], 1.0)
    np.testing.assert_allclose(sample["action"][0, 0], 1.025)


def test_unified_maps_and_masks_are_asymmetric(tmp_path):
    with _mock_video():
        dataset = _dataset(
            tmp_path,
            unify_action=True,
            unify_action_map=["0-9", "68-72"],
            unify_state_map=["0-9", "68-76"],
        )
        sample = dataset[0]
    assert sample["action"].shape == (32, 80)
    assert sample["proprio"].shape == (1, 80)
    action_mask = sample["action_mask"][0].numpy()
    state_mask = sample["proprio_mask"][0].numpy()
    assert action_mask.sum() == 15 and action_mask[:10].all() and action_mask[68:73].all()
    assert state_mask.sum() == 19 and state_mask[:10].all() and state_mask[68:77].all()
    assert action_mask[71]  # torso is retained
    assert not action_mask[73:77].any()
    assert state_mask[73:77].all()


def test_action_and_state_values_land_in_requested_slots(tmp_path):
    with _mock_video():
        dataset = _dataset(tmp_path, unify_action=True)
        sample = dataset[0]
    action = sample["action"][0].numpy()
    state = sample["proprio"][0].numpy()
    np.testing.assert_allclose(action[68:73], [0.1, -0.2, 0.3, 0.0, -1.0])
    np.testing.assert_allclose(state[68:71], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state[71:77], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_separate_normalization_and_denormalization(tmp_path):
    root = _make_repo(tmp_path / "repo")
    with _mock_video():
        dataset = RoboCasa365Dataset(
            str(root),
            task_name="OpenDrawer",
            normalize_mode="min-max",
            multiview=False,
            height=64,
            width=96,
            unify_action=True,
        )
        sample = dataset[0]
    raw = dataset.denormalize_action(sample["action"].numpy())
    assert raw.shape == (32, ACTION_DIM)
    np.testing.assert_allclose(raw[0, 10:15], [0.1, -0.2, 0.3, 0.0, -1.0], atol=1e-6)


def test_prompt_and_video_contract(tmp_path):
    with _mock_video():
        dataset = _dataset(tmp_path, multiview=True, height=384, width=320)
        sample = dataset[0]
    assert sample["prompt"] == PROMPT
    assert len(sample["video"]) == 9
    assert sample["video"][0].size == (320, 384)
    pixels = np.asarray(sample["video"][0])
    np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(pixels[300, 0], [0, 255, 0])
    np.testing.assert_array_equal(pixels[300, 319], [0, 0, 255])


def test_rejects_native_unconverted_schema(tmp_path):
    root = _make_repo(tmp_path / "repo")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["shape"] = [16]
    info["features"]["action"]["shape"] = [12]
    info_path.write_text(json.dumps(info))
    try:
        RoboCasa365Dataset(str(root), normalize_mode=None)
    except ValueError as error:
        assert "state19/action15" in str(error)
    else:
        raise AssertionError("native schema should be rejected")


def test_multiview_rejects_dataset_without_right_agentview(tmp_path):
    root = _make_repo(tmp_path / "repo")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"][RIGHT]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="robot0_agentview_right"):
        RoboCasa365Dataset(str(root), normalize_mode=None, multiview=True)


# ==== from test_robocasa365_config.py ====
# Config-parsing tests for benchmarks/robocasa365/single_eval.py.
#
# Only the pure config helpers are tested here; ``run_eval`` needs the robosuite
# sim (separate env) and is exercised by the user-run smoke. ``single_eval.py`` is
# loaded under a unique module name so it does not collide with the identically
# named ``benchmarks/libero/single_eval.py`` in the same pytest session.

# parents[] index adjusted 1 -> 2: this block moved one directory deeper
# (tests/ -> tests/dataloader/).
_SE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "robocasa365" / "single_eval.py"
# single_eval.py does `from openwam2robocasa365_interface import ...`, so its dir
# must be importable.
if str(_SE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_SE_PATH.parent))
_spec = importlib.util.spec_from_file_location("robocasa365_single_eval", _SE_PATH)
single_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(single_eval)


def test_parse_bool_variants():
    assert single_eval._parse_bool(True, "x") is True
    assert single_eval._parse_bool("yes", "x") is True
    assert single_eval._parse_bool("On", "x") is True
    assert single_eval._parse_bool("off", "x") is False
    assert single_eval._parse_bool("", "x") is False
    with pytest.raises(ValueError):
        single_eval._parse_bool("maybe", "x")


def test_normalize_optional():
    assert single_eval._normalize_optional("none") is None
    assert single_eval._normalize_optional("null") is None
    assert single_eval._normalize_optional("") is None
    assert single_eval._normalize_optional("video.robot0_agentview_left") == "video.robot0_agentview_left"
    assert single_eval._normalize_optional(None) is None


def test_parse_optional_int():
    assert single_eval._parse_optional_int("16", "x") == 16
    assert single_eval._parse_optional_int(16, "x") == 16
    assert single_eval._parse_optional_int("none", "x") is None
    with pytest.raises(ValueError):
        single_eval._parse_optional_int("0", "x")
    with pytest.raises(ValueError):
        single_eval._parse_optional_int("-3", "x")


def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("task: OpenDrawer\nport: 8848\nstate_dim: 16\n", encoding="utf-8")
    cfg = single_eval._load_config(p)
    assert cfg["task"] == "OpenDrawer"
    assert cfg["port"] == 8848
    assert cfg["state_dim"] == 16


def test_resolve_max_steps_uses_official_robocasa_registry(monkeypatch):
    import types

    registry_utils = types.ModuleType("robocasa.utils.dataset_registry_utils")
    registry_utils.get_task_horizon = lambda task: {"OpenDrawer": 750}[task]
    monkeypatch.setitem(sys.modules, "robocasa", types.ModuleType("robocasa"))
    monkeypatch.setitem(sys.modules, "robocasa.utils", types.ModuleType("robocasa.utils"))
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry_utils", registry_utils)

    assert single_eval._resolve_max_steps({"task": "OpenDrawer"}) == 750


def test_resolve_max_steps_explicit_smoke_override_needs_no_robocasa():
    assert single_eval._resolve_max_steps({"task": "OpenDrawer", "max_steps_override": 12}) == 12
    with pytest.raises(ValueError, match="max_steps_override"):
        single_eval._resolve_max_steps({"task": "OpenDrawer", "max_steps_override": 0})


def test_rollout_accumulates_success_and_terminates():
    """_rollout drives env+policy via stubs (no sim/server): trial 0 succeeds at
    step 2, trial 1 never does -> 1 success; policy.act called the right #times.
    Mirrors robotwin's eval-loop test (stub env + stub policy)."""

    class _FakeEnv:
        def __init__(self):
            self.trial = -1
            self.t = 0

        def reset(self, seed=None):
            self.trial += 1
            self.t = 0
            return ({"annotation.human.task_description": "x"}, {"success": False})

        def step(self, action):
            self.t += 1
            done = self.trial == 0 and self.t >= 2  # only trial 0 succeeds, at step 2
            return ({"annotation.human.task_description": "x"}, 1.0 if done else 0.0, done, False, {"success": done})

    class _FakePolicy:
        def __init__(self):
            self.acts = 0

        def reset(self):
            pass

        def act(self, obs, prompt):
            self.acts += 1
            return {"action.dummy": 0}

    env, pol = _FakeEnv(), _FakePolicy()
    successes = single_eval._rollout(env, pol, num_trials=2, max_steps=5, seed=0)
    assert successes == 1
    assert pol.acts == 2 + 5  # trial 0: 2 steps then done; trial 1: 5 (max) steps


def test_rollout_uses_native_prompt():
    """The policy receives RoboCasa's task instruction without a wrapper."""
    seen = []

    class _FakeEnv:
        def reset(self, seed=None):
            return ({"annotation.human.task_description": "open the drawer"}, {})

        def step(self, action):
            return ({}, 0.0, True, False, {"success": False})

    class _FakePolicy:
        def reset(self):
            pass

        def act(self, obs, prompt):
            seen.append(prompt)
            return {}

    single_eval._rollout(_FakeEnv(), _FakePolicy(), num_trials=1, max_steps=3, seed=0)
    assert seen == ["open the drawer"]


def test_rollout_treats_info_success_as_terminal():
    class _FakeEnv:
        def __init__(self):
            self.steps = 0

        def reset(self, seed=None):
            return ({"annotation.human.task_description": "open the drawer"}, {})

        def step(self, action):
            self.steps += 1
            return ({}, 1.0, False, False, {"success": True})

    class _FakePolicy:
        def reset(self):
            pass

        def act(self, obs, prompt):
            return {}

    env = _FakeEnv()
    assert single_eval._rollout(env, _FakePolicy(), num_trials=1, max_steps=750, seed=0) == 1
    assert env.steps == 1


def test_prompt_template_preserves_native_instruction():
    """RoboCasa365 eval prompt handling must not add a prefix or suffix."""
    import prompt_template

    for s in ("open the drawer", "", "pick the apple from the counter and place it in the sink."):
        assert prompt_template.format_prompt_for_inference(s) == s


def test_build_policy_uses_compact_defaults(monkeypatch):
    captured = {}

    class _Capture:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(single_eval, "OpenWAMRoboCasa365Policy", _Capture)
    single_eval._build_policy({})
    assert captured["state_dim"] is None
    assert captured["right_wrist_camera_key"] == "video.robot0_agentview_right"
    assert "mobile_base" not in captured
    assert "base_proprio" not in captured
    assert "mask_torso_action" not in captured


def test_repo_template_declares_compact_contract():
    import yaml as _yaml

    tmpl = Path(single_eval.__file__).parent / "policy_config.yml"
    cfg = _yaml.safe_load(tmpl.read_text())
    assert cfg["state_dim"] == 19
    assert cfg["max_steps_override"] is None
    assert cfg["right_wrist_camera_key"] == "video.robot0_agentview_right"
    assert "max_steps" not in cfg
    assert "mask_torso_action" not in cfg


def test_hydra_compose_compact_maps_and_fixed_eval_semantics():
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["dataloader=robocasa365"])
        assert cfg.dataloader.action_mode == "eef"
        assert list(cfg.dataloader.unify_action_map) == ["0-9", "68-72"]
        assert list(cfg.dataloader.unify_state_map) == ["0-9", "68-76"]
        assert list(cfg.dataloader.camera_layout) == [
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_right",
        ]
        assert "binary_action_dims" not in cfg.dataloader
        assert "gripper_convention" not in cfg.dataloader
        assert "mask_torso_action" not in cfg.dataloader
        # dataset_dir is user-mutable (the benchmark downloader rewrites it in
        # place), so only require a single umbrella string, not a specific value.
        assert isinstance(cfg.dataloader.dataset_dir, str) and cfg.dataloader.dataset_dir
        assert cfg.dataloader.normalization_stats_path is None


def test_only_canonical_robocasa365_dataset_is_registered():
    from openwam.dataloader.registry import DATASET_REGISTRY
    from openwam.dataloader.robocasa365 import MultiTaskRoboCasa365Dataset

    assert DATASET_REGISTRY["robocasa365"] is MultiTaskRoboCasa365Dataset
