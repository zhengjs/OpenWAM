"""Tests for openwam/dataloader/mixture.py — MixtureDataset.

Uses FakeActionDataset (from conftest.FakeActionDataset) so no parquet/mp4
fixtures are needed; the tests target dispatch logic, weight strategies,
strict_action_dim, set_epoch, and the _dataset_name / get_dataset API.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.dataloader import registry as registry_mod
from openwam.dataloader.dataset.mixture_dataset import MixtureDataset
from openwam.train.openwam_trainer import OpenWAMTrainer

# ---------------------------------------------------------------------------
# Construction + basic invariants
# ---------------------------------------------------------------------------


class TestMixtureConstruction:
    def test_empty_buckets_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            MixtureDataset([])

    def test_manual_weight_validation_raises(self, fake_dataset_factory):
        """Regression (M2): all-zero weights must raise a clear error instead of
        a bare ZeroDivisionError from the weight normalization, and a negative
        weight must be rejected explicitly rather than silently treated as a
        zero-weight skip in _build_index_map."""
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        with pytest.raises(ValueError, match="zero-weight"):
            MixtureDataset([a, b], weights=[0.0, 0.0])
        with pytest.raises(ValueError, match="negative"):
            MixtureDataset([a, b], weights=[-1.0, 2.0])

    def test_length_is_sum_of_virtual_n(self, fake_dataset_factory):
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        m = MixtureDataset([a, b], weights=[1.0, 1.0])
        # With uniform weights both sources get round(total_real * 0.5) = 12.
        # 12 + 12 = 24; close to total_real but rounding may bias by 1.
        total_real = len(a) + len(b)
        assert abs(len(m) - total_real) <= 1

    def test_default_weights_proportional_full_coverage(self, fake_dataset_factory):
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(15)
        m = MixtureDataset([a, b], weights=None)
        # weights=None uses [len(a), len(b)] → proportional → virtual_n_i = N_i.
        assert len(m) == len(a) + len(b)

        # Every (di, si) pair appears at least once.
        seen = set()
        for k in range(len(m)):
            di, si = m._index_map[k]
            seen.add((int(di), int(si)))
        expected = {(0, i) for i in range(len(a))} | {(1, i) for i in range(len(b))}
        assert seen == expected

    def test_names_default_to_source_i(self, fake_dataset_factory):
        m = MixtureDataset([fake_dataset_factory(5), fake_dataset_factory(7)], weights=[1.0, 1.0])
        assert m.names == ["source_0", "source_1"]

    def test_named_buckets(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(5), fake_dataset_factory(7)],
            weights=[1.0, 1.0],
            names=["alpha", "beta"],
        )
        assert m.names == ["alpha", "beta"]

    def test_names_length_mismatch_raises(self, fake_dataset_factory):
        with pytest.raises(ValueError, match="names length"):
            MixtureDataset(
                [fake_dataset_factory(5), fake_dataset_factory(7)],
                weights=[1.0, 1.0],
                names=["only-one"],
            )


# ---------------------------------------------------------------------------
# Strict action_dim
# ---------------------------------------------------------------------------


class TestStrictActionDim:
    def test_strict_mismatch_raises(self, fake_dataset_factory):
        a = fake_dataset_factory(5, action_dim=20)
        b = fake_dataset_factory(7, action_dim=24)
        with pytest.raises(ValueError, match="same action_dim"):
            MixtureDataset([a, b], weights=[1.0, 1.0], strict_action_dim=True)

    def test_strict_match_ok(self, fake_dataset_factory):
        a = fake_dataset_factory(5, action_dim=20)
        b = fake_dataset_factory(7, action_dim=20)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], strict_action_dim=True)
        assert m.action_dim == 20

    def test_legacy_pad_to_max(self, fake_dataset_factory):
        # strict=False, dims differ → max dim wins.
        a = fake_dataset_factory(5, action_dim=14)
        b = fake_dataset_factory(7, action_dim=20)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], strict_action_dim=False)
        assert m.action_dim == 20

    def test_normalization_stats_always_none(self, fake_dataset_factory):
        # Even if sub-buckets carry a stats dict, mixture must NOT aggregate.
        a = fake_dataset_factory(5, normalization_stats={"mean": np.zeros(20), "std": np.ones(20)})
        b = fake_dataset_factory(7, normalization_stats={"mean": np.ones(20), "std": np.ones(20) * 2})
        m = MixtureDataset([a, b], weights=[1.0, 1.0])
        assert m.normalization_stats is None


# ---------------------------------------------------------------------------
# Dispatch + name propagation
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_sample_carries_dataset_index_and_name(self, fake_dataset_factory):
        a = fake_dataset_factory(5)
        b = fake_dataset_factory(7)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], names=["alpha", "beta"])
        s = m[0]
        assert s["_dataset_index"] in (0, 1)
        assert s["_dataset_name"] in ("alpha", "beta")
        # Name and index agree.
        assert s["_dataset_name"] == m.names[s["_dataset_index"]]

    def test_get_dataset_by_name(self, fake_dataset_factory):
        a = fake_dataset_factory(5)
        b = fake_dataset_factory(7)
        m = MixtureDataset([a, b], weights=[1.0, 1.0], names=["alpha", "beta"])
        assert m.get_dataset("alpha") is a
        assert m.get_dataset("beta") is b

    def test_get_dataset_unknown_name_raises_keyerror(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(5), fake_dataset_factory(7)],
            weights=[1.0, 1.0],
            names=["alpha", "beta"],
        )
        with pytest.raises(KeyError, match="no sub-dataset named 'gamma'"):
            m.get_dataset("gamma")

    def test_dataset_sample_counts_by_name(self, fake_dataset_factory):
        a = fake_dataset_factory(10)
        b = fake_dataset_factory(20)
        m = MixtureDataset([a, b], weights=None, names=["alpha", "beta"])  # proportional
        counts = m.dataset_sample_counts()
        # Exact counts under proportional weights.
        assert counts == {"alpha": 10, "beta": 20}


# ---------------------------------------------------------------------------
# set_epoch
# ---------------------------------------------------------------------------


class TestSetEpoch:
    def test_set_epoch_changes_order(self, fake_dataset_factory):
        a = fake_dataset_factory(50)
        b = fake_dataset_factory(50)
        m = MixtureDataset([a, b], weights=None, seed=42, names=["a", "b"])  # proportional
        order_0 = m._index_map.copy()
        m.set_epoch(1)
        order_1 = m._index_map.copy()
        # Same total length, different sample ordering.
        assert len(order_0) == len(order_1)
        assert not (order_0 == order_1).all()

    def test_set_epoch_preserves_coverage_under_proportional(self, fake_dataset_factory):
        a = fake_dataset_factory(50)
        b = fake_dataset_factory(70)
        m = MixtureDataset([a, b], weights=None, seed=42)  # proportional
        # Coverage at epoch 0
        seen_0 = {(int(d), int(s)) for d, s in m._index_map}
        m.set_epoch(5)
        seen_5 = {(int(d), int(s)) for d, s in m._index_map}
        # Both epochs cover the full union.
        full = {(0, i) for i in range(50)} | {(1, i) for i in range(70)}
        assert seen_0 == full
        assert seen_5 == full

    def test_set_epoch_idempotent_with_same_arg(self, fake_dataset_factory):
        a = fake_dataset_factory(30)
        m = MixtureDataset([a], weights=None, seed=42)
        initial_map = m._index_map
        m.set_epoch(0)
        assert m._index_map is initial_map

        m.set_epoch(3)
        epoch_3_map = m._index_map
        m.set_epoch(3)
        assert m._index_map is epoch_3_map


@pytest.mark.parametrize("seed", [0, 42])
def test_mixture_preserves_legacy_seeded_order_without_extra_maps(fake_dataset_factory, seed):
    datasets = [fake_dataset_factory(7), fake_dataset_factory(11)]
    mixture = MixtureDataset(datasets, weights=[1.0, 4.0], seed=seed)
    for epoch in (0, 3):
        mixture.set_epoch(epoch)
        parts = []
        for di, (dataset, weight) in enumerate(zip(datasets, mixture.weights)):
            count = max(1, round(18 * weight))
            parts.append(
                np.stack(
                    [
                        np.full(count, di, dtype=np.int32),
                        (np.arange(count, dtype=np.int64) % len(dataset)).astype(np.int32),
                    ],
                    axis=1,
                )
            )
        legacy = np.concatenate(parts)
        legacy = legacy[np.random.RandomState(seed + epoch * 7919).permutation(len(legacy))]
        np.testing.assert_array_equal(mixture._index_map, legacy)
        assert mixture._index_map.dtype == np.int32
        assert mixture.dataset_sample_counts() == {"source_0": 4, "source_1": 14}


@pytest.mark.parametrize("weights", [[float("nan"), 1.0], [float("inf"), 1.0], [1e308, 1e308]])
def test_mixture_rejects_nonfinite_weights(fake_dataset_factory, weights):
    with pytest.raises(ValueError, match="finite"):
        MixtureDataset([fake_dataset_factory(3), fake_dataset_factory(4)], weights=weights)


# ==== from test_mixture_from_config.py ====
# Tests for MixtureDataset.from_config dispatch (dict vs list datasets).
#
# These tests bypass the real RoboCOIN / EgoDex readers via a small
# monkeypatched registry — the focus is from_config's discovery logic
# and weight_strategy validation, not actual reader IO.


@pytest.fixture
def fake_registry(monkeypatch, fake_dataset_factory):
    """Patch build_dataset to return FakeActionDataset for known type names."""

    def _fake_build_dataset(cfg, split="train"):
        # cfg might be dict-like; pull optional "n" (test param) for sizing.
        n = cfg.get("n", 10) if hasattr(cfg, "get") else getattr(cfg, "n", 10)
        return fake_dataset_factory(int(n), action_dim=20)

    monkeypatch.setattr(registry_mod, "build_dataset", _fake_build_dataset)
    return _fake_build_dataset


class TestFromConfigDictDatasets:
    def test_dict_style_picks_names_from_keys(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "robocoin": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "egodex": {
                    "type": "egodex",
                    "enabled": True,
                    "weight": 0.5,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg, split="train")
        assert m.names == ["robocoin", "egodex"]
        assert pytest.approx(m.weights[0]) == 1.0 / 1.5
        assert pytest.approx(m.weights[1]) == 0.5 / 1.5

    def test_disabled_skipped(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "alpha": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "beta": {
                    "type": "egodex",
                    "enabled": False,
                    "weight": 1.0,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg, split="train")
        assert m.names == ["alpha"]

    def test_all_disabled_raises_clear_error(self, fake_registry):
        # Regression (M1): when every sub-source is disabled the friendly
        # "all sub-datasets are disabled" error must surface, not an opaque
        # ThreadPoolExecutor "max_workers must be greater than 0".
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "alpha": {
                    "type": "robocoin",
                    "enabled": False,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "beta": {
                    "type": "egodex",
                    "enabled": False,
                    "weight": 1.0,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(RuntimeError, match="disabled"):
            MixtureDataset.from_config(cfg, split="train")


class TestFromConfigListDatasets:
    def test_list_style_derives_names_from_type(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": [
                {
                    "type": "robotwin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                {
                    "type": "robotwin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            ],
        }
        m = MixtureDataset.from_config(cfg, split="train")
        # Duplicate types get numeric suffix.
        assert m.names == ["robotwin", "robotwin_1"]


class TestWeightStrategyValidation:
    def test_unknown_strategy_raises(self, fake_registry):
        cfg = {
            "weight_strategy": "garbage",
            "datasets": {
                "x": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(ValueError, match="unknown weight_strategy"):
            MixtureDataset.from_config(cfg)

    def test_legacy_token_name_in_error_message(self, fake_registry):
        cfg = {
            "weight_strategy": "token",  # renamed to inverse_size
            "datasets": {
                "x": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(ValueError, match="inverse_size"):
            MixtureDataset.from_config(cfg)


class TestShapeSanity:
    def test_num_frames_mismatch_raises(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "a": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "b": {
                    "type": "egodex",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 49,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(ValueError, match="same 'num_frames'"):
            MixtureDataset.from_config(cfg)


class TestActionDimStrictMode:
    def test_no_action_dim_override_field_uses_strict(self, fake_registry):
        # No action_dim_override field => strict_action_dim=True path.
        # All fake datasets have action_dim=20, should pass.
        cfg = {
            "weight_strategy": "proportional",
            "datasets": {
                "a": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg)
        assert m.action_dim == 20

    def test_action_dim_override_field_present_uses_legacy(self, fake_registry):
        cfg = {
            "weight_strategy": "proportional",
            "action_dim_override": None,  # presence opts into legacy path
            "datasets": {
                "a": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg)
        assert m.action_dim == 20


def test_invalid_strategy_fails_before_loading_sources(monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("invalid mixture configuration must not scan datasets")

    monkeypatch.setattr(registry_mod, "build_dataset", unexpected_load)
    with pytest.raises(ValueError, match="unknown weight_strategy"):
        MixtureDataset.from_config({"weight_strategy": "typo", "datasets": {"a": {"type": "robocoin"}}})


# ==== from test_mixture_trainer_dataloader.py ====
# Trainer sampling policy for the already-shuffled MixtureDataset.


def _bare_trainer(dataset) -> OpenWAMTrainer:
    """Construct only the fields consumed by ``build_dataloader``."""
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.dataset = dataset
    trainer.cfg = SimpleNamespace(training=SimpleNamespace(dataset_num_workers=0))
    trainer._run_seed = None
    trainer._rank = 0
    return trainer


def test_mixture_uses_sequential_sampler_over_its_shuffled_index_map(fake_dataset_factory):
    mixture = MixtureDataset(
        [fake_dataset_factory(8)],
        weights=None,
        names=["tiny"],
        strict_action_dim=True,
    )

    loader = _bare_trainer(mixture).build_dataloader(batch_size=2)

    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)


def test_non_mixture_keeps_random_sampler(fake_dataset_factory):
    loader = _bare_trainer(fake_dataset_factory(8)).build_dataloader(batch_size=2)

    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)


class TestWeightAnnealing:
    """Epoch-driven weight annealing (spec_data_zh.md §6.1)."""

    def test_linear_alpha_endpoints(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(100), fake_dataset_factory(100)],
            seed=1,
            anneal_start=[0.2, 1.0],
            anneal_end=[1.0, 0.5],
            anneal_schedule="linear",
            num_epochs=10,
        )
        assert abs(sum(m.weights) - 1.0) < 1e-9
        # alpha(0) = 0 -> weights at anneal_start
        assert abs(m.weights[0] - 0.2 / 1.2) < 1e-9
        m.set_epoch(9)
        # alpha(9) = 1 -> weights at anneal_end
        assert abs(m.weights[0] - 1.0 / 1.5) < 1e-9

    def test_linear_midpoint_matches_closed_form(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(100), fake_dataset_factory(100)],
            seed=1,
            anneal_start=[0.2, 1.0],
            anneal_end=[1.0, 0.5],
            anneal_schedule="linear",
            num_epochs=10,
        )
        m.set_epoch(5)
        a = 5 / 9
        raw0 = 0.2 + 0.8 * a
        raw1 = 1.0 - 0.5 * a
        assert abs(m.weights[0] - raw0 / (raw0 + raw1)) < 1e-9

    def test_cosine_alpha_endpoints(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(100), fake_dataset_factory(100)],
            seed=1,
            anneal_start=[0.2, 1.0],
            anneal_end=[1.0, 0.5],
            anneal_schedule="cosine",
            num_epochs=5,
        )
        m.set_epoch(4)  # alpha = 1 -> anneal_end exactly
        assert abs(m.weights[0] - 1.0 / 1.5) < 1e-9

    def test_horizon_injected_after_set_epoch(self, fake_dataset_factory):
        """Resume-style order: set_epoch first (static weights + warn), then
        set_num_epochs — the guard reset must let the next set_epoch rebuild."""
        m = MixtureDataset(
            [fake_dataset_factory(10), fake_dataset_factory(10)],
            seed=1,
            anneal_start=[0.0, 1.0],
            anneal_end=[1.0, 0.0],
            num_epochs=None,
        )
        assert m.weights[0] == 0.0
        m.set_epoch(3)  # warns once; weights stay static
        assert m.weights[0] == 0.0
        m.set_num_epochs(4)
        m.set_epoch(3)  # guard reset -> rebuild at alpha=1
        assert abs(m.weights[0] - 1.0) < 1e-9

    def test_zero_weight_source_rejoin_on_rebuild(self, fake_dataset_factory):
        """A source skipped at weight 0 (alpha=0) is re-included once its
        weight becomes positive and the index map is rebuilt."""
        m = MixtureDataset(
            [fake_dataset_factory(10), fake_dataset_factory(10)],
            seed=1,
            anneal_start=[0.0, 1.0],
            anneal_end=[1.0, 0.0],
            anneal_schedule="linear",
            num_epochs=2,
        )
        m.set_epoch(1)  # alpha=1: source_0 takes over, source_1 drops
        assert abs(m.weights[0] - 1.0) < 1e-9

    def test_virtual_sample_conservation_across_epochs(self, fake_dataset_factory):
        m = MixtureDataset(
            [fake_dataset_factory(100), fake_dataset_factory(100)],
            weights=[1.0, 1.0],
            seed=1,
            anneal_start=[0.2, 1.0],
            anneal_end=[1.0, 0.5],
            num_epochs=10,
        )
        sizes = [len(m) for _ in range(10)]
        m.set_epoch(7)
        assert abs(len(m) - sizes[0]) <= 2  # normalized weights conserve the total budget

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"anneal_start": [0.2], "anneal_end": [1.0, 0.5]}, "anneal_start"),
            ({"anneal_start": [0.2, -1.0], "anneal_end": [1.0, 0.5]}, "negative anneal_start"),
            ({"anneal_start": [0.2, 1.0], "anneal_end": [1.0, 0.5], "anneal_schedule": "step"}, "schedule"),
            ({"anneal_start": [0.2, 1.0], "anneal_end": [1.0, 0.5], "num_epochs": 0}, "num_epochs"),
        ],
    )
    def test_validation_errors(self, fake_dataset_factory, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MixtureDataset(
                [fake_dataset_factory(5), fake_dataset_factory(5)],
                seed=1,
                **kwargs,
            )
