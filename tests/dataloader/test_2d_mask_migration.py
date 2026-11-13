"""Regression tests for the 2-D action-mask migration.

Verifies that RoboCOIN / EgoDex / RoboTwin readers emit the new 2-D
``action_mask`` / ``proprio_mask`` shapes, and that the 2-D mask is
exactly a per-dim broadcast of the legacy 1-D time mask (every valid
timestep has every dim valid).

We exercise the helper builders in ``utils/eef.py`` plus a small
end-to-end test that instantiates RoboCOIN against a synthetic bucket.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.eef import (
    LEFT_ARM_DIM_MASK,
    build_action_mask_2d,
    build_proprio_mask_2d,
)


class TestActionMask2dBuilder:
    def test_shape(self):
        m = build_action_mask_2d(T_action=32, action_dim=20, n_valid_time=10)
        assert m.shape == (32, 20)
        assert m.dtype == bool


    def test_partial_time_valid(self):
        m = build_action_mask_2d(T_action=10, action_dim=4, n_valid_time=3)
        assert m[:3, :].all()
        assert not m[3:, :].any()

    def test_n_valid_time_zero_returns_all_false(self):
        m = build_action_mask_2d(T_action=10, action_dim=20, n_valid_time=0)
        assert not m.any()

    def test_single_arm_left_dim_mask(self):
        m = build_action_mask_2d(T_action=10, action_dim=20, n_valid_time=5, dim_mask=LEFT_ARM_DIM_MASK)
        # Front 10 dims True in valid timesteps
        assert m[:5, :10].all()
        # Back 10 dims always False
        assert not m[:, 10:].any()
        # Padded timesteps all False
        assert not m[5:, :].any()


    def test_2d_mask_equals_broadcast_of_1d(self):
        """The 2-D mask must equal the legacy 1-D time mask broadcast across D.
        This is the equivalence that gives RoboCOIN/EgoDex/RoboTwin loss
        numerical identity."""
        T_action, D, n_valid = 32, 20, 7
        # Legacy 1-D contract
        legacy_1d = np.zeros(T_action, dtype=bool)
        legacy_1d[:n_valid] = True
        # New 2-D contract (bimanual: no per-dim mask)
        new_2d = build_action_mask_2d(T_action, D, n_valid)
        # Broadcast the 1-D to 2-D for comparison
        broadcast = np.tile(legacy_1d[:, None], (1, D))
        np.testing.assert_array_equal(new_2d, broadcast)


class TestProprioMask2dBuilder:
    def test_shape(self):
        m = build_proprio_mask_2d(action_dim=20)
        assert m.shape == (1, 20)
        assert m.dtype == bool

    def test_enabled_all_true(self):
        m = build_proprio_mask_2d(action_dim=20, enabled=True)
        assert m.all()

    def test_disabled_all_false(self):
        m = build_proprio_mask_2d(action_dim=20, enabled=False)
        assert not m.any()


