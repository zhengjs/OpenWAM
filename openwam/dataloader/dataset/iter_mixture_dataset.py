"""Iter-based mixture: plan-driven sampling over heterogeneous sources (spec_data_zh.md §6.2/§6.3).

Fixed-total-step training (LLM-style token budget) — most runs never finish an
epoch, so sampling is driven by a precomputed GLOBAL PLAN instead of
``epoch × len(dataset)``:

- ``IterMixtureDataset`` is a thin dispatcher over map-style sources: given a
  plan position it resolves ``(dataset_index, sample_index)``. It holds no
  weights, no index map, no ``set_epoch``.
- ``WeightAnnealingSampler`` owns the plan (normative, spec §6.2):

  * ``dataset_index`` — one entry PER SAMPLE (a batch may mix sources), length
    ``batch_size × total_steps`` per rank, drawn from init→final annealed
    weights with a per-sample seekable RNG (``SeedSequence([seed, rank, step])``),
    so resume never replays step 0..K-1 and no RNG state is checkpointed.
  * ``sample_index`` — ONE sequence per source, shared across ranks: the full
    permutation is generated from a shared seed and each rank keeps only its
    strided slice ``perm[rank::world_size]`` (the rank-local memory cut from
    spec §6.2); ranks jointly cover the permutation in order, no overlap.
  * Offline mode loads the same plan arrays from Lance instead of generating
    them (spec §6.2 offline; landing with the lance stack).

Resume (spec §6.3): the sampler checkpoints ``consumed`` plan positions plus
per-source cursors via torch-native ``state_dict`` / ``load_state_dict``; after
reload the remaining plan stream is byte-identical to an uninterrupted run.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.dataset import BaseDataset

logger = logging.getLogger(__name__)

__all__ = ["IterMixtureDataset", "WeightAnnealingSampler"]


def _anneal_alpha(k: int, total: int, schedule: str) -> float:
    """Anneal progress in [0, 1] (linear default or cosine); ``total`` >= 1."""
    denom = max(total - 1, 1)
    k = min(max(int(k), 0), total - 1)
    if schedule == "cosine":
        return (1.0 - math.cos(k * math.pi / denom)) / 2.0
    return k / denom


class IterMixtureDataset(BaseDataset):
    """Dispatcher over map-style sources, resolved by an attached plan.

    Contract (spec §6.2): ``attach_plan`` receives the sampler-generated
    ``(dataset_index, sample_index)`` arrays; ``__getitem__(plan_pos)`` then
    returns ``sources[plan_di[plan_pos]][plan_si[plan_pos]]`` tagged with
    ``_dataset_index`` / ``_dataset_name`` (same keys as ``MixtureDataset``).
    Before a plan is attached, ``__getitem__`` is a hard error — there is no
    fallback ordering by design.
    """

    def __init__(
        self,
        datasets: Sequence[BaseDataset],
        names: Optional[Sequence[str]] = None,
        strict_action_dim: bool = False,
        action_dim_override: Optional[int] = None,
    ):
        if not datasets:
            raise ValueError("IterMixtureDataset requires at least one sub-dataset")
        self._datasets = list(datasets)
        self._names = (
            [f"source_{i}" for i in range(len(self._datasets))]
            if names is None
            else [str(n) for n in names]
        )
        if len(self._names) != len(self._datasets):
            raise ValueError(f"names length ({len(self._names)}) != datasets length ({len(self._datasets)})")

        dims = [d.action_dim for d in self._datasets]
        if strict_action_dim:
            if len(set(dims)) != 1:
                raise ValueError(
                    "IterMixtureDataset: all enabled sub-datasets must share the same action_dim, "
                    f"got {dict(zip(self._names, dims))}"
                )
            self._action_dim = dims[0]
        else:
            if action_dim_override is not None:
                self._action_dim = action_dim_override
            elif len(set(dims)) == 1:
                self._action_dim = dims[0]
            else:
                self._action_dim = max(dims)
                logger.warning(
                    "IterMixtureDataset: sub-datasets have different action dims %s; padding to max=%d",
                    dims,
                    self._action_dim,
                )
        self._strict_action_dim = strict_action_dim

        self._plan_di: Optional[np.ndarray] = None
        self._plan_si: Optional[np.ndarray] = None

    # ----- plan wiring -------------------------------------------------------

    def attach_plan(self, plan_di: np.ndarray, plan_si: np.ndarray) -> None:
        """Install the sampler-generated plan (owner: WeightAnnealingSampler)."""
        if plan_di.shape != plan_si.shape or plan_di.ndim != 1:
            raise ValueError(f"plan arrays must be 1-D of equal length, got {plan_di.shape} / {plan_si.shape}")
        if len(plan_di) and int(plan_di.max()) >= len(self._datasets):
            raise ValueError("plan dataset_index out of range")
        self._plan_di = plan_di
        self._plan_si = plan_si

    @property
    def plan_attached(self) -> bool:
        return self._plan_di is not None

    # ----- dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        if self._plan_di is not None:
            return int(self._plan_di.shape[0])
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, idx: int) -> dict:
        if self._plan_di is None:
            raise RuntimeError("IterMixtureDataset: no plan attached; build a WeightAnnealingSampler first")
        di = int(self._plan_di[idx])
        si = int(self._plan_si[idx])
        sample = self._datasets[di][si]

        if isinstance(sample, PipelineContext):
            # Track B path: the context already carries source_name /
            # sample_weight from its Lance row; collate consumes it directly.
            return sample

        if not self._strict_action_dim:
            action_traj = sample.get("action")
            if (
                action_traj is not None
                and isinstance(action_traj, torch.Tensor)
                and action_traj.shape[-1] < self._action_dim
            ):
                pad_size = self._action_dim - action_traj.shape[-1]
                sample["action"] = torch.nn.functional.pad(action_traj, (0, pad_size))

        sample["_dataset_index"] = di
        sample["_dataset_name"] = self._names[di]
        return sample

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def normalization_stats(self) -> Optional[dict]:
        return None

    @property
    def datasets(self) -> List[BaseDataset]:
        return self._datasets

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def get_dataset(self, name: str) -> BaseDataset:
        try:
            idx = self._names.index(name)
        except ValueError:
            raise KeyError(f"IterMixtureDataset: no sub-dataset named '{name}'. Known: {self._names}") from None
        return self._datasets[idx]


class WeightAnnealingSampler(Sampler[int]):
    """Plan owner: annealed dataset_index + per-source shared sample sequences.

    See the module docstring for the normative plan semantics (spec §6.2/§6.3).

    Args:
        dataset:        The ``IterMixtureDataset`` to attach the plan to.
        weight_start:   Raw relative draw weight per source at step 0.
        weight_end:     Raw relative draw weight per source at the final step.
        total_steps:    Optimizer-step budget (the horizon ``N`` of the schedule).
        batch_size:     Per-rank batch size; the plan length is ``batch_size × total_steps``.
        rank:           This rank's index (seed binding + stride offset).
        world_size:     Stride for the shared per-source sample sequences.
        seed:           Base seed; the per-step dataset draw binds ``[seed, rank, step]``.
        sample_seed:    Seed for the shared per-source permutations (default ``seed``).
        schedule:       ``linear`` (default) or ``cosine`` annealing.
    """

    def __init__(
        self,
        dataset: IterMixtureDataset,
        *,
        weight_start: Sequence[float],
        weight_end: Sequence[float],
        total_steps: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        sample_seed: Optional[int] = None,
        schedule: str = "linear",
    ):
        n_src = len(dataset.datasets)
        if n_src == 0:
            raise ValueError("WeightAnnealingSampler: dataset has no sources")
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"WeightAnnealingSampler: unknown schedule={schedule!r}; use 'linear' or 'cosine'")
        if int(total_steps) < 1 or int(batch_size) < 1:
            raise ValueError("WeightAnnealingSampler: total_steps and batch_size must be >= 1")
        if not 0 <= int(rank) < int(world_size) or int(world_size) < 1:
            raise ValueError(f"WeightAnnealingSampler: invalid rank/world_size: {rank}/{world_size}")

        start = np.asarray(weight_start, dtype=np.float64)
        end = np.asarray(weight_end, dtype=np.float64)
        if start.shape != (n_src,) or end.shape != (n_src,):
            raise ValueError(f"weight_start/weight_end length must equal the source count ({n_src})")
        for label, arr in (("weight_start", start), ("weight_end", end)):
            if not all(np.isfinite(arr)) or any(arr < 0):
                raise ValueError(f"WeightAnnealingSampler: {label} must be finite and non-negative: {arr.tolist()}")
        if start.sum() <= 0 or end.sum() <= 0:
            raise ValueError("WeightAnnealingSampler: weights must be positive at both endpoints")

        self._dataset = dataset
        self._n_src = n_src
        self._start = start
        self._end = end
        self._schedule = schedule
        self._total_steps = int(total_steps)
        self._batch_size = int(batch_size)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._seed = int(seed)
        self._sample_seed = int(seed) if sample_seed is None else int(sample_seed)

        self._num_samples = self._batch_size * self._total_steps
        self._consumed = 0

        # Rank-local strided slice of each source's shared permutation
        # (spec §6.2: full permutation is rank-shared; each rank keeps only
        # its ``perm[rank::world_size]`` slice). cycled if a source's local
        # slice is exhausted before the run ends (same reuse semantics as
        # MixtureDataset's modulo cycling).
        self._local_slices: List[np.ndarray] = []
        for d, src in enumerate(self._dataset.datasets):
            n = len(src)
            if n < 1:
                raise ValueError(f"WeightAnnealingSampler: source '{self._dataset.names[d]}' is empty")
            seq = np.random.default_rng(np.random.SeedSequence([self._sample_seed, d])).permutation(n)
            self._local_slices.append(seq[self._rank :: self._world_size])

        plan_di, plan_si = self._generate_plan()
        self._cursors = [0] * n_src
        self._dataset.attach_plan(plan_di, plan_si)

    # ----- plan generation -----------------------------------------------------

    def _weights_at(self, step: int) -> np.ndarray:
        alpha = _anneal_alpha(step, self._total_steps, self._schedule)
        raw = self._start + (self._end - self._start) * alpha
        return raw / raw.sum()

    def _generate_plan(self):
        """Materialize ``(dataset_index, sample_index)`` for all plan positions.

        Per step: one seekable generator from ``SeedSequence([seed, rank, step])``
        draws ``batch_size`` uniforms resolved against the annealed cumulative
        weights (O(1) per step on resume — no replay of earlier steps, no RNG
        state to checkpoint; spec §6.3).
        """
        plan_di = np.zeros(self._num_samples, dtype=np.int16)
        plan_si = np.zeros(self._num_samples, dtype=np.int64)
        cursors = [0] * self._n_src

        for step in range(self._total_steps):
            j0 = step * self._batch_size
            cum = np.cumsum(self._weights_at(step))
            cum[-1] = 1.0  # guard float drift so the last source is reachable
            u = np.random.default_rng(
                np.random.SeedSequence([self._seed, self._rank, step])
            ).random(self._batch_size)
            dis = np.searchsorted(cum, u, side="right").clip(0, self._n_src - 1)
            plan_di[j0 : j0 + self._batch_size] = dis
            for d in np.unique(dis):
                cnt = int((dis == d).sum())
                sl = self._local_slices[d]
                take = sl[cursors[d] % len(sl) : (cursors[d] % len(sl)) + cnt]
                if len(take) < cnt:  # local slice exhausted mid-draw: cycle
                    first = len(sl) - cursors[d] % len(sl)
                    take = np.concatenate([sl[-first:], sl[: cnt - first]])
                plan_si[j0 : j0 + self._batch_size][dis == d] = take
                cursors[d] += cnt

        if len(np.unique(plan_di)) < self._n_src:
            never = [self._dataset.names[d] for d in range(self._n_src) if cursors[d] == 0]
            logger.warning("WeightAnnealingSampler: sources never drawn over the run: %s", never)
        return plan_di, plan_si

    # ----- Sampler protocol ------------------------------------------------------

    def __iter__(self):
        return iter(range(self._consumed, self._num_samples))

    def __len__(self) -> int:
        return self._num_samples - self._consumed

    # ----- resume (spec §6.3, torch-native naming) -------------------------------

    def state_dict(self) -> dict:
        """Checkpointable sampling state: master cursor + per-source cursors."""
        return {
            "consumed": int(self._consumed),
            "cursors": [int(c) for c in self._cursors],
            "total_steps": self._total_steps,
            "batch_size": self._batch_size,
            "rank": self._rank,
            "world_size": self._world_size,
            "seed": self._seed,
            "sample_seed": self._sample_seed,
            "schedule": self._schedule,
            "weight_start": self._start.tolist(),
            "weight_end": self._end.tolist(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Seek to the checkpointed plan position; the remaining stream is
        byte-identical to an uninterrupted run (plan is immutable)."""
        consumed = int(state_dict["consumed"])
        if not 0 <= consumed <= self._num_samples:
            raise ValueError(
                f"WeightAnnealingSampler: consumed={consumed} outside plan range [0, {self._num_samples}]"
            )
        for key in ("total_steps", "batch_size", "rank", "world_size"):
            if int(state_dict[key]) != getattr(self, f"_{key}"):
                raise ValueError(
                    f"WeightAnnealingSampler: {key} mismatch (ckpt={state_dict[key]}, "
                    f"current={getattr(self, f'_{key}')})"
                )
        if state_dict["schedule"] != self._schedule:
            raise ValueError(
                f"WeightAnnealingSampler: schedule mismatch (ckpt={state_dict['schedule']!r}, "
                f"current={self._schedule!r})"
            )
        cursors = [int(c) for c in state_dict["cursors"]]
        if len(cursors) != self._n_src:
            raise ValueError(f"WeightAnnealingSampler: cursors length {len(cursors)} != sources {self._n_src}")
        self._consumed = consumed
        self._cursors = cursors
