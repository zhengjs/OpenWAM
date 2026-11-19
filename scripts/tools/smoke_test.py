"""Track B smoke test: chain the FULL training flow on synthetic dummy data.

    python scripts/tools/smoke_test.py                # storage + data check + 2 training steps
    python scripts/tools/smoke_test.py --no-train     # data layer only (no GPU/weights needed)
    python scripts/tools/smoke_test.py --steps 5 --force

Stages:
  1. storage  — generate the synthetic clip + stats under
                outputs/dummy_driving_storage (skipped when present, --force rebuilds);
  2. data     — build the dataset from the dummy_clip config through the real
                registry + preprocessor chain, pull one sample, print the
                PipelineContext shapes the model bridge will consume;
  3. train    — subprocess `bash scripts/train.sh +experiment=wam_driving_smoke`
                (reuses the real entry point verbatim). REQUIRES the host to
                resolve the Wan2.2-TI2V-5B checkpoint (model_path in
                configs/model/video_backbone/wan22_ti2v_5b.yaml) and, in
                practice, a GPU — on a CPU-only box stop at --no-train.

Exit code 0 = all requested stages passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = PROJECT_ROOT / "outputs" / "dummy_driving_storage"


def ensure_storage(root: Path, force: bool) -> None:
    """Stage 1: synthetic clip + stats under ``root`` (idempotent)."""
    from openwam.dataloader.utils.synthetic_clip import build_synthetic_clip, write_dummy_stats

    if (root / "raw_clips").is_dir() and not force:
        print(f"[smoke 1/3] storage exists: {root}")
        return
    if force and root.exists():
        import shutil

        shutil.rmtree(root)
    build_synthetic_clip(root)
    stats = write_dummy_stats(root)
    print(f"[smoke 1/3] storage generated: {root} (stats: {stats.name})")


def check_data_layer(configs_dir: Path) -> None:
    """Stage 2: registry -> dataset -> preprocessor chain -> one sample."""
    from hydra import compose, initialize_config_dir

    sys.path.insert(0, str(PROJECT_ROOT))
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["+experiment=wam_driving_smoke"])

    from openwam.dataloader.dataset.registry import build_dataset

    dataset = build_dataset(cfg.dataloader, split="train")
    print(f"[smoke 2/3] dataset: {type(dataset).__name__}, len={len(dataset)}, action_dim={dataset.action_dim}")

    ctx = dataset[0]
    fields = {
        "pixel_values": getattr(ctx, "pixel_values", None),
        "ego_state_tensor": ctx.ego_state_tensor,
        "gt_trajectory_norm": ctx.gt_trajectory_norm,
        "trajectory_mask": ctx.trajectory_mask,
        "frame_timestamps_ns": ctx.frame_timestamps_ns,
        "prompt": ctx.prompt,
    }
    print("[smoke 2/3] sample slots:")
    for name, value in fields.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else value
        print(f"    {name}: {shape}")

    from openwam.contracts.collate import collate_pipeline_contexts

    batch = collate_pipeline_contexts([dataset[0], dataset[1]])
    print(f"[smoke 2/3] collated: pixel_values={tuple(batch.pixel_values.shape)}, "
          f"actions={tuple(batch.gt_trajectory_norm.shape)}, prompt={batch.prompt}")


def run_training(steps: int) -> int:
    """Stage 3: the real training entry on the smoke experiment (subprocess)."""
    cmd = ["bash", "scripts/train.sh", "+experiment=wam_driving_smoke",
           f"training.max_steps={steps}", "training.debug=true"]
    print(f"[smoke 3/3] launching: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--steps", type=int, default=2, help="training.max_steps for stage 3")
    parser.add_argument("--no-train", action="store_true", help="stop after the data-layer check (CPU-only hosts)")
    parser.add_argument("--force", action="store_true", help="regenerate the synthetic storage")
    args = parser.parse_args()

    ensure_storage(args.storage_root, args.force)
    check_data_layer(PROJECT_ROOT / "configs")

    if args.no_train:
        print("[smoke] --no-train: data layer OK, training skipped")
        return
    sys.exit(run_training(args.steps))


if __name__ == "__main__":
    main()
