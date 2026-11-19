"""Pre-launch config validation: merge a launch config the way training would
and show / save the result, WITHOUT loading models or starting torchrun.

Usage (every form composes the SAME merged config the matching launch would):

    # experiment yaml appended onto the DEFAULT train.yaml — this is the
    # +experiment=qwen_oft launch, validated before torchrun
    python scripts/tools/check_config.py configs/experiment/qwen_oft.yaml

    # same, plus the overrides you plan to pass at launch
    python scripts/tools/check_config.py configs/experiment/qwen_oft.yaml training.batch_size=4 --out merged.yaml

    # the base config alone (no experiment appended)
    python scripts/tools/check_config.py configs/train.yaml

    # equivalent to the path form, written as raw overrides
    python scripts/tools/check_config.py +experiment=qwen_pi dataloader.num_frames=9

Equivalences:
    python scripts/tools/check_config.py configs/experiment/qwen_oft.yaml [overrides]
        == bash scripts/train.sh +experiment=qwen_oft [overrides]

Resolution rules (mirrors ``scripts/train.py``):
- a path under ``configs/experiment/`` composes as
  ``train + experiment=<stem>`` (experiment yamls are sparse diffs and cannot
  stand alone — they need the train.yaml base);
- ``configs/train.yaml`` composes as itself;
- any other path is rejected (the +experiment workflow owns launch configs);
- args starting with ``+``/``-`` or containing ``=`` are forwarded as hydra
  overrides — preview exactly the overrides a real launch would receive.

Exit code 0 = composed AND the architecture block resolves
(``resolve_architecture_config`` — pure config logic, no model loading);
exit 1 with the hydra error on stderr otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (script lives in scripts/tools/)
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiment"


def _looks_like_override(arg: str) -> bool:
    return arg.startswith(("+", "-")) or "=" in arg


def build_overrides(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Return (config_name, hydra_overrides) from the CLI arguments."""
    overrides = list(args.overrides)
    target = args.target

    if _looks_like_override(target):
        return "train", [target, *overrides]

    path = Path(target).resolve()
    if path == CONFIGS_DIR / "train.yaml":
        return "train", overrides
    if path.parent == EXPERIMENTS_DIR and path.suffix == ".yaml":
        return "train", [f"+experiment={path.stem}", *overrides]

    sys.exit(
        f"error: {path} is not a launch config. Point at a yaml under "
        f"configs/experiment/ (composed as `train +experiment=<name>`) or "
        f"configs/train.yaml itself, or pass overrides directly, e.g. "
        f"`+experiment=my_exp training.batch_size=4`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="yaml path (configs/experiment/*.yaml or configs/train.yaml) or a hydra override")
    parser.add_argument("overrides", nargs="*", help="extra hydra overrides forwarded to compose")
    parser.add_argument("--out", default=None, help="also write the merged config to this yaml file")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir

    config_name, overrides = build_overrides(args)
    try:
        with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides)

        # Semantic check on the architecture block: framework/variant/head
        # pairing resolves to a registry entry (pure config logic, no I/O).
        from openwam.model import resolve_architecture_config

        if OmegaConf.select(cfg, "model") is not None:
            resolved = resolve_architecture_config(cfg.model)
            print(f"[check_config] architecture -> {resolved.registry_name}", file=sys.stderr)
    except Exception as exc:  # surface hydra/OmegaConf errors verbatim, fail loudly
        print(f"[check_config] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    merged = OmegaConf.to_yaml(cfg)
    print(merged)
    if args.out:
        Path(args.out).write_text(merged, encoding="utf-8")
        print(f"[check_config] merged config written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
