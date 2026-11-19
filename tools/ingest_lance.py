"""Offline Track B ingestion: raw_clips/ -> <name>.lance + dataset_statistics.json
(spec_data_zh.md §4 / §2).

Per clip and per anchor timestamp this tool:

1. matches each camera's nearest MP4 frame to the anchor (``cam_frame_offsets``;
   anchors outside ``--frame-tol-us`` on ANY camera are skipped);
2. projects ``topics/localization.parquet`` world poses into the ego rig at the
   anchor → ``gt_future_traj_ego`` ``[T_pred, D_action]`` (x, y, yaw) plus
   ``valid_mask`` ``[T_pred]``;
3. folds ``topics/navigation.parquet`` into ``nav_instruction`` / ``cot_text``;
4. accumulates per-column statistics (on the topic_sync-compatible state layout,
   i.e. AFTER optional quat→rot6d conversion) → ``dataset_statistics.json``;
5. writes the fixed Lance schema (§4.3).

``lance`` is only required for the final write — ``--dry-run`` runs steps 1-4
and prints the stats, so the geometry/sync logic is verifiable on hosts
without lance. Offline sampling-sequence emission (§6.2 offline mode) lands
with the lance stack as a separate tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openwam.dataloader.utils.clip_reader import ClipHandle  # noqa: E402
from openwam.dataloader.utils.poses import quat_wxyz_to_rot6d  # noqa: E402

STAT_KEYS = ("min", "max", "q01", "q99", "mean", "std")


# ---------------------------------------------------------------- anchors ----


def match_cam_frames(pts_ns: np.ndarray, anchor_ns: np.ndarray, tol_us: int) -> np.ndarray:
    """Nearest frame index per anchor; raises IndexError beyond clip end."""
    idx = np.searchsorted(pts_ns, anchor_ns)
    idx = np.clip(idx, 1, len(pts_ns) - 1)
    left, right = pts_ns[idx - 1], pts_ns[idx]
    choose_left = (anchor_ns - left) <= (right - anchor_ns)
    nearest = np.where(choose_left, idx - 1, idx)
    dist_us = np.abs(pts_ns[nearest] - anchor_ns) / 1000
    bad = dist_us > tol_us
    if bad.any():
        raise LookupError(f"{int(bad.sum())} anchors exceed frame tolerance ({tol_us}us)")
    return nearest


# ------------------------------------------------------ gt trajectory (ego) ----


def gt_future_traj_ego(loc_ts: np.ndarray, xy: np.ndarray, yaw: np.ndarray,
                       anchor_i: int, anchor_ns: int, dt_ns: int, t_pred: int) -> tuple:
    """World poses → ego-rig future trajectory ``[T_pred, 3]`` (dx, dy, dyaw).

    Returns ``(traj, valid_mask)``; steps beyond the localization horizon are
    marked invalid (mask 0) with the last valid value held.
    """
    horizon = anchor_ns + dt_ns * t_pred
    future_ts = anchor_ns + dt_ns * np.arange(1, t_pred + 1, dtype=np.int64)
    del horizon
    j = np.clip(np.searchsorted(loc_ts, future_ts), 0, len(loc_ts) - 1)
    valid = (future_ts <= loc_ts[-1]).astype(np.int8)

    x0, y0, th0 = xy[anchor_i, 0], xy[anchor_i, 1], yaw[anchor_i]
    dx = xy[j, 0] - x0
    dy = xy[j, 1] - y0
    # Rotate world delta into the anchor's ego frame: R(-yaw0) @ [dx, dy].
    c, s = np.cos(th0), np.sin(th0)
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    traj = np.stack([ex, ey, yaw[j] - th0], axis=1).astype(np.float32)
    traj = traj * valid[:, None]  # hold-last semantics: invalid steps zeroed
    return traj, valid


def nearest_instruction(nav_ts: np.ndarray, texts: list, cots: list, anchor_ns: int):
    """Instruction in force at the anchor = last row at/before it ("" if none)."""
    k = int(np.searchsorted(nav_ts, anchor_ns, side="right")) - 1
    if k < 0:
        return "", None
    return texts[k], (cots[k] if cots[k] else None)


# ------------------------------------------------------------------ stats ----


def merge_stats(acc: dict, arr: np.ndarray) -> None:
    for k, v in _vec_stats(arr).items():
        acc.setdefault(k, []).append(v)


def _vec_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
    }


def finalize_stats(acc: dict, dim: int) -> dict:
    out = {}
    for k in STAT_KEYS:
        stacked = np.stack(acc[k], axis=0)  # [n_anchors, dim]
        if k in ("min", "q01"):
            out[k] = stacked.min(axis=0).tolist()
        elif k in ("max", "q99"):
            out[k] = stacked.max(axis=0).tolist()
        else:  # mean/std over anchors weighted equally (per-clip aggregates)
            out[k] = stacked.mean(axis=0).tolist()
    assert len(out["min"]) == dim
    return out


# ------------------------------------------------------------------ main ----


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--storage-root", required=True, help="dir containing raw_clips/")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--cams", required=True, help="comma-separated camera names (first = ref)")
    p.add_argument("--t-obs", type=int, default=6)
    p.add_argument("--t-pred", type=int, default=40)
    p.add_argument("--t-hist", type=int, default=1)
    p.add_argument("--dt-pred-us", type=int, default=100_000, help="future trajectory stride")
    p.add_argument("--anchor-stride-us", type=int, default=500_000)
    p.add_argument("--frame-tol-us", type=int, default=10_000)
    p.add_argument("--rot6d-source", default=None, help="ego_motion quat column → rot6d")
    p.add_argument("--scene-tags", default="", help="comma-separated tags stamped on every row")
    p.add_argument("--stats-mode", default="q99", choices=["q99", "min_max", "mean_std"])
    p.add_argument("--dry-run", action="store_true", help="compute + print stats, skip lance write")
    args = p.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(args.storage_root)
    cams = [c for c in args.cams.split(",") if c]
    tags = [t for t in args.scene_tags.split(",") if t]
    stride_ns = args.anchor_stride_us * 1000
    dt_ns = args.dt_pred_us * 1000

    rows = []
    ego_acc: dict = {}
    traj_acc: dict = {}
    for clip_dir in sorted((root / "raw_clips").iterdir()):
        if not (clip_dir / "cams").is_dir():
            continue
        handle = ClipHandle(str(clip_dir))
        ref_pts = handle.frame_timestamps_ns(cams[0])
        t0, t1 = int(ref_pts[0]), int(ref_pts[-1])
        anchors = np.arange(t0, t1 + 1, stride_ns, dtype=np.int64)

        offsets_by_cam = {}
        for cam in cams:
            pts = ref_pts if cam == cams[0] else handle.frame_timestamps_ns(cam)
            try:
                offsets_by_cam[cam] = match_cam_frames(pts, anchors, args.frame_tol_us)
            except LookupError as e:
                print(f"[skip] {clip_dir.name}/{cam}: {e}")
                anchors = None
                break
        if anchors is None:
            continue

        table = handle.ego_motion
        ts = np.asarray(table.column("timestamp_ns").to_pylist(), dtype=np.int64)
        order = np.argsort(ts)
        ts = ts[order]
        state_cols = [c for c in table.column_names if c != "timestamp_ns"]
        col_blocks, widths = [], []
        for c in state_cols:
            arr = np.asarray(table.column(c).to_pylist(), dtype=np.float32)[order]
            if arr.ndim == 1:
                arr = arr[:, None]
            widths.append(arr.shape[1])
            col_blocks.append(arr)
        state = np.concatenate(col_blocks, axis=1)
        if args.rot6d_source:
            k = state_cols.index(args.rot6d_source)
            start = sum(widths[:k])
            state = np.concatenate(
                [state[:, :start], quat_wxyz_to_rot6d(state[:, start : start + widths[k]]),
                 state[:, start + widths[k] :]], axis=1)
        d_state = state.shape[1]

        nav_table = pq.read_table(clip_dir / "topics" / "navigation.parquet") if (
            clip_dir / "topics" / "navigation.parquet").is_file() else None
        loc_table = pq.read_table(clip_dir / "topics" / "localization.parquet") if (
            clip_dir / "topics" / "localization.parquet").is_file() else None

        for ai, anchor_ns in enumerate(anchors):
            ego_at = _interp_rows(ts, state, anchor_ns - (args.t_hist - 1) * stride_ns)
            merge_stats(ego_acc, ego_at.reshape(1, -1))

            nav_instr, cot = "", None
            if nav_table is not None:
                nav_ts = np.asarray(nav_table.column("timestamp_ns").to_pylist(), dtype=np.int64)
                texts = nav_table.column("instruction").to_pylist()
                cots = (nav_table.column("cot_text").to_pylist()
                        if "cot_text" in nav_table.column_names else [""] * len(nav_ts))
                nav_instr, cot = nearest_instruction(nav_ts, texts, cots, int(anchor_ns))

            traj = np.zeros((args.t_pred, 3), dtype=np.float32)
            valid = np.zeros(args.t_pred, dtype=np.int8)
            if loc_table is not None:
                loc_ts = np.asarray(loc_table.column("timestamp_ns").to_pylist(), dtype=np.int64)
                lorder = np.argsort(loc_ts)
                loc_ts = loc_ts[lorder]
                xy = np.stack([
                    np.asarray(loc_table.column("x").to_pylist(), dtype=np.float64)[lorder],
                    np.asarray(loc_table.column("y").to_pylist(), dtype=np.float64)[lorder],
                ], axis=1)
                yaw = np.asarray(loc_table.column("yaw").to_pylist(), dtype=np.float64)[lorder]
                ref_i = int(np.clip(np.searchsorted(loc_ts, anchor_ns), 0, len(loc_ts) - 1))
                traj, valid = gt_future_traj_ego(loc_ts, xy, yaw, ref_i, int(anchor_ns), dt_ns, args.t_pred)
                merge_stats(traj_acc, traj)

            rows.append({
                "sample_id": f"{clip_dir.name}_{int(anchor_ns)}",
                "clip_uri": str(clip_dir),
                "anchor_timestamp_us": int(anchor_ns) // 1000,
                "cam_frame_offsets": {cam: offsets_by_cam[cam][max(0, ai - args.t_obs + 1): ai + 1].tolist()
                                      for cam in cams},
                "nav_instruction": nav_instr,
                "cot_text": cot,
                "scene_tags": tags,
                "sample_weight": 1.0,
                "gt_future_traj_ego": traj.reshape(-1).tolist(),
                "valid_mask": valid.tolist(),
            })

    if not rows:
        print("no samples ingested")
        return 1

    d_action = 3
    stats = {
        "mode": args.stats_mode,
        "ego_state": finalize_stats(ego_acc, d_state),
        "trajectory": finalize_stats(traj_acc, d_action),
        "T_hist": args.t_hist, "D_state": d_state,
        "T_pred": args.t_pred, "D_action": d_action,
    }
    stats_path = root / "dataset_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[stats] {stats_path}: D_state={d_state} D_action={d_action} "
          f"T_hist={args.t_hist} T_pred={args.t_pred}")

    if args.dry_run:
        print(f"[dry-run] {len(rows)} samples; skipping lance write")
        return 0

    import lance

    schema = pa.schema([
        ("sample_id", pa.string()),
        ("clip_uri", pa.string()),
        ("anchor_timestamp_us", pa.int64()),
        ("cam_frame_offsets", pa.map_(pa.string(), pa.list_(pa.int64()))),
        ("nav_instruction", pa.string()),
        ("cot_text", pa.string()),
        ("scene_tags", pa.list_(pa.string())),
        ("sample_weight", pa.float32()),
        ("gt_future_traj_ego", pa.list_(pa.float32(), args.t_pred * d_action)),
        ("valid_mask", pa.list_(pa.int8(), args.t_pred)),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    out_dir = root / "lance_indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    lance.write_table(table, str(out_dir / f"{args.dataset_name}.lance"))
    print(f"[lance] {out_dir / f'{args.dataset_name}.lance'}: {len(rows)} rows")
    return 0


def _interp_rows(ts: np.ndarray, values: np.ndarray, query_ns: int) -> np.ndarray:
    """Linear interpolation row at ``query_ns`` (clamped to the table range)."""
    i = int(np.clip(np.searchsorted(ts, query_ns, side="right") - 1, 0, len(ts) - 1))
    j = min(i + 1, len(ts) - 1)
    w = 0.0 if ts[j] == ts[i] else (query_ns - ts[i]) / (ts[j] - ts[i])
    return values[i] * (1.0 - w) + values[j] * w


if __name__ == "__main__":
    raise SystemExit(main())
