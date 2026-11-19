"""Synthetic Track B storage builder: the dummy-data source of truth.

``build_synthetic_clip`` writes the exact on-disk layout ``ClipReader``
consumes (``cams/<cam>.mp4`` + ``topics/*.parquet`` + ``calib.json``) as ONE
deterministic 100-frame clip; ``write_dummy_stats`` writes a matching small
``dataset_statistics.json``. Originally a test fixture
(``tests/integration/conftest_driving.py``), promoted here so production-side
dummy tooling (``dataset/dummy_dataset.py``, ``scripts/tools/smoke_test.py``)
and tests share one builder.
"""

import json
from pathlib import Path

import numpy as np

__all__ = ["build_synthetic_clip", "write_dummy_stats"]


def build_synthetic_clip(root: Path) -> dict:
    """Create ``<root>/raw_clips/clipA`` with three cameras (front/left/rear,
    100-frame 5s@20fps, deterministic per-cam color ramps), 1 kHz ego_motion
    (speed / yaw_rate / yaw_quat), 100 Hz localization (straight line), and a
    navigation topic — the production camera layout, at test-scale resolution.
    Returns topic timestamp arrays (ns)."""
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    cams = ("cam_front", "cam_left", "cam_rear")
    clip = root / "raw_clips" / "clipA"
    (clip / "cams").mkdir(parents=True, exist_ok=True)
    (clip / "topics").mkdir(parents=True, exist_ok=True)

    n, h, w = 100, 16, 24
    for cam_idx, cam in enumerate(cams):
        container = av.open(str(clip / "cams" / f"{cam}.mp4"), mode="w")
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, cam_idx] = i * 2  # per-cam color ramp: front=G, left=R, rear=B
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    # Align the topic timeline to the ACTUAL video PTS timeline (encoders may
    # rescale the timebase).
    probe = av.open(str(clip / "cams" / "cam_front.mp4"))
    vstream = probe.streams.video[0]
    pts_ns = (np.asarray([f.pts for f in probe.decode(vstream)]) * float(vstream.time_base) * 1e9).astype(np.int64)
    probe.close()
    t0 = int(pts_ns[0])
    duration_ns = int(pts_ns[-1] - pts_ns[0])

    ts = t0 + np.arange(0, duration_ns + 1_000_000, 1_000_000, dtype=np.int64)
    pq.write_table(pa.table({
        "timestamp_ns": ts.tolist(),
        "speed": np.linspace(0, 10, len(ts)).tolist(),
        "yaw_rate": np.full(len(ts), 0.05).tolist(),
        "yaw_quat": [[1.0, 0.0, 0.0, 0.0]] * len(ts),
    }), str(clip / "topics" / "ego_motion.parquet"))

    loc_ts = t0 + np.arange(0, duration_ns + 10_000_000, 10_000_000, dtype=np.int64)
    pq.write_table(pa.table({
        "timestamp_ns": loc_ts.tolist(),
        "x": (np.arange(len(loc_ts)) * 0.2).tolist(),
        "y": np.zeros(len(loc_ts)).tolist(),
        "yaw": np.zeros(len(loc_ts)).tolist(),
    }), str(clip / "topics" / "localization.parquet"))

    pq.write_table(pa.table({
        "timestamp_ns": [t0],
        "instruction": ["go straight"],
        "cot_text": [""],
    }), str(clip / "topics" / "navigation.parquet"))

    (clip / "calib.json").write_text(json.dumps({"cameras": {cam: {
        "intrinsics": [[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]],
        "extrinsics": np.eye(4).tolist(),
    } for cam in cams}}), encoding="utf-8")

    return {"root": root, "clip": clip, "ts": ts, "t0": t0}


def write_dummy_stats(
    storage_root: Path,
    t_hist: int = 3,
    d_state: int = 8,
    t_pred: int = 3,
    d_action: int = 3,
) -> Path:
    """Write a small ``dataset_statistics.json`` (min_max mode) matching the
    synthetic clip's state layout (speed + yaw_rate + quat->rot6d = 8-D)."""
    path = Path(storage_root) / "dataset_statistics.json"
    if path.exists():
        return path

    def block(d, lo_hi):
        return {"min": [lo_hi[0]] * d, "max": [lo_hi[1]] * d,
                "q01": [lo_hi[0]] * d, "q99": [lo_hi[1]] * d,
                "mean": [0.0] * d, "std": [1.0] * d}

    ego = block(d_state, (-1.0, 1.0))
    traj = block(d_action, (-1.0, 1.0))
    path.write_text(json.dumps({
        "mode": "min_max",
        "ego_state": ego,
        "trajectory": traj,
        "T_hist": t_hist, "D_state": d_state, "T_pred": t_pred, "D_action": d_action,
    }, indent=2), encoding="utf-8")
    return path
