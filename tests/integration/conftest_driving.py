"""Shared synthetic Track B storage fixture (mp4 + parquet topics + calib).

Used by the driving preprocessor-chain and ingest-lance integration tests.
Builds ONE deterministic clip under ``raw_clips/clipA`` in a tmp dir.
"""

import json
from pathlib import Path

import numpy as np
import pytest


def build_synthetic_clip(root: Path) -> dict:
    """Create ``<root>/raw_clips/clipA`` with 100-frame 5s@20fps video, 1 kHz
    ego_motion (speed / yaw_rate / yaw_quat), 100 Hz localization (straight
    line), and a navigation topic. Returns topic timestamp arrays (ns)."""
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    clip = root / "raw_clips" / "clipA"
    (clip / "cams").mkdir(parents=True, exist_ok=True)
    (clip / "topics").mkdir(parents=True, exist_ok=True)

    n, h, w = 100, 16, 24
    container = av.open(str(clip / "cams" / "cam_front.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=20)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = i * 2
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

    (clip / "calib.json").write_text(json.dumps({"cameras": {"cam_front": {
        "intrinsics": [[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]],
        "extrinsics": np.eye(4).tolist(),
    }}}), encoding="utf-8")

    return {"root": root, "clip": clip, "ts": ts, "t0": t0}


@pytest.fixture(scope="session")
def driving_storage(tmp_path_factory):
    return build_synthetic_clip(tmp_path_factory.mktemp("driving_storage"))
