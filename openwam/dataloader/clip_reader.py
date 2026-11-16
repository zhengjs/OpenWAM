"""Clip reader for Track B raw storage (spec_data_zh.md §4.4).

``ClipHandle`` owns ONE clip directory's open resources: ``cams/*.mp4``
(decoder handles, lazy per camera), ``topics/ego_motion.parquet``, and
``calib.json``. ``ClipReader.instance()`` is a per-PROCESS singleton with a
bounded LRU of handles — DataLoader fork workers each get their own copy of
the LRU (file handles are never shared across processes).

decord is NOT a dependency here: decode uses PyAV (``av``), which is already
required by the robot readers. Random access walks from the nearest preceding
keyframe; the common access pattern (``cam_frame_offsets`` are consecutive
indices) decodes each camera at most once per sample.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ClipHandle", "ClipReader"]


class ClipHandle:
    """Open resources for one ``raw_clips/<clip_id>/`` directory (lazy per cam)."""

    def __init__(self, clip_uri: str):
        self.clip_uri = Path(clip_uri)
        if not self.clip_uri.is_dir():
            raise FileNotFoundError(f"ClipHandle: clip directory not found: {self.clip_uri}")
        self._containers: Dict[str, "av.container.InputContainer"] = {}
        self._streams: Dict[str, "av.video.stream.VideoStream"] = {}
        self._ego_motion = None  # lazy parquet
        self._calib = None  # lazy json

    # ----- video ---------------------------------------------------------------

    def _open_cam(self, cam: str):
        import av  # local import: module import stays cheap for non-decode workers

        if cam not in self._containers:
            path = self.clip_uri / "cams" / f"{cam}.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"ClipHandle: camera stream not found: {path}")
            container = av.open(str(path))
            self._containers[cam] = container
            self._streams[cam] = container.streams.video[0]
        return self._containers[cam], self._streams[cam]

    def read_frames(self, cam: str, indices: Sequence[int]) -> List[np.ndarray]:
        """Decode the given frame indices (HWC uint8, video order = index order).

        Indices need not be consecutive, but MUST be ascending (the contract
        from ingest: ``cam_frame_offsets`` are consecutive ascending). Decode
        restarts whenever a lower index is requested (short driving clips make
        the rewind cheap; avoid caching whole videos in RAM).
        """
        if list(indices) != sorted(set(indices)):
            raise ValueError(f"ClipHandle.read_frames: indices must be ascending & unique, got {list(indices)}")
        container, stream = self._open_cam(cam)
        out: Dict[int, np.ndarray] = {}
        container.seek(0, stream=stream)
        wanted = set(int(i) for i in indices)
        for pos, frame in enumerate(container.decode(stream)):
            if pos in wanted:
                out[pos] = frame.to_ndarray(format="rgb24")
            if pos >= max(wanted):
                break
        missing = sorted(wanted - out.keys())
        if missing:
            raise IndexError(
                f"ClipHandle: camera {cam!r} of {self.clip_uri} has {pos + 1} frames; "
                f"requested out-of-range indices {missing[:5]}"
            )
        return [out[int(i)] for i in indices]

    def frame_timestamps_ns(self, cam: str) -> np.ndarray:
        """PTS-derived per-frame timestamps (int64 ns); computed on demand by
        walking the stream once (PTS IS the exposure timestamp, spec §4.1)."""
        import av

        container, stream = self._open_cam(cam)
        time_base = stream.time_base
        pts_list = []
        container.seek(0, stream=stream)
        for frame in container.decode(stream):
            pts_list.append(frame.pts)
        return (np.asarray(pts_list, dtype=np.int64) * time_base * 1e9).astype(np.int64)

    # ----- topics / calib --------------------------------------------------------

    @property
    def ego_motion(self):
        """``topics/ego_motion.parquet`` as a pyarrow Table (read-once, cached)."""
        if self._ego_motion is None:
            import pyarrow.parquet as pq

            self._ego_motion = pq.read_table(self.clip_uri / "topics" / "ego_motion.parquet")
        return self._ego_motion

    @property
    def calib(self) -> dict:
        """``calib.json`` (read-once, cached): ``{"cameras": {cam: {intrinsics, extrinsics}}}``."""
        if self._calib is None:
            self._calib = json.loads((self.clip_uri / "calib.json").read_text(encoding="utf-8"))
        return self._calib


class ClipReader:
    """Per-process singleton facade with a bounded LRU over ``ClipHandle``s."""

    _instances: Dict[int, "ClipReader"] = {}

    def __init__(self, lru_size: int = 4):
        self._lru_size = max(1, int(lru_size))
        self._cache: "OrderedDict[str, ClipHandle]" = OrderedDict()

    @classmethod
    def instance(cls, lru_size: int = 4) -> "ClipReader":
        """Singleton per OS process (fork workers never share handles)."""
        key = os.getpid()
        if key not in cls._instances:
            cls._instances[key] = cls(lru_size=lru_size)
        return cls._instances[key]

    def handle(self, clip_uri: str) -> ClipHandle:
        cached = self._cache.get(clip_uri)
        if cached is not None:
            self._cache.move_to_end(clip_uri)
            return cached
        handle = ClipHandle(clip_uri)
        self._cache[clip_uri] = handle
        while len(self._cache) > self._lru_size:
            self._cache.popitem(last=False)
        return handle
