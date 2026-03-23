"""
Video frame reader.

Wraps cv2.VideoCapture to yield (frame_index, timestamp, ndarray) tuples
and expose video metadata (FPS, resolution, frame count).
"""

from __future__ import annotations

from typing import Generator, Tuple

import cv2
import numpy as np


class VideoReader:
    """Lazy, iterable video reader."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

    # ── Metadata ──────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    # ── Iteration ─────────────────────────────────────────────────────

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_index, timestamp_sec, bgr_frame) for every frame."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            ts = idx / self.fps
            yield idx, ts, frame
            idx += 1

    def read_frame(self, index: int) -> Tuple[float, np.ndarray]:
        """Read a single frame by index.

        Returns:
            (timestamp, bgr_frame)

        Raises:
            IndexError if frame cannot be read.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise IndexError(f"Cannot read frame {index}")
        return index / self.fps, frame

    # ── Lifecycle ─────────────────────────────────────────────────────

    def release(self) -> None:
        self._cap.release()

    def __del__(self) -> None:
        self.release()
