"""
Video frame writer.

Wraps cv2.VideoWriter for codec-agnostic output, then re-encodes
to H.264 via ffmpeg for universal playback compatibility.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Tuple

import cv2
import numpy as np


class VideoWriter:
    """Wrapper around cv2.VideoWriter with ffmpeg H.264 re-encoding."""

    def __init__(
        self,
        path: str,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self._final_path = path
        self._fps = fps
        # Write to a temp file first, then re-encode
        base, ext = os.path.splitext(path)
        self._tmp_path = f"{base}_raw{ext}"

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            self._tmp_path, fourcc, fps, resolution
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self._tmp_path}")
        self._frames_written = 0
        self._released = False

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame."""
        self._writer.write(frame)
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def release(self) -> None:
        if self._released:
            return
        self._writer.release()
        self._released = True
        self._reencode()

    def _reencode(self) -> None:
        """Re-encode the raw file to H.264 using ffmpeg."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            # No ffmpeg — just rename the raw file as-is
            os.replace(self._tmp_path, self._final_path)
            return

        cmd = [
            ffmpeg, "-y",
            "-i", self._tmp_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self._final_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            os.remove(self._tmp_path)
        except subprocess.CalledProcessError as exc:
            # Fall back to raw file if ffmpeg fails
            print(f"  ⚠ ffmpeg re-encode failed: {exc.stderr.decode()}")
            os.replace(self._tmp_path, self._final_path)

    def __del__(self) -> None:
        self.release()
