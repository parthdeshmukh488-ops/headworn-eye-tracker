"""Cameras and the fullscreen display for live sessions. Needs OpenCV.

Two USB cameras are read as a pair: both are *grabbed* first and decoded
afterwards, so the eye and scene frames of one sample are as close in time as
the drivers allow. A file path instead of a camera index replays a recording,
which is how the live code is tested without hardware.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

WINDOW = "headgaze"


def open_capture(source: str | int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        # MJPG keeps two cameras within the bandwidth of one USB 2 bus.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera or file {source!r}")
    return cap


@dataclass
class FramePair:
    t: float
    eye: np.ndarray
    scene: np.ndarray


class CameraPair:
    def __init__(self, eye: str | int, scene: str | int):
        self.eye = open_capture(eye)
        self.scene = open_capture(scene)

    def read(self) -> FramePair | None:
        if not (self.eye.grab() and self.scene.grab()):
            return None
        t = time.time()
        ok_eye, eye = self.eye.retrieve()
        ok_scene, scene = self.scene.retrieve()
        return FramePair(t, eye, scene) if ok_eye and ok_scene else None

    def close(self) -> None:
        self.eye.release()
        self.scene.release()


class Display:
    """One fullscreen OpenCV window showing a screen-sized image."""

    def __init__(self, width_px: int, height_px: int):
        self.width, self.height = width_px, height_px
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, image: np.ndarray, wait_ms: int = 1) -> int:
        cv2.imshow(WINDOW, image)
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(WINDOW)


def draw_dot(base: np.ndarray, xy_px, progress: float) -> np.ndarray:
    """A calibration dot whose outer ring shrinks in the first part of its time,
    pulling the gaze onto the centre."""
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else base.copy()
    centre = tuple(int(round(v)) for v in xy_px)
    radius = int(26 - 16 * min(progress, 1.0))
    cv2.circle(out, centre, radius, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(out, centre, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def put_lines(image: np.ndarray, lines: list[str], origin=(None, None), scale: float = 0.9):
    """Centred text block (default) or at a given origin."""
    h, w = image.shape[:2]
    x0 = origin[0]
    y = origin[1] if origin[1] is not None else h // 2 - 20 * len(lines)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = x0 if x0 is not None else (w - tw) // 2
        cv2.putText(
            image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 2, cv2.LINE_AA
        )
        y += th + 18
    return image
