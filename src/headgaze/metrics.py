"""Screen geometry and gaze data quality, in degrees of visual angle.

Pixels are meaningless without the screen size and the viewing distance, so
every accuracy and precision figure goes through the true subtended angle
between two points as seen from the eye, not a small-angle approximation.
The eye is assumed to sit on the perpendicular through the screen centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Screen:
    width_px: int = 1920
    height_px: int = 1080
    width_cm: float = 34.5  # visible width of a 15.6-inch 16:9 panel
    distance_cm: float = 60.0

    @property
    def cm_per_px(self) -> float:
        return self.width_cm / self.width_px

    def eye_vectors(self, screen_px: np.ndarray) -> np.ndarray:
        """Screen pixels (n, 2) -> vectors from the eye to those points, in cm (n, 3)."""
        p = np.atleast_2d(np.asarray(screen_px, dtype=float))
        centred = (p - [self.width_px / 2, self.height_px / 2]) * self.cm_per_px
        depth = np.full((len(p), 1), self.distance_cm)
        return np.hstack([centred, depth])

    def angle_deg(self, a_px: np.ndarray, b_px: np.ndarray) -> np.ndarray:
        """Visual angle between screen points a and b, row by row (n,)."""
        va, vb = self.eye_vectors(a_px), self.eye_vectors(b_px)
        cos = np.sum(va * vb, axis=1) / (
            np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)
        )
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def accuracy_deg(screen: Screen, gaze_px: np.ndarray, target_px: np.ndarray) -> float:
    """Angle between the mean gaze position and the target: the systematic offset.

    Valid samples only; NaN rows (dropouts) are ignored. NaN if nothing is left.
    """
    gaze = np.atleast_2d(np.asarray(gaze_px, dtype=float))
    gaze = gaze[np.all(np.isfinite(gaze), axis=1)]
    if len(gaze) == 0:
        return float("nan")
    return float(screen.angle_deg(gaze.mean(axis=0), np.asarray(target_px, dtype=float))[0])


def precision_rms_s2s_deg(screen: Screen, gaze_px: np.ndarray) -> float:
    """RMS of the angles between successive valid samples: the jitter."""
    gaze = np.atleast_2d(np.asarray(gaze_px, dtype=float))
    gaze = gaze[np.all(np.isfinite(gaze), axis=1)]
    if len(gaze) < 2:
        return float("nan")
    steps = screen.angle_deg(gaze[:-1], gaze[1:])
    return float(np.sqrt(np.mean(steps**2)))


def data_loss_pct(valid: np.ndarray) -> float:
    valid = np.asarray(valid, dtype=bool)
    return float(100.0 * (1.0 - valid.mean())) if len(valid) else float("nan")
