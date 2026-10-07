"""From an eye-camera image to a fitted pupil. Needs OpenCV (the `camera` extra).

:mod:`headgaze.pupil` holds the numpy pieces; this module chains them on a real
image:

    greyscale -> blur -> glint suppression
    -> for several dark fractions: threshold -> clean the mask -> blobs
    -> per blob: boundary -> robust ellipse fit -> confidence and contrast
    -> the darkest believable blob is the seed -> threshold again halfway
       between its level and its surround's -> final fit

**Why several thresholds.** How much of the frame the pupil covers depends on
the rig, the person and the pupil's size, which changes with the light. A single
"darkest x %" threshold either cuts the pupil in half or swallows the iris with
it. Trying a few fractions and letting the ellipse fit judge each blob avoids
committing to one.

**Why a contrast test.** A threshold always finds *something*: in a frame with
no pupil (a blink, a camera pointing at skin) the darkest pixels are noise, and
a smooth enough noise blob can still fit an ellipse. A pupil is clearly darker
than the ring around it; noise is not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from headgaze.pupil import (
    PupilEstimate,
    confidence_from_fit,
    estimate_threshold,
    fit_ellipse_robust,
    suppress_glints,
)

DARK_FRACTIONS = (0.005, 0.01, 0.02, 0.04, 0.08)
_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def invalid_estimate() -> PupilEstimate:
    return PupilEstimate(np.array([np.nan, np.nan]), np.zeros(2), 0.0, 0.0, False)


def detect_pupil(
    image: np.ndarray,
    *,
    previous_centre: np.ndarray | None = None,
    dark_fractions: tuple[float, ...] = DARK_FRACTIONS,
    min_area_fraction: float = 0.002,
    max_area_fraction: float = 0.2,
    min_confidence: float = 0.4,
    min_contrast: float = 20.0,
) -> PupilEstimate:
    """Find the pupil in one eye-camera frame (greyscale or BGR, uint8).

    Returns an estimate with ``valid=False`` when nothing is plausible; downstream
    that is a dropout, never a guess. ``previous_centre`` (pixels) favours blobs
    near the last pupil, which keeps tracking stable when a lash shadow competes.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clean = suppress_glints(cv2.GaussianBlur(gray, (5, 5), 0))
    total = clean.shape[0] * clean.shape[1]
    diagonal = float(np.hypot(*clean.shape))

    candidates = []
    for fraction in dark_fractions:
        threshold = estimate_threshold(clean, pupil_fraction=fraction)
        mask = (clean <= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)  # lash speckle
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)  # holes left by glints

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        for i in np.argsort(areas)[::-1][:3]:  # the three largest blobs are enough
            if min_area_fraction * total <= areas[i] <= max_area_fraction * total:
                candidate = _fit_blob(clean, (labels == i + 1).astype(np.uint8))
                if candidate is not None:
                    candidates.append(candidate)

    seeds = [
        c for c in candidates if c.contrast >= min_contrast and c.estimate.confidence >= 0.2
    ]
    if seeds and previous_centre is not None and np.all(np.isfinite(previous_centre)):
        near = [
            c
            for c in seeds
            if np.linalg.norm(c.estimate.centre - previous_centre) < 0.15 * diagonal
        ]
        seeds = near or seeds
    if not seeds:
        return invalid_estimate()

    # The pupil is the darkest structure in the eye, so the darkest believable blob
    # is the seed. Its threshold rarely matches the pupil's edge: a low one keeps
    # only the dark core, the next one already merges lashes and iris. Thresholding
    # again halfway between the pupil's level and its surround's cuts at the edge.
    seed = min(seeds, key=lambda c: c.mean_level)
    final = _refine(clean, seed, max_area_fraction * total) or seed
    valid = final.estimate.confidence >= min_confidence and final.contrast >= min_contrast
    return replace(final.estimate, valid=valid)


@dataclass
class _Candidate:
    estimate: PupilEstimate
    blob: np.ndarray  # uint8 mask
    mean_level: float
    ring_level: float  # mean grey level just outside the blob

    @property
    def contrast(self) -> float:
        return self.ring_level - self.mean_level


def _fit_blob(clean: np.ndarray, blob: np.ndarray) -> _Candidate | None:
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len).reshape(-1, 2).astype(float)
    if len(points) < 5:
        return None
    centre, axes, angle = fit_ellipse_robust(points)
    confidence = confidence_from_fit(points, centre, axes, angle)

    inside = blob.astype(bool)
    ring = cv2.dilate(blob, _RING).astype(bool) & ~inside
    mean_level = float(clean[inside].mean())
    ring_level = float(clean[ring].mean()) if ring.any() else mean_level
    estimate = PupilEstimate(centre, axes, angle, confidence, valid=False)
    return _Candidate(estimate, blob, mean_level, ring_level)


def _refine(clean: np.ndarray, seed: _Candidate, max_area: float) -> _Candidate | None:
    """Re-threshold at the midpoint between the seed's level and its surround's."""
    level = (seed.mean_level + seed.ring_level) / 2.0
    mask = (clean <= level).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    overlap = labels[seed.blob.astype(bool)]
    overlap = overlap[overlap > 0]
    if len(overlap) == 0:
        return None
    blob = (labels == np.bincount(overlap).argmax()).astype(np.uint8)
    if blob.sum() > max_area:  # merged with something large: keep the seed instead
        return None
    return _fit_blob(clean, blob)


def draw_pupil(image: np.ndarray, pupil: PupilEstimate) -> np.ndarray:
    """A copy of the frame with the fitted ellipse drawn on it (green valid, red not)."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if np.all(np.isfinite(pupil.centre)) and pupil.axes[0] > 0:
        colour = (0, 200, 0) if pupil.valid else (0, 0, 230)
        centre = tuple(int(round(c)) for c in pupil.centre)
        axes = tuple(int(round(a)) for a in pupil.axes)
        cv2.ellipse(out, centre, axes, np.degrees(pupil.angle_rad), 0, 360, colour, 1)
        cv2.drawMarker(out, centre, colour, cv2.MARKER_CROSS, 8, 1)
    label = f"conf {pupil.confidence:.2f}"
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out
