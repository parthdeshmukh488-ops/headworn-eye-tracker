"""One frame pair in, one gaze sample out; plus calibration and validation from samples.

Kept free of any window or camera so the whole chain can be run on synthetic
frames in tests. Needs OpenCV (pupil and marker detection).

    eye frame   -> pupil centre (eye-camera px)  -> fitted map -> scene-camera px
    scene frame -> ArUco markers -> homography H (scene -> screen, every frame)
    gaze on screen = H(map(pupil))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from headgaze.aruco import detect_markers, make_detector
from headgaze.detect import detect_pupil
from headgaze.mapping import EyeToSceneMap, cross_validated_residuals
from headgaze.metrics import Screen, accuracy_deg, data_loss_pct, precision_rms_s2s_deg
from headgaze.pupil import PupilEstimate
from headgaze.screen import ScreenLocator, apply_homography

# Normalised screen positions. Calibration and validation never share a point:
# accuracy measured on the calibration points is the in-sample number.
CALIBRATION_POINTS = [(x, y) for y in (0.2, 0.5, 0.8) for x in (0.2, 0.5, 0.8)]
VALIDATION_POINTS = [(x, y) for y in (0.3, 0.55, 0.75) for x in (0.3, 0.55, 0.75)]


@dataclass
class Sample:
    t: float
    pupil: PupilEstimate
    markers: dict
    homography: np.ndarray | None  # scene -> screen
    reprojection_px: float
    gaze_px: np.ndarray | None  # on the screen, None when either stage failed


class Tracker:
    def __init__(self, locator: ScreenLocator, model: EyeToSceneMap | None = None):
        self.locator, self.model = locator, model
        self.detector = make_detector()
        self._previous_pupil = None

    def process(self, t: float, eye: np.ndarray, scene: np.ndarray) -> Sample:
        pupil = detect_pupil(eye, previous_centre=self._previous_pupil)
        if pupil.valid:
            self._previous_pupil = pupil.centre
        markers = detect_markers(scene, self.detector)
        homography, error = self.locator.locate(markers)
        gaze = None
        if self.model is not None and pupil.valid and homography is not None:
            gaze = apply_homography(homography, self.model.predict(pupil.centre))[0]
        return Sample(t, pupil, markers, homography, error, gaze)


def fit_calibration(
    samples: list[tuple[int, Sample]], points_px: np.ndarray
) -> tuple[EyeToSceneMap, dict]:
    """Fit the eye-to-scene map from (dot index, sample) pairs recorded during the dots.

    Each usable sample pairs its pupil centre with where the dot appears *in the
    scene image of that frame* (the dot's screen position pulled back through the
    frame's own homography), so head movement during calibration does no harm.
    """
    pupils, scenes, weights, groups = [], [], [], []
    for index, s in samples:
        if not s.pupil.valid or s.homography is None:
            continue
        scene_dot = apply_homography(np.linalg.inv(s.homography), points_px[index])[0]
        pupils.append(s.pupil.centre)
        scenes.append(scene_dot)
        weights.append(s.pupil.confidence)
        groups.append(index)
    if not pupils:
        raise ValueError("no usable samples: no frame had both a pupil and the screen markers")
    pupils, scenes, groups = np.array(pupils), np.array(scenes), np.array(groups)

    model = EyeToSceneMap(degree=2).fit(pupils, scenes, weights=np.array(weights))

    # Leave-one-DOT-out on per-dot medians: samples of one dot are nearly identical,
    # so leaving out single samples would leak the answer.
    dots = np.unique(groups)
    med_pupil = np.array([np.median(pupils[groups == d], axis=0) for d in dots])
    med_scene = np.array([np.median(scenes[groups == d], axis=0) for d in dots])
    held_out = cross_validated_residuals(med_pupil, med_scene)
    report = {
        "n_samples": int(len(pupils)),
        "samples_per_dot": {int(d): int((groups == d).sum()) for d in dots},
        "leave_one_dot_out_scene_px": {
            "mean": float(np.nanmean(held_out)),
            "max": float(np.nanmax(held_out)),
        },
        "pupil_reference": med_pupil.tolist(),
    }
    return model, report


def validation_summary(
    samples: list[tuple[int, Sample]], points_px: np.ndarray, screen: Screen
) -> dict:
    """Accuracy, precision and data loss over the validation dots, in degrees."""
    points = []
    for index, target in enumerate(points_px):
        rows = [s for i, s in samples if i == index]
        gaze = np.array(
            [s.gaze_px if s.gaze_px is not None else (np.nan, np.nan) for s in rows]
        )
        points.append(
            {
                "point": index,
                "target_px": [float(v) for v in target],
                "n_frames": len(rows),
                "accuracy_deg": accuracy_deg(screen, gaze, target)
                if len(rows)
                else float("nan"),
                "precision_rms_s2s_deg": precision_rms_s2s_deg(screen, gaze)
                if len(rows)
                else float("nan"),
            }
        )
    valid = np.array([s.gaze_px is not None for _, s in samples])
    times = np.array([s.t for _, s in samples])
    return {
        "accuracy_deg": float(np.nanmean([p["accuracy_deg"] for p in points])),
        "precision_rms_s2s_deg": float(
            np.nanmean([p["precision_rms_s2s_deg"] for p in points])
        ),
        "data_loss_pct": data_loss_pct(valid),
        "sampling_rate_hz": float(1.0 / np.median(np.diff(times))) if len(times) > 1 else None,
        "points": points,
    }
