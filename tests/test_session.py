"""The whole chain on synthetic camera frames: pupil -> map -> homography -> screen."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from headgaze.aruco import render_screen  # noqa: E402
from headgaze.metrics import Screen  # noqa: E402
from headgaze.screen import ScreenLocator, ScreenMarkers, apply_homography  # noqa: E402
from headgaze.session import (  # noqa: E402
    CALIBRATION_POINTS,
    VALIDATION_POINTS,
    Tracker,
    fit_calibration,
    validation_summary,
)
from headgaze.synthetic import synthetic_eye  # noqa: E402

W, H = 1920, 1080


def pupil_for_scene_point(g):
    """The (unknown to the tracker) eye: pupil position for a gaze point in the scene image."""
    u, v = (g[0] - 320) / 320, (g[1] - 240) / 240
    return np.array([160 + 55 * u + 4 * u * u, 120 + 35 * v - 3 * u * v])


def head_pose(rng):
    """screen -> scene homography for one head pose; the head drifts during the run."""
    s, a = 0.30, np.radians(rng.normal(0, 3))
    return np.array(
        [
            [s * np.cos(a), -s * np.sin(a), 30 + rng.normal(0, 10)],
            [s * np.sin(a), s * np.cos(a), 60 + rng.normal(0, 10)],
            [rng.normal(0, 1.5e-5), rng.normal(0, 1.5e-5), 1.0],
        ]
    )


def record(tracker, rng, screen_img, points_norm, frames_per_dot):
    records = []
    for index, p in enumerate(np.array(points_norm) * [W, H]):
        for _ in range(frames_per_dot):
            m = head_pose(rng)
            scene = cv2.warpPerspective(screen_img, m, (640, 480), borderValue=60)
            pupil = pupil_for_scene_point(apply_homography(m, p)[0])
            eye = synthetic_eye(tuple(pupil), rng=rng)
            records.append((index, tracker.process(len(records) / 30, eye, scene)))
    return records


def test_calibrate_then_validate_with_a_moving_head():
    rng = np.random.default_rng(7)
    markers = ScreenMarkers.for_screen(W, H)
    screen_img = render_screen(markers, W, H)

    cal = record(Tracker(ScreenLocator(markers)), rng, screen_img, CALIBRATION_POINTS, 6)
    model, report = fit_calibration(cal, np.array(CALIBRATION_POINTS) * [W, H])
    # 1 scene px is about 3.3 screen px here, so 6 scene px is roughly 0.35 deg.
    assert report["leave_one_dot_out_scene_px"]["mean"] < 6.0

    val = record(Tracker(ScreenLocator(markers), model), rng, screen_img, VALIDATION_POINTS, 6)
    summary = validation_summary(val, np.array(VALIDATION_POINTS) * [W, H], Screen())
    assert summary["data_loss_pct"] == 0.0
    assert summary["accuracy_deg"] < 0.5  # synthetic eye, so this tests the code, not a rig
