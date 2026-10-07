"""The image-level pipeline on synthetic camera frames. Needs OpenCV.

Example 01 starts from pupil *positions*. This one starts from *images*: a
synthetic eye with glints and lashes for the pupil detector, and a rendered
marker screen seen through a tilted scene camera for the ArUco stage. Then the
full chain (calibrate on 9 dots, validate on 9 others) with the head moving on
every frame.

Everything here is synthetic. It tests that the code finds what it should and
composes correctly; a real eye under room light is harder, and only a recorded
session can say by how much.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_session import CALIBRATION_POINTS, VALIDATION_POINTS, H, W, record  # noqa: E402

from headgaze.aruco import render_screen  # noqa: E402
from headgaze.detect import detect_pupil  # noqa: E402
from headgaze.metrics import Screen  # noqa: E402
from headgaze.screen import ScreenLocator, ScreenMarkers  # noqa: E402
from headgaze.session import Tracker, fit_calibration, validation_summary  # noqa: E402
from headgaze.synthetic import synthetic_eye  # noqa: E402

SEED = 20261007


def pupil_detection(rng) -> None:
    print("Pupil detection, 200 synthetic eyes per iris colour (glints and lashes on)")
    for name, level in [("dark brown", 70), ("brown", 95), ("light blue", 140)]:
        errors, dropouts = [], 0
        for _ in range(200):
            centre = (rng.uniform(110, 210), rng.uniform(90, 150))
            pupil = detect_pupil(synthetic_eye(centre, iris_level=level, rng=rng))
            if not pupil.valid:
                dropouts += 1
                continue
            errors.append(np.linalg.norm(pupil.centre - centre))
        errors = np.array(errors)
        print(
            f"  {name:<11} centre error median {np.median(errors):.2f} px, "
            f"p95 {np.percentile(errors, 95):.2f} px, dropouts {dropouts}/200"
        )
    print()


def full_chain(rng) -> None:
    markers = ScreenMarkers.for_screen(W, H)
    screen_img = render_screen(markers, W, H)
    cal = record(Tracker(ScreenLocator(markers)), rng, screen_img, CALIBRATION_POINTS, 6)
    model, report = fit_calibration(cal, np.array(CALIBRATION_POINTS) * [W, H])
    val = record(Tracker(ScreenLocator(markers), model), rng, screen_img, VALIDATION_POINTS, 6)
    summary = validation_summary(val, np.array(VALIDATION_POINTS) * [W, H], Screen())
    loo = report["leave_one_dot_out_scene_px"]
    print("Full chain from images, head pose different on every frame")
    print(f"  calibration, leave-one-dot-out   {loo['mean']:.1f} scene px (max {loo['max']:.1f})")
    print(f"  validation accuracy              {summary['accuracy_deg']:.2f} deg")
    print(f"  validation precision (RMS-S2S)   {summary['precision_rms_s2s_deg']:.2f} deg")
    print(f"  data loss                        {summary['data_loss_pct']:.1f} %")


if __name__ == "__main__":
    generator = np.random.default_rng(SEED)
    pupil_detection(generator)
    full_chain(generator)
