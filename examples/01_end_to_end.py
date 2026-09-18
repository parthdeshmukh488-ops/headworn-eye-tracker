"""The full pipeline on a synthetic wearer, with the head moving.

Pupil in the eye camera, through the fitted eye-to-scene map, through the
per-frame screen homography, to a point on the screen — then scored against
where the synthetic wearer was actually looking.

The head moves throughout. That is the part worth watching: the eye-to-scene
map is fitted once and never updated, and it does not need to be, because every
frame's head pose is absorbed by recomputing the homography from the markers.
If the two stages were conflated, this is where it would fall apart.

Everything here is synthetic. It demonstrates that the geometry composes
correctly and that the error reporting is honest; it is not a measurement of a
real rig.
"""

import numpy as np

from headgaze.mapping import EyeToSceneMap, cross_validated_residuals
from headgaze.screen import ScreenLocator, ScreenMarkers, apply_homography

SCREEN_W, SCREEN_H = 1920, 1080
SCREEN_WIDTH_MM = 531.0
VIEWING_DISTANCE_MM = 600.0

N_CALIBRATION = 12
N_TEST = 200
PUPIL_NOISE_PX = 0.4
SEED = 20260918


def screen_to_pupil(screen_xy):
    """The true, unknown-to-the-system relationship, inverted.

    A real eye's relationship to the eye camera is monotonic and curved. The
    cubic terms are what make a quadratic fit imperfect rather than exact,
    which is the realistic case.
    """
    u = screen_xy[:, 0] / SCREEN_W - 0.5
    v = screen_xy[:, 1] / SCREEN_H - 0.5

    return np.column_stack(
        [
            150.0 + 52.0 * u + 6.0 * u**3 + 3.0 * u * v,
            130.0 + 44.0 * v - 5.0 * v**3 - 2.0 * u * v,
        ]
    )


def head_pose_homography(rng, magnitude=1.0):
    """A screen-to-scene-image homography for one head pose."""
    h = np.eye(3)
    h[:2, :2] += rng.normal(0.0, 0.05 * magnitude, size=(2, 2))
    h[:2, 2] = rng.normal(0.0, 40.0 * magnitude, size=2)
    h[2, :2] = rng.normal(0.0, 3e-5 * magnitude, size=2)
    return h / h[2, 2]


def px_to_degrees(error_px):
    """Screen pixels to degrees of visual angle, near the screen centre."""
    mm_per_px = SCREEN_WIDTH_MM / SCREEN_W
    return np.degrees(np.arctan2(error_px * mm_per_px, VIEWING_DISTANCE_MM))


def calibration_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    margin = 0.12
    coords = np.linspace(margin, 1.0 - margin, side)
    grid = np.array([(x, y) for y in coords for x in coords])[:n]
    return grid * np.array([SCREEN_W, SCREEN_H])


def main() -> None:
    rng = np.random.default_rng(SEED)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers, max_reprojection_px=8.0)

    # ---- calibration: wearer holds still, looks at a grid of points --------
    calib_screen = calibration_grid(N_CALIBRATION)
    calib_pupil = screen_to_pupil(calib_screen) + rng.normal(
        0.0, PUPIL_NOISE_PX, size=(N_CALIBRATION, 2)
    )

    calib_pose = head_pose_homography(rng, magnitude=0.4)
    calib_scene = apply_homography(calib_pose, calib_screen)

    model = EyeToSceneMap(degree=2).fit(calib_pupil, calib_scene)

    in_sample_px = float(np.mean(model.residuals_px(calib_pupil, calib_scene)))
    held_out_px = float(np.nanmean(cross_validated_residuals(calib_pupil, calib_scene)))

    print(f"Calibration: {N_CALIBRATION} points, {model.n_parameters} parameters")
    print(
        f"  in-sample residual     {in_sample_px:7.2f} px   ({px_to_degrees(in_sample_px):.2f} deg)"
    )
    print(
        f"  leave-one-out          {held_out_px:7.2f} px   ({px_to_degrees(held_out_px):.2f} deg)"
    )
    print(f"  overstatement          {held_out_px / in_sample_px:.1f}x if in-sample is quoted")
    print()

    # ---- test: wearer looks around while the head keeps moving ------------
    targets = rng.uniform([0.12, 0.12], [0.88, 0.88], size=(N_TEST, 2)) * np.array(
        [SCREEN_W, SCREEN_H]
    )
    pupil = screen_to_pupil(targets) + rng.normal(0.0, PUPIL_NOISE_PX, size=(N_TEST, 2))

    errors_px = []
    rejected = 0

    for i in range(N_TEST):
        pose = head_pose_homography(rng, magnitude=1.0)

        detected = {
            marker_id: apply_homography(pose, corners)
            + rng.normal(0.0, 0.3, size=(4, 2))  # detector jitter
            for marker_id, corners in markers.corners_px.items()
        }

        homography, _ = locator.locate(detected)
        if homography is None:
            rejected += 1
            continue

        in_scene = model.predict(pupil[i])
        on_screen = apply_homography(homography, in_scene)[0]
        errors_px.append(float(np.linalg.norm(on_screen - targets[i])))

    errors_px = np.array(errors_px)
    errors_deg = px_to_degrees(errors_px)

    print(f"Test: {N_TEST} gaze points, head pose different on every frame")
    print(f"  frames rejected        {rejected} ({rejected / N_TEST * 100:.1f}%)")
    print(f"  accuracy, mean         {errors_deg.mean():.2f} deg")
    print(f"  accuracy, median       {np.median(errors_deg):.2f} deg")
    print(f"  accuracy, p95          {np.percentile(errors_deg, 95):.2f} deg")
    print()
    print("The head pose is different on every one of those frames and the")
    print("eye-to-scene map was never refitted. Head motion is absorbed entirely")
    print("by recomputing the homography from the markers, which is why the two")
    print("stages are kept separate.")


if __name__ == "__main__":
    main()
