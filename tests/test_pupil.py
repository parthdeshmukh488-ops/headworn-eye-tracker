"""Tests for pupil detection in visible light.

Synthetic ellipses with known parameters, then the same ellipses degraded the
three ways a real visible-light eye image is degraded: glints on the cornea,
eyelashes crossing the pupil, and low iris contrast.
"""

import numpy as np
import pytest

from headgaze.pupil import (
    confidence_from_fit,
    estimate_threshold,
    fit_ellipse_robust,
    suppress_glints,
)


def ellipse_points(centre, axes, angle, n=60, noise=0.0, seed=0):
    """Points on an ellipse boundary, optionally with radial noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    local = np.column_stack([axes[0] * np.cos(t), axes[1] * np.sin(t)])
    if noise:
        local = local * (1.0 + rng.normal(0.0, noise, size=(n, 1)))

    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    return local @ rotation.T + np.asarray(centre)


def test_recovers_a_circle():
    points = ellipse_points((100.0, 80.0), (20.0, 20.0), 0.0)
    centre, axes, _ = fit_ellipse_robust(points)

    assert centre == pytest.approx([100.0, 80.0], abs=0.1)
    assert axes[0] == pytest.approx(20.0, rel=0.02)
    assert axes[1] == pytest.approx(20.0, rel=0.02)


def test_recovers_an_ellipse_centre_and_axes():
    points = ellipse_points((150.0, 120.0), (30.0, 18.0), 0.0)
    centre, axes, _ = fit_ellipse_robust(points)

    assert centre == pytest.approx([150.0, 120.0], abs=0.2)
    assert axes[0] == pytest.approx(30.0, rel=0.05)
    assert axes[1] == pytest.approx(18.0, rel=0.05)


def test_recovers_a_rotated_ellipse_centre():
    """The centre is what feeds the gaze map, so it is the number that matters.

    The reported angle is only defined modulo pi and swaps meaning when the
    axes are close in length, so the centre is asserted and the angle is not.
    """
    points = ellipse_points((200.0, 90.0), (28.0, 14.0), np.pi / 5)
    centre, axes, _ = fit_ellipse_robust(points)

    assert centre == pytest.approx([200.0, 90.0], abs=0.3)
    assert axes[0] > axes[1]


def test_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 5"):
        fit_ellipse_robust(np.zeros((4, 2)))


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        fit_ellipse_robust(np.zeros((10, 3)))


def test_survives_noise_on_the_boundary():
    points = ellipse_points((120.0, 100.0), (22.0, 16.0), 0.3, noise=0.03, seed=1)
    centre, _, _ = fit_ellipse_robust(points)
    assert centre == pytest.approx([120.0, 100.0], abs=1.5)


def test_trimming_beats_plain_least_squares_on_eyelash_outliers():
    """The reason the fit is robust rather than plain least squares.

    A handful of eyelash pixels pull an unweighted conic fit several pixels off
    centre, and at this rig's geometry a few pixels of pupil error is around a
    degree of gaze error.
    """
    truth = np.array([140.0, 110.0])
    points = ellipse_points(truth, (24.0, 17.0), 0.2, n=70, seed=2)

    # Eyelashes: a short dark streak crossing the upper pupil.
    outliers = np.column_stack(
        [np.linspace(130.0, 150.0, 8), np.full(8, 128.0) + np.linspace(0, 4, 8)]
    )
    contaminated = np.vstack([points, outliers])

    robust_centre, _, _ = fit_ellipse_robust(contaminated, iterations=3, keep_fraction=0.85)
    plain_centre, _, _ = fit_ellipse_robust(contaminated, iterations=1, keep_fraction=1.0)

    robust_error = np.linalg.norm(robust_centre - truth)
    plain_error = np.linalg.norm(plain_centre - truth)

    assert robust_error < plain_error
    assert robust_error < 1.5


def test_suppress_glints_removes_bright_pixels():
    rng = np.random.default_rng(0)
    image = rng.normal(60.0, 5.0, size=(40, 40))
    image[18:21, 18:21] = 255.0  # a corneal reflection inside the pupil

    cleaned = suppress_glints(image, percentile=99.0)

    assert cleaned.max() < 200.0
    assert cleaned[19, 19] < 100.0


def test_suppress_glints_leaves_a_clean_image_alone():
    """Only the extreme tail should move, so the image stays usable."""
    rng = np.random.default_rng(1)
    image = rng.normal(60.0, 5.0, size=(50, 50))

    cleaned = suppress_glints(image, percentile=99.0)
    changed = np.mean(cleaned != image.astype(np.float32))

    assert changed < 0.02


def test_suppress_glints_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        suppress_glints(np.zeros((0, 0)))


def test_threshold_adapts_to_exposure():
    """A fixed threshold cannot survive the wearer turning towards a window."""
    rng = np.random.default_rng(2)
    dim = rng.normal(40.0, 8.0, size=(60, 60))
    bright = dim + 90.0

    assert estimate_threshold(bright) > estimate_threshold(dim)


def test_threshold_sits_below_the_bulk_of_the_image():
    rng = np.random.default_rng(3)
    image = rng.normal(100.0, 15.0, size=(80, 80))
    image[30:40, 30:40] = 20.0  # the pupil

    threshold = estimate_threshold(image, pupil_fraction=0.06)

    assert threshold < np.median(image)


def test_threshold_rejects_bad_fraction():
    with pytest.raises(ValueError, match="pupil_fraction"):
        estimate_threshold(np.zeros((10, 10)), pupil_fraction=0.0)


def test_confidence_is_high_for_a_clean_fit():
    points = ellipse_points((100.0, 100.0), (20.0, 16.0), 0.0)
    centre, axes, angle = fit_ellipse_robust(points)

    assert confidence_from_fit(points, centre, axes, angle) > 0.9


def test_confidence_falls_when_points_do_not_lie_on_the_ellipse():
    rng = np.random.default_rng(4)
    scatter = rng.uniform(80.0, 120.0, size=(40, 2))
    centre, axes, angle = fit_ellipse_robust(scatter)

    assert confidence_from_fit(scatter, centre, axes, angle) < 0.7


def test_confidence_penalises_an_implausibly_elongated_fit():
    """Fitting the eyelid edge produces a long, thin, confident-looking ellipse."""
    points = ellipse_points((100.0, 100.0), (60.0, 4.0), 0.0)
    centre, axes, angle = fit_ellipse_robust(points)

    assert confidence_from_fit(points, centre, axes, angle, max_aspect=4.0) < 0.5


def test_confidence_penalises_a_tiny_fit():
    points = ellipse_points((100.0, 100.0), (2.0, 1.5), 0.0)
    centre, axes, angle = fit_ellipse_robust(points)

    assert confidence_from_fit(points, centre, axes, angle, min_axis_px=3.0) < 1.0


def test_confidence_is_zero_with_too_few_points():
    assert confidence_from_fit(np.zeros((3, 2)), np.zeros(2), np.ones(2), 0.0) == 0.0
