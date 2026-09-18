"""Tests for the homography and the screen locator.

The homography is exact geometry, so most of these check it against known
transforms rather than tolerances: a synthetic projective map applied to points
must be recovered to numerical precision.

The locator tests are about a subtler property — whether a *wrong* detection
can be caught. It can only be caught when the system is over-determined, which
is the whole reason every marker corner is kept rather than just the centre.
"""

import numpy as np
import pytest

from headgaze.screen import (
    ScreenLocator,
    ScreenMarkers,
    apply_homography,
    homography_from_points,
    reprojection_error,
)

SCREEN_W, SCREEN_H = 1920, 1080
UNIT_QUAD = np.array([[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]])


def _random_homography(rng, scale=1.0):
    """A well-conditioned random projective transform."""
    h = np.eye(3)
    h[:2, :2] += rng.normal(0.0, 0.15 * scale, size=(2, 2))
    h[:2, 2] = rng.normal(0.0, 50.0 * scale, size=2)
    h[2, :2] = rng.normal(0.0, 1e-4, size=2)
    return h / h[2, 2]


def _observe(markers, truth):
    """What the scene camera sees, given a screen-to-scene homography."""
    return {
        marker_id: apply_homography(truth, corners)
        for marker_id, corners in markers.corners_px.items()
    }


# --------------------------------------------------------------- marker layout


def test_for_screen_makes_four_markers_of_four_corners():
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)

    assert len(markers.corners_px) == 4
    for corners in markers.corners_px.values():
        assert np.asarray(corners).shape == (4, 2)


def test_for_screen_places_markers_inside_the_screen():
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H, marker_size_px=120, inset_px=20)
    everything = np.vstack(list(markers.corners_px.values()))

    assert everything[:, 0].min() >= 0.0
    assert everything[:, 1].min() >= 0.0
    assert everything[:, 0].max() <= SCREEN_W
    assert everything[:, 1].max() <= SCREEN_H


def test_for_screen_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="distinct"):
        ScreenMarkers.for_screen(SCREEN_W, SCREEN_H, ids=(0, 1, 1, 3))


def test_for_screen_rejects_bad_marker_size():
    with pytest.raises(ValueError, match="marker_size_px"):
        ScreenMarkers.for_screen(SCREEN_W, SCREEN_H, marker_size_px=0.0)


def test_single_marker_layout_is_rejected():
    """One marker's four corners exactly determine the homography.

    Exactly determined means zero residual for any input, so nothing can be
    validated. Refusing the layout is better than silently offering a check
    that cannot fail.
    """
    with pytest.raises(ValueError, match="at least 2 markers"):
        ScreenMarkers(corners_px={0: UNIT_QUAD})


def test_marker_corners_shape_is_checked():
    with pytest.raises(ValueError, match=r"\(4, 2\)"):
        ScreenMarkers(corners_px={0: UNIT_QUAD, 1: np.zeros((3, 2))})


# ----------------------------------------------------------------- homography


def test_identity_homography_is_recovered():
    h = homography_from_points(UNIT_QUAD, UNIT_QUAD)
    assert h == pytest.approx(np.eye(3), abs=1e-9)


def test_recovers_a_known_projective_transform():
    rng = np.random.default_rng(0)
    truth = _random_homography(rng)

    destination = apply_homography(truth, UNIT_QUAD)
    recovered = homography_from_points(UNIT_QUAD, destination)

    # A homography is defined up to scale, so compare what it does rather than
    # its entries.
    probe = rng.uniform([0, 0], [SCREEN_W, SCREEN_H], size=(40, 2))
    assert apply_homography(recovered, probe) == pytest.approx(
        apply_homography(truth, probe), rel=1e-6, abs=1e-6
    )


def test_normalisation_matters_at_pixel_scale():
    """Why the normalisation step inside the DLT is not optional.

    Without it, a design matrix built from raw coordinates in the thousands is
    poorly conditioned. The test is that accuracy holds at exactly that scale,
    which is the regime a 4K screen puts it in.
    """
    rng = np.random.default_rng(3)
    truth = _random_homography(rng)

    big = np.array([[0.0, 0.0], [3840.0, 0.0], [3840.0, 2160.0], [0.0, 2160.0]])
    destination = apply_homography(truth, big)

    recovered = homography_from_points(big, destination)
    assert reprojection_error(recovered, big, destination) < 1e-6


def test_rejects_too_few_correspondences():
    with pytest.raises(ValueError, match="at least 4"):
        homography_from_points(UNIT_QUAD[:3], UNIT_QUAD[:3])


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        homography_from_points(UNIT_QUAD, UNIT_QUAD[:3])


def test_collinear_points_are_rejected():
    collinear = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        homography_from_points(collinear, UNIT_QUAD)


def test_point_at_infinity_becomes_nan_not_inf():
    """A silent inf would propagate into a gaze estimate; NaN is rejected."""
    horizon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    mapped = apply_homography(horizon, np.array([[0.0, 5.0]]))
    assert np.all(np.isnan(mapped))


def test_four_correspondences_always_fit_exactly():
    """The fact that forces the whole corner-based design.

    Eight unknowns, eight equations. The residual is zero for *any* four
    correspondences, including nonsense ones, so it says nothing about whether
    the detection was correct.
    """
    rng = np.random.default_rng(11)
    nonsense = rng.uniform(0.0, 500.0, size=(4, 2))

    h = homography_from_points(nonsense, UNIT_QUAD)
    assert reprojection_error(h, nonsense, UNIT_QUAD) < 1e-6


# -------------------------------------------------------------------- locator


def test_locator_round_trip_maps_gaze_onto_the_screen():
    rng = np.random.default_rng(6)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers)

    truth = _random_homography(rng)  # screen -> scene image
    h, error = locator.locate(_observe(markers, truth))

    assert h is not None
    assert error < 1e-6

    wanted = np.array([[1200.0, 700.0]])
    in_scene = apply_homography(truth, wanted)
    assert apply_homography(h, in_scene) == pytest.approx(wanted, abs=1e-6)


def test_locator_needs_a_minimum_number_of_markers():
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers, min_markers=2)

    rng = np.random.default_rng(7)
    observed = _observe(markers, _random_homography(rng))
    only_one = {markers.ids[0]: observed[markers.ids[0]]}

    h, error = locator.locate(only_one)

    assert h is None
    assert np.isinf(error)


def test_locator_works_with_two_of_four_markers_visible():
    """The wearer turning their head takes markers out of frame constantly."""
    rng = np.random.default_rng(8)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers)

    truth = _random_homography(rng)
    observed = _observe(markers, truth)
    partial = {
        markers.ids[0]: observed[markers.ids[0]],
        markers.ids[2]: observed[markers.ids[2]],
    }

    h, _ = locator.locate(partial)

    assert h is not None
    wanted = np.array([[800.0, 400.0]])
    assert apply_homography(h, apply_homography(truth, wanted)) == pytest.approx(
        wanted, abs=1e-4
    )


def test_locator_ignores_unrelated_marker_ids():
    """Other ArUco markers in the room must not break anything."""
    rng = np.random.default_rng(4)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)

    observed = _observe(markers, _random_homography(rng))
    observed[77] = np.array([[5.0, 5.0], [9.0, 5.0], [9.0, 9.0], [5.0, 9.0]])  # a poster

    h, error = ScreenLocator(markers).locate(observed)

    assert h is not None
    assert error < 1e-6


def test_locator_catches_a_displaced_marker():
    """The check that only works because the system is over-determined."""
    rng = np.random.default_rng(5)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers, max_reprojection_px=5.0)

    observed = _observe(markers, _random_homography(rng))
    bad_id = markers.ids[2]
    observed[bad_id] = observed[bad_id] + np.array([400.0, -300.0])

    h, error = locator.locate(observed)

    assert h is None
    assert error > 5.0


def test_locator_catches_two_swapped_markers():
    """A mislabelled detection maps gaze confidently to the wrong place."""
    rng = np.random.default_rng(2)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers, max_reprojection_px=8.0)

    observed = _observe(markers, _random_homography(rng))
    a, b = markers.ids[1], markers.ids[3]
    observed[a], observed[b] = observed[b], observed[a]

    h, error = locator.locate(observed)

    assert h is None
    assert error > 8.0


def test_locator_rejects_a_malformed_detection():
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers)

    with pytest.raises(ValueError, match=r"\(4, 2\)"):
        locator.locate({markers.ids[0]: np.zeros((2, 2)), markers.ids[1]: np.zeros((4, 2))})


def test_locator_tolerates_detection_noise():
    """Sub-pixel jitter on every corner should not reject the frame."""
    rng = np.random.default_rng(9)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers, max_reprojection_px=8.0)

    observed = _observe(markers, _random_homography(rng))
    noisy = {k: v + rng.normal(0.0, 0.3, size=v.shape) for k, v in observed.items()}

    h, error = locator.locate(noisy)

    assert h is not None
    assert error < 8.0
