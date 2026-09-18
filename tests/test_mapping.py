"""Tests for the fitted eye-to-scene map.

Head-worn calibration has few points — a dozen or so, because the wearer has to
hold each one — so the tests are built around what happens at that sample size,
which is where the in-sample residual stops meaning anything.
"""

import numpy as np
import pytest

from headgaze.mapping import (
    EyeToSceneMap,
    cross_validated_residuals,
    slippage_indicator,
)


def synthetic_eye(n_points=12, *, noise=0.0, seed=0, offset=(0.0, 0.0)):
    """Pupil positions and the scene points they correspond to.

    The true relationship is mildly nonlinear, which is what makes a quadratic
    the right model and a linear one visibly insufficient.
    """
    rng = np.random.default_rng(seed)

    side = int(np.ceil(np.sqrt(n_points)))
    grid = np.array(
        [(i / (side - 1), j / (side - 1)) for j in range(side) for i in range(side)]
    )[:n_points]

    pupil = 120.0 + grid * 60.0 + np.array(offset)

    u, v = grid[:, 0] - 0.5, grid[:, 1] - 0.5
    scene = np.column_stack(
        [
            960.0 + 1500.0 * u + 260.0 * u**2 + 90.0 * u * v,
            540.0 + 820.0 * v - 180.0 * v**2 - 60.0 * u * v,
        ]
    )

    if noise:
        pupil = pupil + rng.normal(0.0, noise, size=pupil.shape)

    return pupil, scene


def test_fits_a_clean_relationship():
    pupil, scene = synthetic_eye(16)
    model = EyeToSceneMap(degree=2).fit(pupil, scene)

    assert model.fitted
    assert np.max(model.residuals_px(pupil, scene)) < 1.0


def test_quadratic_beats_linear_on_a_curved_relationship():
    pupil, scene = synthetic_eye(16)

    linear = EyeToSceneMap(degree=1).fit(pupil, scene)
    quadratic = EyeToSceneMap(degree=2).fit(pupil, scene)

    assert np.mean(quadratic.residuals_px(pupil, scene)) < np.mean(
        linear.residuals_px(pupil, scene)
    )


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        EyeToSceneMap().predict(np.zeros((1, 2)))


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        EyeToSceneMap().fit(np.zeros((10, 2)), np.zeros((8, 2)))


def test_requires_enough_samples_for_the_degree():
    """Six parameters cannot be fitted from four calibration points."""
    pupil, scene = synthetic_eye(4)
    with pytest.raises(ValueError, match="at least 6"):
        EyeToSceneMap(degree=2).fit(pupil, scene)


def test_n_parameters_matches_the_degree():
    assert EyeToSceneMap(degree=1).n_parameters == 3
    assert EyeToSceneMap(degree=2).n_parameters == 6
    assert EyeToSceneMap(degree=3).n_parameters == 10


def test_rejects_unsupported_degree():
    pupil, scene = synthetic_eye(25)
    with pytest.raises(ValueError, match="degree must be"):
        EyeToSceneMap(degree=4).fit(pupil, scene)


def test_constant_pupil_coordinate_does_not_produce_nan():
    """A wearer who never moves one axis gives a zero-variance feature."""
    pupil, scene = synthetic_eye(16)
    pupil[:, 1] = 130.0

    model = EyeToSceneMap(degree=2).fit(pupil, scene)
    assert np.all(np.isfinite(model.predict(pupil)))


def test_weights_downweight_bad_samples():
    """Low-confidence pupil fits should pull the map less.

    Half the calibration points are corrupted. Weighting them near zero should
    recover a map close to the one fitted on the clean half alone.
    """
    pupil, scene = synthetic_eye(16)

    corrupted = pupil.copy()
    corrupted[8:] += np.array([25.0, -30.0])

    weights = np.ones(16)
    weights[8:] = 0.001

    weighted = EyeToSceneMap(degree=2).fit(corrupted, scene, weights=weights)
    unweighted = EyeToSceneMap(degree=2).fit(corrupted, scene)

    clean_error_weighted = np.mean(weighted.residuals_px(pupil[:8], scene[:8]))
    clean_error_unweighted = np.mean(unweighted.residuals_px(pupil[:8], scene[:8]))

    assert clean_error_weighted < clean_error_unweighted


def test_rejects_negative_weights():
    pupil, scene = synthetic_eye(16)
    with pytest.raises(ValueError, match="non-negative"):
        EyeToSceneMap().fit(pupil, scene, weights=-np.ones(16))


def test_rejects_wrong_weight_count():
    pupil, scene = synthetic_eye(16)
    with pytest.raises(ValueError, match="one entry per sample"):
        EyeToSceneMap().fit(pupil, scene, weights=np.ones(5))


def test_in_sample_residual_is_optimistic_at_realistic_sample_sizes():
    """The reason leave-one-out exists in this module.

    Twelve calibration points against six parameters leaves half the degrees of
    freedom. The in-sample residual is not an estimate of tracker accuracy at
    that ratio, and quoting it as one overstates the rig substantially.
    """
    pupil, scene = synthetic_eye(12, noise=1.2, seed=3)

    model = EyeToSceneMap(degree=2).fit(pupil, scene)
    in_sample = float(np.mean(model.residuals_px(pupil, scene)))
    held_out = float(np.nanmean(cross_validated_residuals(pupil, scene)))

    assert held_out > in_sample


def test_cross_validated_residuals_are_finite_for_a_usable_calibration():
    pupil, scene = synthetic_eye(16, noise=0.8, seed=4)
    residuals = cross_validated_residuals(pupil, scene)

    assert residuals.shape == (16,)
    assert np.all(np.isfinite(residuals))


def test_cross_validation_degrades_gracefully_when_too_small():
    """Dropping one point from a minimal calibration leaves it unfittable."""
    pupil, scene = synthetic_eye(6)
    residuals = cross_validated_residuals(pupil, scene, degree=2)

    assert np.all(np.isnan(residuals))


def test_higher_degree_overfits_a_small_calibration():
    """Degree 3 has ten parameters; twelve points cannot support it.

    The in-sample residual improves while the held-out residual gets worse,
    which is the textbook signature and the reason degree 2 is the default.
    """
    pupil, scene = synthetic_eye(12, noise=1.5, seed=5)

    in_sample_2 = float(
        np.mean(EyeToSceneMap(degree=2).fit(pupil, scene).residuals_px(pupil, scene))
    )
    in_sample_3 = float(
        np.mean(EyeToSceneMap(degree=3).fit(pupil, scene).residuals_px(pupil, scene))
    )

    held_out_2 = float(np.nanmean(cross_validated_residuals(pupil, scene, degree=2)))
    held_out_3 = float(np.nanmean(cross_validated_residuals(pupil, scene, degree=3)))

    assert in_sample_3 <= in_sample_2
    assert held_out_3 > held_out_2


def test_slippage_is_zero_without_movement():
    pupil, _ = synthetic_eye(16)
    assert slippage_indicator(pupil, pupil) == pytest.approx(0.0, abs=1e-9)


def test_slippage_detects_a_bulk_shift():
    """Glasses moving on the face, which nothing else in the pipeline notices."""
    pupil, _ = synthetic_eye(16)
    shifted = pupil + np.array([6.0, -4.0])

    assert slippage_indicator(pupil, shifted) == pytest.approx(np.hypot(6.0, 4.0), rel=1e-6)


def test_slippage_rejects_empty_input():
    pupil, _ = synthetic_eye(16)
    with pytest.raises(ValueError, match="non-empty"):
        slippage_indicator(pupil, np.zeros((0, 2)))
