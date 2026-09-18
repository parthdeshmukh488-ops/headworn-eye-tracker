"""Mapping pupil position in the eye camera to a point in the scene camera.

This is the calibration that makes a head-worn tracker work, and it is a
different problem from the screen homography in :mod:`headgaze.screen`.

- **Eye to scene** (here) is a *fitted* relationship. It depends on the rig's
  geometry, the wearer's eye, and how the glasses sit on their face. It has no
  closed form and it changes if the glasses are nudged.
- **Scene to screen** (there) is a *geometric* relationship. A homography,
  solved from the detected marker corners, recomputed every frame.

Keeping them separate is what lets the head move freely. Only the second one
depends on head pose, and it is solved from scratch each frame, so the fitted
part never has to account for where the head is.

**The parallax limitation, stated up front.** The eye camera sees the eye, not
what the eye is looking at, so this map is fitted at one depth and is only
exact at that depth. A wearer calibrated at a screen 60 cm away and then
looking at something 3 m away has a systematic offset. Fixing it properly needs
a stereo scene camera or a depth estimate; this rig has neither, so the
limitation is documented rather than corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _design_matrix(pupil_xy: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial expansion of the pupil position, with a bias column.

    Degree 2 is the default for the same reason as in any gaze pipeline: the
    eye-to-scene relationship is curved, and degree 3 has enough freedom to
    chase noise through a calibration of a dozen points.
    """
    pupil_xy = np.atleast_2d(np.asarray(pupil_xy, dtype=float))
    if pupil_xy.shape[1] != 2:
        raise ValueError(f"pupil_xy must be (n, 2), got {pupil_xy.shape}")

    x, y = pupil_xy[:, 0], pupil_xy[:, 1]
    ones = np.ones_like(x)

    if degree == 1:
        columns = [ones, x, y]
    elif degree == 2:
        columns = [ones, x, y, x * x, x * y, y * y]
    elif degree == 3:
        columns = [ones, x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3]
    else:
        raise ValueError(f"degree must be 1, 2 or 3, got {degree}")

    return np.column_stack(columns)


@dataclass
class EyeToSceneMap:
    """Fitted map from pupil centre in the eye camera to a scene-camera point.

    Args:
        degree: polynomial degree.
        ridge: small L2 penalty, for conditioning. The bias is not penalised.
    """

    degree: int = 2
    ridge: float = 1e-6

    _coefficients: np.ndarray | None = field(default=None, repr=False)
    _pupil_mean: np.ndarray | None = field(default=None, repr=False)
    _pupil_scale: np.ndarray | None = field(default=None, repr=False)

    @property
    def fitted(self) -> bool:
        return self._coefficients is not None

    @property
    def n_parameters(self) -> int:
        return _design_matrix(np.zeros((1, 2)), self.degree).shape[1]

    def _standardise(self, pupil_xy: np.ndarray) -> np.ndarray:
        pupil_xy = np.atleast_2d(np.asarray(pupil_xy, dtype=float))
        return (pupil_xy - self._pupil_mean) / self._pupil_scale

    def fit(
        self,
        pupil_xy: np.ndarray,
        scene_xy: np.ndarray,
        *,
        weights: np.ndarray | None = None,
    ) -> EyeToSceneMap:
        """Fit the map from calibration samples.

        Args:
            pupil_xy: (n, 2) pupil centres in eye-camera pixels.
            scene_xy: (n, 2) the corresponding points in scene-camera pixels.
            weights: optional (n,) per-sample weights. Pass the pupil fit
                confidence here — a sample from a half-blinked frame should not
                carry the same weight as a clean one, and dropping it entirely
                throws away a usable if noisy observation.
        """
        pupil_xy = np.atleast_2d(np.asarray(pupil_xy, dtype=float))
        scene_xy = np.atleast_2d(np.asarray(scene_xy, dtype=float))

        if len(pupil_xy) != len(scene_xy):
            raise ValueError("pupil_xy and scene_xy must have the same length")
        if scene_xy.shape[1] != 2:
            raise ValueError(f"scene_xy must be (n, 2), got {scene_xy.shape}")
        if len(pupil_xy) < self.n_parameters:
            raise ValueError(
                f"need at least {self.n_parameters} samples for a degree-{self.degree} "
                f"fit, got {len(pupil_xy)}"
            )

        self._pupil_mean = pupil_xy.mean(axis=0)
        scale = pupil_xy.std(axis=0)
        self._pupil_scale = np.where(scale < 1e-12, 1.0, scale)

        design = _design_matrix(self._standardise(pupil_xy), self.degree)

        if weights is not None:
            weights = np.asarray(weights, dtype=float).ravel()
            if len(weights) != len(pupil_xy):
                raise ValueError("weights must have one entry per sample")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            root_w = np.sqrt(weights)[:, None]
            design = design * root_w
            scene_xy = scene_xy * root_w

        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0

        gram = design.T @ design + penalty
        self._coefficients = np.linalg.solve(gram, design.T @ scene_xy)
        return self

    def predict(self, pupil_xy: np.ndarray) -> np.ndarray:
        """Map pupil centres to scene-camera points. Returns (n, 2)."""
        if not self.fitted:
            raise RuntimeError("map is not fitted; call fit() first")
        return _design_matrix(self._standardise(pupil_xy), self.degree) @ self._coefficients

    def residuals_px(self, pupil_xy: np.ndarray, scene_xy: np.ndarray) -> np.ndarray:
        """Per-sample error in scene-camera pixels."""
        predicted = self.predict(pupil_xy)
        return np.linalg.norm(predicted - np.atleast_2d(scene_xy), axis=1)


def cross_validated_residuals(
    pupil_xy: np.ndarray,
    scene_xy: np.ndarray,
    *,
    degree: int = 2,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Leave-one-out residuals, in scene-camera pixels.

    A head-worn calibration has few points — a dozen or so, because the wearer
    has to hold each one — so leave-one-out is affordable and the in-sample
    residual is badly optimistic at that sample size. With 12 points and 6
    parameters, the in-sample fit has half its degrees of freedom left; quoting
    it as the tracker's accuracy is close to meaningless.
    """
    pupil_xy = np.atleast_2d(np.asarray(pupil_xy, dtype=float))
    scene_xy = np.atleast_2d(np.asarray(scene_xy, dtype=float))

    n = len(pupil_xy)
    residuals = np.full(n, np.nan)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        model = EyeToSceneMap(degree=degree, ridge=ridge)
        try:
            model.fit(pupil_xy[mask], scene_xy[mask])
        except ValueError:
            continue  # not enough points left to fit; leave this one NaN

        residuals[i] = float(np.linalg.norm(model.predict(pupil_xy[i]) - scene_xy[i]))

    return residuals


def slippage_indicator(
    reference_pupil_xy: np.ndarray,
    current_pupil_xy: np.ndarray,
) -> float:
    """How far the pupil distribution has shifted since calibration, in pixels.

    The failure mode specific to head-worn rigs: the glasses move on the face.
    Nothing about the signal looks wrong afterwards — the pupil is still found,
    the fit still returns a confident answer — but every gaze estimate now
    carries a constant offset.

    Comparing the centroid of recent pupil positions against the calibration
    centroid detects the common case, a bulk shift. It cannot detect rotation
    of the frame, and it will false-positive on someone who genuinely spent the
    last minute looking at one corner, so it is an indicator to surface to the
    wearer, not a trigger for anything automatic.
    """
    reference = np.atleast_2d(np.asarray(reference_pupil_xy, dtype=float))
    current = np.atleast_2d(np.asarray(current_pupil_xy, dtype=float))

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("both point sets must be non-empty")

    return float(np.linalg.norm(current.mean(axis=0) - reference.mean(axis=0)))
