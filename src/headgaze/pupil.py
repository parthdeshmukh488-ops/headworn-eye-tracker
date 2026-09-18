"""Finding the pupil in a visible-light eye image.

Almost every head-worn eye tracker uses infrared illumination, for a good
reason: under IR the pupil is the darkest thing in the image by a wide margin,
the iris is bright, and the boundary between them is trivially thresholdable.
This rig deliberately does not, because it puts no emitter next to the eye.

That choice costs accuracy and it is worth being explicit about why:

- **Under visible light the pupil-iris contrast depends on iris colour.** A
  dark brown iris gives a boundary a threshold can barely find. A blue iris
  gives an easy one. An IR tracker does not care.
- **Specular reflections land on the cornea** from whatever is in the room.
  Under IR there is one known glint from a known emitter, which is useful. Here
  there are several, in unknown places, and they are contaminants.
- **Eyelashes cross the pupil** and are dark, so they survive thresholding and
  distort a naive fit.

The pipeline below is built around those three problems in that order, and the
ellipse fit is deliberately robust rather than least-squares, because a
least-squares fit is pulled badly by exactly the contaminants that survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PupilEstimate:
    """A fitted pupil, in eye-camera pixel coordinates."""

    centre: np.ndarray  # (2,)
    axes: np.ndarray  # (2,) semi-axis lengths, major first
    angle_rad: float
    confidence: float  # 0..1
    valid: bool

    @property
    def area_px(self) -> float:
        return float(np.pi * self.axes[0] * self.axes[1])

    @property
    def aspect_ratio(self) -> float:
        """Minor over major. Near 1 is circular; low means a steep view angle."""
        if self.axes[0] < 1e-9:
            return 0.0
        return float(self.axes[1] / self.axes[0])


def suppress_glints(gray: np.ndarray, *, percentile: float = 99.0) -> np.ndarray:
    """Replace specular highlights with the local dark level.

    Glints are small, very bright and sit *inside* the pupil as often as not.
    Left alone they punch bright holes in the region a threshold is trying to
    find, and the ellipse fit then cuts a bite out of the pupil.

    Filling them with a dark value rather than interpolating is deliberate: a
    glint inside the pupil should read as pupil, and one on the iris is
    discarded by the threshold anyway.
    """
    gray = np.asarray(gray)
    if gray.size == 0:
        raise ValueError("empty image")

    cutoff = np.percentile(gray, percentile)
    dark_level = np.percentile(gray, 5.0)

    out = gray.astype(np.float32).copy()
    out[out >= cutoff] = dark_level
    return out


def estimate_threshold(gray: np.ndarray, *, pupil_fraction: float = 0.06) -> float:
    """Pick a threshold from the image's own darkest fraction.

    A fixed threshold cannot work here: exposure changes as the wearer turns
    towards or away from a window, and iris colour shifts the whole
    distribution between people. Taking a percentile of the image adapts to
    both without a per-user setting.

    ``pupil_fraction`` is the expected share of the eye crop the pupil covers.
    It is a geometry assumption about the rig, not about the person, which is
    why it can be a constant.
    """
    gray = np.asarray(gray)
    if not 0.0 < pupil_fraction < 1.0:
        raise ValueError("pupil_fraction must lie in (0, 1)")
    return float(np.percentile(gray, pupil_fraction * 100.0))


def fit_ellipse_robust(
    points: np.ndarray,
    *,
    iterations: int = 3,
    keep_fraction: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit an ellipse to boundary points, trimming the worst residuals.

    A plain least-squares conic fit is pulled hard by eyelash pixels and by any
    glint the suppression missed, and a single outlier can move the centre by
    several pixels — which at this rig's geometry is a degree of gaze error.

    The fix is cheap: fit, discard the worst-fitting fraction, refit. Three
    rounds is enough; more starts eating real boundary on a partly occluded
    pupil.

    Returns:
        centre, semi-axes (major first), rotation in radians.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (n, 2), got {points.shape}")
    if len(points) < 5:
        raise ValueError(f"need at least 5 points to fit an ellipse, got {len(points)}")

    kept = points
    centre = kept.mean(axis=0)
    axes = np.array([1.0, 1.0])
    angle = 0.0

    for _ in range(iterations):
        centre, axes, angle = _fit_ellipse_direct(kept)

        if len(kept) <= 5:
            break

        residuals = _algebraic_residual(kept, centre, axes, angle)
        cutoff = np.quantile(residuals, keep_fraction)
        trimmed = kept[residuals <= cutoff]

        if len(trimmed) < 5:
            break
        kept = trimmed

    return centre, axes, angle


def _fit_ellipse_direct(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares conic fit, then conic-to-geometric conversion.

    Points are centred and scaled before the fit. Without that the design
    matrix for pixel coordinates in the hundreds is badly conditioned and the
    recovered axes can come back complex.
    """
    mean = points.mean(axis=0)
    scale = float(np.sqrt((np.sum((points - mean) ** 2, axis=1)).mean()))
    scale = scale if scale > 1e-12 else 1.0
    p = (points - mean) / scale

    x, y = p[:, 0], p[:, 1]
    design = np.column_stack([x**2, x * y, y**2, x, y, np.ones_like(x)])

    # The conic is defined up to scale, so the fit is the null space of the
    # design matrix: the right singular vector with the smallest value.
    _, _, vt = np.linalg.svd(design, full_matrices=False)
    a, b, c, d, e, f = vt[-1]

    if b * b - 4 * a * c >= 0:
        # Not an ellipse (parabola or hyperbola). Fall back to the point
        # cloud's own moments, which is always defined and roughly right.
        return _moment_ellipse(points)

    # Matrix form of the conic. Going through the matrices rather than the
    # closed-form axis expressions is deliberate: the closed form has a
    # branch that has to pick which root belongs to which axis, and it picks
    # wrong for near-circular ellipses -- which is most pupils. The
    # eigendecomposition has no such branch.
    conic = np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])
    quadratic = conic[:2, :2]

    det_quadratic = float(np.linalg.det(quadratic))
    if abs(det_quadratic) < 1e-15:
        return _moment_ellipse(points)

    try:
        centre_local = np.linalg.solve(quadratic, -np.array([d / 2, e / 2]))
    except np.linalg.LinAlgError:
        return _moment_ellipse(points)

    # After translating to the centre the conic reads v' Q v' = -constant,
    # where the constant is det(conic) / det(quadratic).
    constant = float(np.linalg.det(conic)) / det_quadratic

    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    if np.any(np.abs(eigenvalues) < 1e-15):
        return _moment_ellipse(points)

    squared_axes = -constant / eigenvalues
    if np.any(squared_axes <= 0):
        return _moment_ellipse(points)

    semi_axes = np.sqrt(squared_axes)

    # Larger semi-axis first, with the angle taken from its own eigenvector.
    order = np.argsort(semi_axes)[::-1]
    semi_axes = semi_axes[order]
    major_direction = eigenvectors[:, order[0]]
    angle = float(np.arctan2(major_direction[1], major_direction[0]))

    centre = centre_local * scale + mean
    return centre, semi_axes * scale, angle


def _moment_ellipse(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Ellipse from the second moments of the point cloud."""
    centre = points.mean(axis=0)
    centred = points - centre
    cov = np.cov(centred.T)

    if not np.all(np.isfinite(cov)):
        return centre, np.array([1.0, 1.0]), 0.0

    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    axes = 2.0 * np.sqrt(np.maximum(values, 1e-12))
    angle = float(np.arctan2(vectors[1, 0], vectors[0, 0]))
    return centre, axes, angle


def _algebraic_residual(
    points: np.ndarray, centre: np.ndarray, axes: np.ndarray, angle: float
) -> np.ndarray:
    """How far each point sits from the unit ellipse, after un-rotating."""
    cos, sin = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    local = (points - centre) @ rotation.T

    safe = np.maximum(axes, 1e-9)
    return np.abs((local[:, 0] / safe[0]) ** 2 + (local[:, 1] / safe[1]) ** 2 - 1.0)


def confidence_from_fit(
    points: np.ndarray,
    centre: np.ndarray,
    axes: np.ndarray,
    angle: float,
    *,
    min_axis_px: float = 3.0,
    max_aspect: float = 4.0,
) -> float:
    """A 0-1 score for whether this fit should be believed.

    Three independent ways a fit goes wrong, each of which produces a
    confident-looking ellipse that is not a pupil:

    - the boundary points do not actually lie on it (high residual)
    - it is implausibly small, which usually means a shadow was fitted
    - it is implausibly elongated, which usually means eyelid edge was fitted

    Reporting one number rather than a hard reject lets the caller decide;
    downstream, a low-confidence sample is treated as a dropout.
    """
    if len(points) < 5:
        return 0.0

    residual = float(np.mean(_algebraic_residual(points, centre, axes, angle)))
    residual_score = float(np.exp(-residual * 4.0))

    size_score = 1.0 if axes[1] >= min_axis_px else float(axes[1] / min_axis_px)

    aspect = axes[0] / max(axes[1], 1e-9)
    aspect_score = 1.0 if aspect <= max_aspect else float(max_aspect / aspect)

    return float(np.clip(residual_score * size_score * aspect_score, 0.0, 1.0))
