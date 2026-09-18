"""Locating the screen in the scene camera, and mapping gaze onto it.

A head-worn tracker gives you gaze in the *scene camera's* frame. That is not
useful on its own: the wearer's head moves, so a fixed point on the screen
lands somewhere different in every frame. Something has to tie the scene image
back to the screen.

Four ArUco markers, one near each screen corner, do that. Detecting them gives
four image-to-screen point correspondences, which determine a homography — and
a homography is exactly the right model here, because a flat screen viewed by a
pinhole camera is a plane-to-plane projective map. No approximation is being
made in choosing it.

**Why the homography is recomputed every frame** rather than once at setup: the
whole point of a head-worn rig is that the head moves. A homography from one
head pose is wrong for every other one. Recomputing is cheap; caching it is the
bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScreenMarkers:
    """Where each marker's four corners sit on the screen, in screen pixels.

    **Every corner is kept, not just the marker centre.** That is not a detail.
    A homography has eight degrees of freedom, so four point pairs determine it
    exactly — and an exact fit has zero residual whatever the points are, which
    means reprojection error cannot detect a marker that was mislabelled or
    detected in the wrong place. Using all four corners of each marker gives
    sixteen correspondences against eight unknowns, and the residual finally
    carries information.

    ArUco detectors return the four corners anyway, so this costs nothing.

    Corner order within each marker is the detector's own: top-left,
    top-right, bottom-right, bottom-left, clockwise in the marker's frame.
    """

    corners_px: dict  # marker id -> (4, 2) screen-pixel corners

    def __post_init__(self) -> None:
        if len(self.corners_px) < 2:
            raise ValueError(
                "need at least 2 markers; one marker's four corners are coplanar "
                "and exactly determine the homography, leaving nothing to check"
            )
        for marker_id, corners in self.corners_px.items():
            array = np.asarray(corners, dtype=float)
            if array.shape != (4, 2):
                raise ValueError(
                    f"marker {marker_id} corners must be (4, 2), got {array.shape}"
                )

    @property
    def ids(self) -> tuple:
        return tuple(self.corners_px)

    @classmethod
    def for_screen(
        cls,
        width_px: int,
        height_px: int,
        *,
        ids: tuple[int, int, int, int] = (0, 1, 2, 3),
        marker_size_px: float = 120.0,
        inset_px: float = 20.0,
    ) -> ScreenMarkers:
        """Four square markers, one tucked into each screen corner."""
        if len(set(ids)) != 4:
            raise ValueError("the four marker ids must be distinct")
        if marker_size_px <= 0:
            raise ValueError("marker_size_px must be positive")

        s, i = marker_size_px, inset_px
        origins = [
            (i, i),  # top-left
            (width_px - i - s, i),  # top-right
            (width_px - i - s, height_px - i - s),  # bottom-right
            (i, height_px - i - s),  # bottom-left
        ]

        corners = {}
        for marker_id, (ox, oy) in zip(ids, origins, strict=True):
            corners[marker_id] = np.array(
                [[ox, oy], [ox + s, oy], [ox + s, oy + s], [ox, oy + s]], dtype=float
            )
        return cls(corners_px=corners)


def homography_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Solve for the 3x3 homography taking source points to destination points.

    Direct linear transform, with the normalisation step that textbooks list as
    optional and is not. Raw pixel coordinates in the hundreds give a design
    matrix whose singular values span many orders of magnitude, and the
    recovered homography is visibly wrong at the image corners. Normalising
    each point set to zero mean and mean distance sqrt(2) fixes it.
    """
    source = np.asarray(source, dtype=float)
    destination = np.asarray(destination, dtype=float)

    if source.shape != destination.shape:
        raise ValueError("source and destination must have the same shape")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"points must be (n, 2), got {source.shape}")
    if len(source) < 4:
        raise ValueError(f"need at least 4 correspondences, got {len(source)}")

    src_norm, src_transform = _normalise(source)
    dst_norm, dst_transform = _normalise(destination)

    rows = []
    for (sx, sy), (dx, dy) in zip(src_norm, dst_norm, strict=True):
        rows.append([-sx, -sy, -1.0, 0.0, 0.0, 0.0, dx * sx, dx * sy, dx])
        rows.append([0.0, 0.0, 0.0, -sx, -sy, -1.0, dy * sx, dy * sy, dy])

    _, _, vt = np.linalg.svd(np.array(rows))
    h_norm = vt[-1].reshape(3, 3)

    homography = np.linalg.inv(dst_transform) @ h_norm @ src_transform

    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("degenerate homography; are the points collinear?")

    return homography / homography[2, 2]


def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to zero mean and scale to mean distance sqrt(2)."""
    centroid = points.mean(axis=0)
    centred = points - centroid

    mean_distance = float(np.mean(np.linalg.norm(centred, axis=1)))
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0

    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return centred * scale, transform


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map points through a homography. Returns (n, 2)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    homogeneous = np.hstack([points, np.ones((len(points), 1))])

    projected = homogeneous @ homography.T
    w = projected[:, 2:3]

    # A point on the horizon maps to infinity. Rather than returning inf and
    # letting it propagate silently into a gaze estimate, it becomes NaN, which
    # the validity checks downstream already know how to reject.
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return projected[:, :2] / w


def reprojection_error(
    homography: np.ndarray, source: np.ndarray, destination: np.ndarray
) -> float:
    """RMS distance between mapped source points and their destinations.

    Worth checking every frame. A marker detected at the wrong corner, or a
    spurious detection, produces a homography that still solves cleanly and
    maps gaze to a confidently wrong place. A high reprojection error is the
    only cheap signal that this has happened.
    """
    mapped = apply_homography(homography, source)
    destination = np.atleast_2d(np.asarray(destination, dtype=float))

    finite = np.all(np.isfinite(mapped), axis=1)
    if not np.any(finite):
        return float("inf")

    residuals = np.linalg.norm(mapped[finite] - destination[finite], axis=1)
    return float(np.sqrt(np.mean(residuals**2)))


@dataclass
class ScreenLocator:
    """Turns per-frame marker detections into a scene-to-screen homography.

    Args:
        markers: where the markers are on the screen.
        max_reprojection_px: above this the frame is rejected rather than
            producing a plausible-looking wrong answer.
    """

    markers: ScreenMarkers
    max_reprojection_px: float = 8.0
    min_markers: int = 2

    def locate(self, detected: dict) -> tuple[np.ndarray | None, float]:
        """Compute the homography from a frame's detections.

        Args:
            detected: marker id to its (4, 2) corners in scene-image pixels,
                in the detector's own corner order. Ids not in
                :attr:`markers` are ignored, which is what lets other ArUco
                markers exist in the room without breaking anything.

        Returns:
            The scene-to-screen homography and its reprojection error in screen
            pixels, or ``(None, inf)`` when the frame cannot be trusted.

        Two markers is the working minimum. It gives eight correspondences
        against eight unknowns — still an exact fit, so the residual is
        uninformative — but it keeps working when the wearer turns far enough
        that two markers leave the scene camera's field of view, which happens
        constantly in practice. Three or more is where the residual starts
        doing its job.
        """
        source, destination = [], []
        for marker_id, screen_corners in self.markers.corners_px.items():
            if marker_id not in detected:
                continue

            observed = np.asarray(detected[marker_id], dtype=float)
            if observed.shape != (4, 2):
                raise ValueError(
                    f"marker {marker_id} detection must be (4, 2), got {observed.shape}"
                )

            source.append(observed)
            destination.append(np.asarray(screen_corners, dtype=float))

        if len(source) < self.min_markers:
            return None, float("inf")

        source_points = np.vstack(source)
        destination_points = np.vstack(destination)

        try:
            homography = homography_from_points(source_points, destination_points)
        except (ValueError, np.linalg.LinAlgError):
            return None, float("inf")

        error = reprojection_error(homography, source_points, destination_points)
        if not np.isfinite(error) or error > self.max_reprojection_px:
            return None, error

        return homography, error
