"""ArUco markers: drawing them on the screen and finding them in the scene camera.

Needs OpenCV (the `camera` extra). The corner order OpenCV returns (top-left,
top-right, bottom-right, bottom-left, in the marker's own frame) is the order
:class:`headgaze.screen.ScreenMarkers` uses, so detections go straight into
:class:`headgaze.screen.ScreenLocator`.
"""

from __future__ import annotations

import cv2
import numpy as np

from headgaze.screen import ScreenMarkers

DICTIONARY = cv2.aruco.DICT_4X4_50


def make_detector(dictionary: int = DICTIONARY) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corners: the homography is solved from these, so their noise is gaze noise.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), params)


def detect_markers(image: np.ndarray, detector: cv2.aruco.ArucoDetector) -> dict:
    """Marker id -> (4, 2) corners in image pixels."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(i): c.reshape(4, 2).astype(float) for c, i in zip(corners, ids.ravel(), strict=True)
    }


def render_screen(
    markers: ScreenMarkers,
    width_px: int,
    height_px: int,
    *,
    dictionary: int = DICTIONARY,
    background: int = 235,
) -> np.ndarray:
    """A greyscale screen image with every marker drawn exactly at its listed corners.

    The background must stay light: a marker is found by its black border, and it
    needs a light margin around that border.
    """
    screen = np.full((height_px, width_px), background, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    for marker_id, corners in markers.corners_px.items():
        x0, y0 = (int(round(v)) for v in corners[0])
        size = int(round(corners[1, 0] - corners[0, 0]))
        screen[y0 : y0 + size, x0 : x0 + size] = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, size
        )
    return screen
