import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from headgaze.aruco import detect_markers, make_detector, render_screen  # noqa: E402
from headgaze.screen import ScreenLocator, ScreenMarkers, apply_homography  # noqa: E402

W, H = 1920, 1080


def view_of_screen(rng, screen_img):
    """A scene-camera view of the screen: scaled down, slightly rotated and tilted."""
    s, a = 0.30, np.radians(rng.normal(0, 4))
    m = np.array(
        [
            [s * np.cos(a), -s * np.sin(a), 30 + rng.normal(0, 8)],
            [s * np.sin(a), s * np.cos(a), 60 + rng.normal(0, 8)],
            [rng.normal(0, 2e-5), rng.normal(0, 2e-5), 1.0],
        ]
    )
    scene = cv2.warpPerspective(screen_img, m, (640, 480), borderValue=60)
    scene = np.clip(scene + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    return scene, m


def test_rendered_markers_sit_at_their_corners():
    markers = ScreenMarkers.for_screen(W, H)
    found = detect_markers(render_screen(markers, W, H), make_detector())
    assert sorted(found) == [0, 1, 2, 3]
    for marker_id, corners in markers.corners_px.items():
        assert np.abs(found[marker_id] - corners).max() < 1.0


def test_screen_located_through_a_tilted_camera():
    rng = np.random.default_rng(0)
    markers = ScreenMarkers.for_screen(W, H)
    screen_img = render_screen(markers, W, H)
    locator, detector = ScreenLocator(markers), make_detector()
    for _ in range(5):
        scene, screen_to_scene = view_of_screen(rng, screen_img)
        homography, error = locator.locate(detect_markers(scene, detector))
        assert homography is not None and error < 4.0
        truth = rng.uniform([200, 150], [1720, 930], (40, 2))
        recovered = apply_homography(homography, apply_homography(screen_to_scene, truth))
        assert np.linalg.norm(recovered - truth, axis=1).max() < 3.0  # screen pixels


def test_no_markers_in_view():
    blank = np.full((480, 640), 120, np.uint8)
    assert detect_markers(blank, make_detector()) == {}
