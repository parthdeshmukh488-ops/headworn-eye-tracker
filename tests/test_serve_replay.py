"""`python -m headgaze serve` on replayed videos: gaze arrives over UDP in the robot-camera format."""

import json
import socket

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from headgaze.__main__ import main  # noqa: E402
from headgaze.aruco import render_screen  # noqa: E402
from headgaze.mapping import EyeToSceneMap  # noqa: E402
from headgaze.screen import ScreenMarkers, apply_homography  # noqa: E402
from headgaze.synthetic import synthetic_eye  # noqa: E402

W, H = 1920, 1080
SCREEN_WIDTH_CM, MARKER_MM = 34.5, 30.0


def write_video(path, frames):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f)
    writer.release()


def test_serve_streams_gaze_from_recordings(tmp_path):
    rng = np.random.default_rng(3)
    # Paper markers: 30 mm black squares taped on the screen corners.
    side_px = MARKER_MM / (SCREEN_WIDTH_CM * 10 / W)
    markers = ScreenMarkers.for_screen(W, H, marker_size_px=side_px, inset_px=0.0)
    screen_to_scene = np.array([[0.3, 0, 32], [0, 0.3, 78], [0, 0, 1.0]])
    scene = cv2.warpPerspective(
        render_screen(markers, W, H), screen_to_scene, (640, 480), borderValue=60
    )

    # A calibration in which the scene point is a known linear function of the pupil.
    pupils = rng.uniform([100, 80], [220, 160], (30, 2))
    model = EyeToSceneMap().fit(pupils, (pupils - [160, 120]) * 5 + [320, 240])
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"map": model.to_dict()}))

    pupil_track = [(140 + 2 * i, 120) for i in range(20)]
    write_video(tmp_path / "eye.avi", [synthetic_eye(c, rng=rng) for c in pupil_track])
    write_video(tmp_path / "scene.avi", [scene] * len(pupil_track))

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.5)
    main(
        [
            "serve",
            "--eye",
            str(tmp_path / "eye.avi"),
            "--scene",
            str(tmp_path / "scene.avi"),
            "--calibration",
            str(calibration),
            "--port",
            str(receiver.getsockname()[1]),
            "--markers",
            "paper",
            "--marker-mm",
            str(MARKER_MM),
            "--screen-width-cm",
            str(SCREEN_WIDTH_CM),
        ]
    )

    messages = []
    try:
        while True:
            messages.append(json.loads(receiver.recv(4096)))
    except TimeoutError:
        pass
    receiver.close()

    assert len(messages) == len(pupil_track)
    valid = [m for m in messages if m["valid"]]
    assert len(valid) >= len(pupil_track) - 2  # MJPG compression may cost a frame or two
    # Expected raw gaze for the last frame: pupil -> scene (the linear map) -> screen.
    last_scene = (np.array(pupil_track[-1]) - [160, 120]) * 5 + [320, 240]
    expected = apply_homography(np.linalg.inv(screen_to_scene), last_scene)[0] / [W, H]
    assert np.allclose([valid[-1]["raw_x"], valid[-1]["raw_y"]], expected, atol=0.01)
