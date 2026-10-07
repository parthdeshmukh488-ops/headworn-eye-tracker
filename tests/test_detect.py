import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from headgaze.detect import detect_pupil  # noqa: E402
from headgaze.synthetic import synthetic_eye  # noqa: E402


@pytest.mark.parametrize("iris_level", [70, 95, 140])  # dark brown .. light blue
def test_centre_found_despite_glints_and_lashes(iris_level):
    rng = np.random.default_rng(iris_level)
    errors = []
    for _ in range(30):
        centre = (rng.uniform(110, 210), rng.uniform(90, 150))
        pupil = detect_pupil(synthetic_eye(centre, iris_level=iris_level, rng=rng))
        assert pupil.valid
        errors.append(np.linalg.norm(pupil.centre - centre))
    assert np.median(errors) < 0.5  # pixels
    assert np.max(errors) < 2.5


def test_radius_close_to_truth():
    pupil = detect_pupil(synthetic_eye((160, 120), pupil_radius=18.0))
    assert abs(np.mean(pupil.axes) - 18.0) < 1.5


def test_no_pupil_is_a_dropout_not_a_guess():
    rng = np.random.default_rng(0)
    blank = np.clip(180 + rng.normal(0, 4, (240, 320)), 0, 255).astype(np.uint8)
    assert not detect_pupil(blank).valid

    closed = np.full((240, 320), 175, np.uint8)
    cv2.line(closed, (40, 120), (280, 125), 40, 3)  # a closed lid is a line, not an ellipse
    assert not detect_pupil(closed).valid


def test_bgr_input_accepted():
    gray = synthetic_eye((150, 110))
    pupil = detect_pupil(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert pupil.valid and np.linalg.norm(pupil.centre - (150, 110)) < 1.0
