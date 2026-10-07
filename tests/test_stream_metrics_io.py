import json

import numpy as np

from headgaze.mapping import EyeToSceneMap
from headgaze.metrics import Screen, accuracy_deg, data_loss_pct, precision_rms_s2s_deg
from headgaze.stream import OneEuroFilter, encode_sample


def test_message_matches_the_robot_camera_protocol():
    message = json.loads(encode_sample(12.5, (0.4, 0.6), (0.41, 0.59), 7, "one_euro"))
    assert set(message) == {"t", "x", "y", "valid", "raw_x", "raw_y", "seq", "filter"}
    assert message["valid"] and message["x"] == 0.4 and message["raw_y"] == 0.59

    invalid = json.loads(encode_sample(13.0, None, None, 8, "one_euro"))
    assert invalid["valid"] is False and invalid["x"] is None and invalid["raw_x"] is None


def test_one_euro_smooths_rest_and_follows_jumps():
    rng = np.random.default_rng(0)
    filt = OneEuroFilter()
    out = [filt(i / 30, 500 + rng.normal(0, 10), 300) for i in range(60)]
    assert np.std([o[0] for o in out[20:]]) < 4  # jitter of 10 px reduced
    for i in range(60, 75):  # a saccade of 400 px
        x, _ = filt(i / 30, 900, 300)
    assert x > 850  # arrived within half a second


def test_visual_angle_is_exact_not_small_angle():
    screen = Screen(1920, 1080, 34.5, 60.0)
    centre = np.array([960.0, 540.0])
    one_cm_right = centre + [1920 / 34.5, 0]
    assert np.isclose(screen.angle_deg(centre, one_cm_right)[0], np.degrees(np.arctan(1 / 60)))


def test_quality_metrics():
    screen = Screen()
    target = np.array([960.0, 540.0])
    gaze = np.array([[1018.0, 540.0]] * 10 + [[np.nan, np.nan]] * 2)
    assert 0.9 < accuracy_deg(screen, gaze, target) < 1.1  # 58 px is about 1 deg at 60 cm
    assert np.isclose(precision_rms_s2s_deg(screen, gaze), 0.0, atol=1e-5)
    assert np.isclose(data_loss_pct(np.isfinite(gaze[:, 0])), 100 * 2 / 12)


def test_map_survives_save_and_load():
    rng = np.random.default_rng(1)
    pupil = rng.uniform(100, 200, (20, 2))
    scene = pupil * 2.5 + 0.01 * pupil**2
    model = EyeToSceneMap().fit(pupil, scene)
    loaded = EyeToSceneMap.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.allclose(loaded.predict(pupil), model.predict(pupil))
