"""Synthetic eye images with a known pupil, for tests and examples. Needs OpenCV.

The image degrades in the three ways the visible-light pipeline is built
around: a corneal glint on the pupil edge, eyelashes crossing the pupil, and a
pupil-iris contrast set by the iris colour. It is a test of the code, not a
stand-in for a real eye: real skin, lids and lashes are messier.
"""

from __future__ import annotations

import cv2
import numpy as np


def synthetic_eye(
    centre: tuple[float, float],
    *,
    size: tuple[int, int] = (240, 320),
    pupil_radius: float = 18.0,
    iris_radius: float = 48.0,
    iris_level: int = 95,
    pupil_level: int = 25,
    glints: int = 2,
    lashes: int = 6,
    noise: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greyscale uint8 eye image whose true pupil centre is `centre` (x, y) in pixels.

    ``iris_level`` sets the colour: about 70 is dark brown, 140 light blue.
    Drawing uses 4-bit sub-pixel shifts, so the centre is not snapped to a pixel.
    """
    rng = rng or np.random.default_rng(0)
    h, w = size
    shift = 4
    scale = 1 << shift

    def fixed(p):
        return tuple(int(round(v * scale)) for v in p)

    image = np.full((h, w), 175, dtype=np.uint8)  # skin
    cv2.ellipse(
        image,
        fixed((w / 2, h / 2)),
        fixed((w * 0.45, h * 0.32)),
        0,
        0,
        360,
        225,
        -1,
        cv2.LINE_AA,
        shift,
    )  # sclera inside the eye opening
    cv2.circle(
        image, fixed(centre), int(iris_radius * scale), iris_level, -1, cv2.LINE_AA, shift
    )
    cv2.circle(
        image, fixed(centre), int(pupil_radius * scale), pupil_level, -1, cv2.LINE_AA, shift
    )

    for _ in range(glints):  # specular highlights, often on the pupil edge
        angle = rng.uniform(0, 2 * np.pi)
        r = pupil_radius * rng.uniform(0.6, 1.0)
        spot = (centre[0] + r * np.cos(angle), centre[1] + r * np.sin(angle))
        cv2.circle(
            image, fixed(spot), int(rng.uniform(2.0, 3.5) * scale), 255, -1, cv2.LINE_AA, shift
        )

    for _ in range(lashes):  # dark curved strokes from the upper lid
        x0 = centre[0] + rng.uniform(-1.5, 1.5) * pupil_radius
        top = (x0, centre[1] - 2.2 * pupil_radius)
        end = (x0 + rng.uniform(-8, 8), centre[1] - rng.uniform(0.2, 0.9) * pupil_radius)
        cv2.line(image, fixed(top), fixed(end), 40, 1, cv2.LINE_AA, shift)

    noisy = image.astype(np.float32) + rng.normal(0.0, noise, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
