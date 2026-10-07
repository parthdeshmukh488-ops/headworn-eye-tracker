"""Gaze over local UDP, in the message format of the gaze-robot-camera project.

One JSON object per datagram, so the robot-camera app can take this headset as
its gaze source (`python -m app free --gaze udp`) without knowing it is head-worn:

    {"t": 1791239046.12, "x": 0.51, "y": 0.43, "valid": true,
     "raw_x": 0.53, "raw_y": 0.41, "seq": 1234, "filter": "one_euro"}

x/y are filtered and raw_x/raw_y unfiltered, in normalised screen coordinates
(0..1, origin top-left); all four are null when `valid` is false.
"""

from __future__ import annotations

import json
import math
import socket

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005


def encode_sample(t, xy, raw_xy, seq: int, filter_name: str) -> bytes:
    """xy / raw_xy: normalised screen positions, or None for an invalid sample."""
    valid = xy is not None and raw_xy is not None
    message = {
        "t": float(t),
        "x": float(xy[0]) if valid else None,
        "y": float(xy[1]) if valid else None,
        "valid": valid,
        "raw_x": float(raw_xy[0]) if valid else None,
        "raw_y": float(raw_xy[1]) if valid else None,
        "seq": int(seq),
        "filter": filter_name,
    }
    return json.dumps(message).encode()


class GazeSender:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, t, xy, raw_xy, seq: int, filter_name: str = "one_euro") -> None:
        self.sock.sendto(encode_sample(t, xy, raw_xy, seq, filter_name), self.addr)

    def close(self) -> None:
        self.sock.close()


class OneEuroFilter:
    """One Euro filter (Casiez, Roussel and Vogel, CHI 2012) for a 2D point.

    An exponential low-pass whose cutoff rises with speed: heavy smoothing while
    the eye rests (jitter is noise), little smoothing during a saccade (lag would
    make the cursor trail the eye). Units of `beta` follow the input units.
    """

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.003, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._x = [0.0, 0.0]
        self._dx = [0.0, 0.0]

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        if self._t is None:
            self._t, self._x = t, [float(x), float(y)]
            return self._x[0], self._x[1]
        dt = t - self._t
        if dt <= 0:  # duplicate timestamp: keep the previous estimate
            return self._x[0], self._x[1]
        for i, value in enumerate((x, y)):
            speed = (value - self._x[i]) / dt
            self._dx[i] += self._alpha(self.d_cutoff, dt) * (speed - self._dx[i])
            cutoff = self.min_cutoff + self.beta * abs(self._dx[i])
            self._x[i] += self._alpha(cutoff, dt) * (value - self._x[i])
        self._t = t
        return self._x[0], self._x[1]
