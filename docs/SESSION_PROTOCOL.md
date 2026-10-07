# First real session: parts, build, recording

Nothing in this repository has run on real hardware yet. This page is the plan
for the first session; fill in the results table at the end with what really
happens, including what fails.

## Parts (about 30–50 euro)

| Part | What to look for |
| --- | --- |
| Eye camera | USB endoscope camera with a 5.5–8 mm head that focuses at 2–5 cm; its LEDs must dim or switch off |
| Scene camera | Small USB webcam, 640×480 MJPG at 30 fps, field of view around 70° |
| Frame | `hardware/frame.scad` printed in PETG, or cheap safety glasses with printed clips |
| Markers | `python -m headgaze print-markers`, printed at 100 % on plain paper, four pieces of tape |
| Small parts | Zip ties, a USB extension cable, a kitchen scale for the weight |

## Safety

- **No infrared light near the eye.** This rig works in visible light on purpose.
- Turn the endoscope camera's LEDs off, or to the lowest setting if they cannot be switched off.
- Stop if the eye feels uncomfortable or dry.

## Set-up (about 20 minutes)

1. Print the markers and measure the black squares. If the printer scaled them, pass the
   measured size to `serve` with `--marker-mm`. Tape them upright
   on the screen corners: the black square's outer corner exactly on the corner of the
   visible screen (0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left).
2. Light from the front, no window behind the screen.
3. `python -m headgaze preview --eye 1 --scene 2`. Swap the indices if the views are
   swapped. Adjust the eye camera until the whole eye is in view with the pupil near the
   centre, and the status line shows the pupil found and 4/4 markers.

## Recording (about 30 minutes)

Use your measured eye-to-screen distance and screen width in every command
(`--distance-cm`, `--screen-width-cm`).

1. `python -m headgaze calibrate`: keep the head still, follow the dots with the eyes.
2. `python -m headgaze validate --note "head still"`.
3. `python -m headgaze validate --note "head moving"`: follow the dots while moving the head
   gently.
4. **Slippage:** take the glasses off and put them back on, then
   `python -m headgaze validate --note "re-seated 1"`, without recalibrating. Repeat five times.
5. **Comparison:** in the same light, run the webcam tracker's validation from
   `gaze-robot-camera` (its RUN_ME_FIRST.md, steps 2–3).
6. Weigh the headset and take a photo of it.
7. **Robot demo:** `python -m headgaze serve --markers paper` here, and
   `python -m app free --gaze udp --fullscreen` in `gaze-robot-camera`.

## Results (fill in only with recorded numbers)

| Condition | Head-worn accuracy (°) | Webcam accuracy (°) | Head-worn data loss (%) |
| --- | --- | --- | --- |
| Head still | [TBD] | [TBD] | [TBD] |
| Head moving | [TBD] | [TBD] | [TBD] |
| After re-seating (mean of 5) | [TBD] | not applicable | [TBD] |

Headset weight: [TBD] g. What failed first, and why: [TBD].

The expected story, to check rather than assume: the webcam tracker gets worse when
the head moves; the head-worn one should not, but it gets worse after re-seating,
because its calibration assumes the camera sits where it sat during calibration.
