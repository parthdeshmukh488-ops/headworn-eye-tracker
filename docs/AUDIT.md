# Audit, 7 Oct 2026

What was checked before extending the repository, and what changed.

## Reproduced

- All 58 existing tests pass on Windows 10, Python 3.13.
- `examples/01_end_to_end.py` prints exactly the numbers quoted in the README.
- `ruff check` and `ruff format --check` pass.

## Gaps found

1. **No image-to-pupil step.** `pupil.py` had the pieces (glint suppression,
   threshold estimate, robust ellipse fit, confidence), but nothing turned an eye
   image into a `PupilEstimate`. The README's "detection pipeline" existed only for
   synthetic point sets. Added `detect.py`.
2. **No camera, ArUco or live code.** Nothing read cameras, detected markers in an
   image or ran a calibration with a person. Added `aruco.py`, `live.py`,
   `session.py` and the `python -m headgaze` tools.
3. **No connection to the robot-camera project.** Added `stream.py`, which sends the
   same UDP messages as `gaze-robot-camera`.
4. **Wording that read as hardware experience.** The status line ("personal project,
   rebuilt and published") and the PETG paragraph read as if the frame had been
   printed and worn. Neither has happened yet; both now say so.

## Notes

- Example 01 converts calibration residuals from scene-camera pixels to degrees with
  the screen's pixel size. That is acceptable there because its simulated scene
  camera sees the screen at close to 1:1 scale; it would not be with a real camera.
  `session.py` reports calibration error in scene pixels and leaves degrees to
  `validate`, which measures them on the screen.
- The first version of `detect.py` used one fixed dark fraction and, on synthetic
  eyes, often fitted the iris instead of the pupil (right centre, because the two
  are concentric, but the wrong size). The tests caught it; the seed-then-midpoint
  threshold replaced it.

## Added tests

- `test_detect.py`: pupil centre under glints and lashes for three iris colours; no
  pupil reported for a blank frame or a closed eye.
- `test_aruco.py`: markers drawn where the layout says; the screen located through a
  tilted, noisy scene camera to within 3 screen pixels.
- `test_session.py`: calibration and validation from synthetic images with the head
  moving on every frame.
- `test_serve_replay.py`: `serve` on replayed videos sends correct UDP messages.
- `test_stream_metrics_io.py`: message format, filter, visual-angle maths, save/load.
