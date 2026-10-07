# Head-worn eye tracker on a 3D-printed frame

A pair of printed glasses carrying an eye camera and a scene camera, tracking
the iris in **visible light** — no infrared emitter next to the eye — and
mapping gaze onto a screen using ArUco markers.

The frame is parametric OpenSCAD rather than a dropped STL, so the numbers that
decide whether it fits a given face and a given camera are editable and
readable.

**Status:** software only, so far. The frame has not been printed and no real
eye has been recorded yet; every number on this page comes from synthetic data.
The live tools for a real rig are in place (see *From images to gaze, live*).
Python 3.10+, numpy only for the core, OpenCV for everything camera-side.

```bash
pip install -e ".[dev,camera]"
pytest                             # 74 tests, no hardware
python examples/01_end_to_end.py   # the geometry, from pupil positions
python examples/02_image_pipeline.py   # the same chain, from images
python examples/make_figures.py    # redraws docs/figures/
```

---

## Why visible light, and what it costs

Almost every head-worn tracker uses IR illumination, and for a good reason:
under IR the pupil is by far the darkest thing in the image and the boundary is
trivially thresholdable. This rig puts no emitter next to the eye, which costs:

- **Pupil-iris contrast now depends on iris colour.** A dark brown iris gives a
  boundary a threshold can barely find; a blue one is easy. IR does not care.
- **Specular reflections land on the cornea** from whatever is in the room —
  several of them, in unknown places, rather than one known glint.
- **Eyelashes cross the pupil**, are dark, and survive thresholding.

The detection pipeline is built around those three in that order. The ellipse
fit is **robust rather than least-squares**, because least-squares is pulled
badly by exactly the contaminants that survive:

![Trimmed fit against plain least squares](docs/figures/01_pupil_fit.png)

A handful of eyelash pixels drags the plain conic fit off centre. Trimming the
worst-fitting fraction and refitting costs three extra iterations and recovers
the centre to a tenth of a pixel.

---

## Two stages, kept separate

This is the design decision the whole thing rests on.

| | Eye → scene | Scene → screen |
|---|---|---|
| Kind | **fitted**, from calibration | **geometric**, exact |
| Model | degree-2 polynomial | homography |
| Depends on head pose | no | yes |
| Recomputed | once, at calibration | **every frame** |

A flat screen viewed by a pinhole camera is a plane-to-plane projective map, so
a homography is not an approximation here — it is the exact model. And because
head pose lives entirely in that stage, the fitted stage never has to account
for where the head is.

![Scene camera to screen, through the homography](docs/figures/02_screen_mapping.png)

Caching the homography is the bug. The whole point of a head-worn rig is that
the head moves; a homography from one head pose is wrong for every other one.

---

## The over-determination problem

A homography has eight degrees of freedom, and each point correspondence
contributes two equations. So **four points — the four marker centres — fit
exactly, including nonsense ones.** The residual is zero whatever you feed it,
and it cannot tell you a marker was mislabelled or detected in the wrong place.

ArUco detectors return four corners per marker anyway, so this repo keeps all
of them. Four markers become sixteen points, thirty-two equations against eight
unknowns, and reprojection error finally carries information:

![Centres are blind, corners are not](docs/figures/03_overdetermination.png)

There are tests for a displaced marker and for two swapped markers, and both
are caught. There is also a test asserting that four correspondences fit
anything exactly, so the reason for the design cannot quietly be forgotten.

Two markers is the working minimum — the wearer turning their head takes
markers out of frame constantly — and at two the system is already
over-determined, so the check still works.

---

## Numbers

`python examples/01_end_to_end.py`, synthetic wearer, head pose different on
every test frame:

```
Calibration: 12 points, 6 parameters
  in-sample residual        6.41 px   (0.17 deg)
  leave-one-out            13.48 px   (0.36 deg)
  overstatement          2.1x if in-sample is quoted

Test: 200 gaze points, head pose different on every frame
  frames rejected        0 (0.0%)
  accuracy, mean         2.86 deg
  accuracy, median       2.54 deg
  accuracy, p95          5.98 deg
```

![In-sample against leave-one-out, per calibration point](docs/figures/04_calibration_honesty.png)

Three things in that output worth reading carefully:

1. **The in-sample residual overstates by 2.1×.** Twelve calibration points
   against six parameters leaves half the degrees of freedom. At that ratio the
   in-sample number is not an accuracy estimate, and a repo that quoted 0.17°
   here would be claiming commercial-tracker performance from a printed frame.
2. **The test accuracy is worse again**, at 2.86°. The test points are spread
   across the whole screen, including outside the calibration grid, and the
   head poses are stronger. This is the number that describes the rig.
3. **The map was fitted once** and never refitted, across 200 different head
   poses, with nothing rejected. That is the two-stage separation working.

---

## From images to gaze, live

Example 01 starts from pupil *positions*. A real rig starts from *images*, so
`detect.py` turns an eye frame into a pupil and `aruco.py` turns a scene frame
into marker corners.

**Finding the pupil.** A single "darkest x %" threshold does not work: how much
of the frame the pupil covers depends on the rig, the person and the light. A
low threshold keeps only the pupil's dark core; the next one up already merges
lashes and iris. So the detector tries several dark fractions, fits an ellipse
to each plausible blob, keeps the darkest blob that fits and stands out from
its surround as a **seed**, then thresholds again **halfway between the seed's
grey level and its surround's**, which cuts at the pupil's actual edge. A blob
that is not clearly darker than the ring around it is never accepted, so a
blink or a blank frame is a dropout, not a guess.

`python examples/02_image_pipeline.py`, synthetic eyes with glints and lashes,
and a rendered marker screen seen through a tilted scene camera:

```
Pupil detection, 200 synthetic eyes per iris colour (glints and lashes on)
  dark brown  centre error median 0.22 px, p95 0.96 px, dropouts 0/200
  brown       centre error median 0.25 px, p95 1.17 px, dropouts 0/200
  light blue  centre error median 0.20 px, p95 0.92 px, dropouts 0/200

Full chain from images, head pose different on every frame
  calibration, leave-one-dot-out   2.1 scene px (max 4.3)
  validation accuracy              0.08 deg
  validation precision (RMS-S2S)   0.21 deg
  data loss                        0.0 %
```

These numbers say the code is right, not that a printed rig reaches 0.08°.
Synthetic eyes have clean edges and no lids; a real eye under room light will
be much worse, and only a recorded session can say by how much.

**The live tools** (`python -m headgaze ...`, eye and scene camera given by index
or as a recorded video):

| Command | What it does |
|---|---|
| `print-markers` | A 300 dpi page with the four markers at real size, to tape on the screen corners |
| `preview` | Fullscreen marker screen with both camera views: is the pupil found, are the markers seen? |
| `calibrate` | Nine dots; fits the eye-to-scene map, reports leave-one-dot-out error, saves `data/calibration.json` |
| `validate` | Nine *other* dots; accuracy, precision and data loss in degrees, saved to `results/` |
| `serve` | Streams gaze over UDP in the format of [gaze-robot-camera](https://github.com/parthdeshmukh488-ops/gaze-robot-camera-mujoco), so the headset can steer the robot camera |

The session protocol for a first real recording, including the slippage test, is
in [docs/SESSION_PROTOCOL.md](docs/SESSION_PROTOCOL.md).

---

## The frame

`hardware/frame.scad` — parametric, three printable parts.

```bash
openscad -o frame.stl hardware/frame.scad
openscad -o eye_arm.stl -D 'part="eye_arm"' hardware/frame.scad
```

The parameters that actually matter are pupillary distance (get it wrong and
the eye camera looks at a cheekbone), and the eye camera's drop, reach and
pitch. One eye camera, not two: gaze is conjugate for anything beyond arm's
length, so a second doubles the weight, cabling and synchronisation for very
little.

**Prefer PETG to PLA.** PETG tolerates heat better (a warm room, a sunny desk)
and creeps less under a constant load. That matters for the eye arm: if it
sags even slightly, gaze drifts slowly, which looks exactly like calibration
decay and is not. This is a design choice from material properties; the frame
has not yet been printed or worn.

---

## What this does not do

- **Parallax is not corrected.** The eye camera sees the eye, not what the eye
  is looking at, so the fitted map is exact only at the depth it was calibrated
  at. Calibrate at a 60 cm screen, look at something 3 m away, and there is a
  systematic offset. Fixing it needs a stereo scene camera or a depth estimate;
  this rig has neither.
- **Slippage is detected, not corrected.** `slippage_indicator` compares the
  recent pupil centroid against the calibration centroid, which catches a bulk
  shift of the glasses on the face. It cannot see frame rotation, and it will
  false-positive on someone who genuinely spent a minute looking at one corner.
  It is something to show the wearer, not to trigger anything automatic.
- **No blink detection beyond the confidence score.** A fit that looks like an
  eyelid edge scores low and is treated as a dropout; there is no dedicated
  blink classifier.
- **Detection is not benchmarked on real eye images yet.** The tests use
  synthetic eye images degraded in the three ways described above. That
  validates the pipeline, not its behaviour on real skin, lids and lashes.
- **Nothing has run on hardware yet.** The camera code is tested on replayed
  synthetic videos, not on two real USB cameras.
- **Monocular.** No vergence, so no depth of gaze.

---

## Layout

```
src/headgaze/
  pupil.py      visible-light pupil detection pieces and robust ellipse fitting
  screen.py     ArUco marker layout, homography, per-frame screen locator
  mapping.py    fitted eye-to-scene map, cross-validation, slippage, save/load
  metrics.py    accuracy, precision and data loss in degrees of visual angle
  stream.py     gaze over UDP, One Euro filter
  detect.py     eye image -> pupil (OpenCV)
  aruco.py      drawing and detecting the screen markers (OpenCV)
  session.py    one frame pair -> one gaze sample; calibration and validation
  live.py       two cameras and the fullscreen display (OpenCV)
  synthetic.py  synthetic eye images for tests and examples
  __main__.py   the live tools
hardware/
  frame.scad    parametric glasses frame, three printable parts
examples/
  01_end_to_end.py     the geometry numbers above
  02_image_pipeline.py the image-level numbers above
  make_figures.py      the figures above
docs/
  SESSION_PROTOCOL.md  parts list and the first real recording
  AUDIT.md             what was checked on 7 Oct 2026, and what changed
```

---

## Built with AI assistance

Large parts of this repository were written with AI coding tools (Claude). The
author reviewed it and can explain how each part works; see
[EXPLAIN.md](EXPLAIN.md) for the plain-English version.

---

## License

MIT — see [LICENSE](LICENSE).

Parth Deshmukh
