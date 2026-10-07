# How this project works, in plain English

Read this until you can tell it without looking.

## 1. What it is (say this first)

A pair of glasses with two small cameras: one looks at your eye, one looks forward
at what you see. From the eye camera the program finds your pupil; from the forward
camera it finds four printed markers on the corners of the screen. Together they tell
where on the screen you are looking, even while your head moves. It uses normal light
only, with no infrared lamp next to the eye.

**Honest status:** the software is written and tested on simulated images. The glasses
have not been built yet, so there are no measurements with a real eye.

## 2. How it works: five steps

1. **Find the pupil.** The eye camera image is searched for the darkest round blob.
   Bright reflections are painted dark first, specks from eyelashes are cleaned away,
   and an ellipse is fitted in a way that ignores the worst-fitting edge points.
2. **Pupil to forward-camera point.** A calibration (you look at 9 dots) teaches the
   program which pupil position means which point in the forward camera's picture.
   This never changes while the glasses sit still on your face.
3. **Find the screen.** The forward camera sees the four markers. From their corners the
   program computes, *in every single frame*, how the camera picture maps onto the screen
   (a *homography*: the exact mapping between two flat surfaces).
4. **Gaze on the screen.** Step 2 gives the point in the camera picture, step 3 maps it
   onto the screen.
5. **Send it on.** The gaze point goes over a local network message to the robot-camera
   demo, so the glasses can steer the robot instead of the webcam.

## 3. The ideas worth saying in an interview

- **Why no infrared?** Infrared makes the pupil easy to find, but putting a light source
  next to an eye needs a proper safety check. With normal light the pupil is harder to find:
  dark brown eyes give little contrast, room lights make reflections, and eyelashes are dark too.
- **Why two separate stages?** The pupil-to-camera relation only depends on how the glasses
  sit, so it is calibrated once. The camera-to-screen relation depends on where your head is,
  so it is recomputed every frame. Keeping them apart is what lets you move your head.
- **Why use all marker corners?** Four points always fit a homography exactly, even wrong ones,
  so you cannot tell when a marker was misread. Sixteen corners give a check.
- **How it fails differently from the webcam tracker.** The webcam tracker gets worse when you
  move your head. The glasses don't care about head movement, but they get worse when they
  *slip* on your nose, because the calibration assumed the camera stays where it was.
- **Why validate on new dots?** Accuracy measured on the calibration dots looks far better
  than it really is; only new points give an honest number.

## 4. Files

| File | What it is for |
| --- | --- |
| `src/headgaze/detect.py` | Eye image to pupil |
| `src/headgaze/pupil.py` | The pieces: reflection removal, threshold, robust ellipse fit, confidence |
| `src/headgaze/mapping.py` | The calibrated pupil-to-camera map, its honest error estimate, slippage check |
| `src/headgaze/aruco.py` | Drawing and finding the screen markers |
| `src/headgaze/screen.py` | Camera picture to screen (the homography), recomputed every frame |
| `src/headgaze/session.py` | One pair of frames to one gaze point; calibration and validation |
| `src/headgaze/metrics.py` | Errors in degrees of visual angle |
| `src/headgaze/stream.py` | Sends gaze to the robot demo; smoothing filter |
| `src/headgaze/live.py`, `__main__.py` | The cameras, the fullscreen screen, the commands |
| `hardware/frame.scad` | The glasses frame as an editable 3D design |
| `docs/SESSION_PROTOCOL.md` | What to buy and how the first real recording works |

## 5. Results

**Simulated only** (say "simulated" every time): on synthetic eye images the pupil centre is
found to within about a quarter of a pixel, and the whole chain reaches about 0.1° on the screen
with the head moving. That shows the code is right. A real eye in room light will be much worse;
the real number comes from the first session: [TBD].

## 6. Likely questions, honest answers

1. **"Have you built it?"** "Not yet. The software is done and tested on simulated images; the
   cameras and frame are the next step, and I've written the recording plan."
2. **"Why not infrared like everyone else?"** See section 3: safety, at the price of contrast.
3. **"What's the biggest problem with head-worn trackers?"** Slippage of the glasses, and parallax:
   calibrated at the screen distance, it is off when you look much nearer or further.
4. **"How would you test it?"** Accuracy on new dots with the head still, with the head moving,
   and after taking the glasses off and on five times; compared with the webcam tracker in the same light.
5. **"Did you write this yourself?"** "I built it with a lot of help from AI coding tools and went
   through how each part works."
