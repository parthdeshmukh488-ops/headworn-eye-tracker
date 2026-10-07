"""Live use of the head-worn tracker. Needs OpenCV (`pip install -e ".[camera]"`).

    python -m headgaze print-markers                 # printable markers for the screen corners
    python -m headgaze preview                       # check both cameras, pupil and markers
    python -m headgaze calibrate                     # 9 dots -> data/calibration.json
    python -m headgaze validate                      # 9 new dots -> accuracy in degrees
    python -m headgaze serve --markers paper         # gaze over UDP to the robot-camera app

--eye and --scene take a camera index or a video file (to replay a recording).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from headgaze.aruco import DICTIONARY, render_screen
from headgaze.detect import draw_pupil
from headgaze.live import CameraPair, Display, draw_dot, put_lines
from headgaze.mapping import EyeToSceneMap
from headgaze.metrics import Screen
from headgaze.screen import ScreenLocator, ScreenMarkers
from headgaze.session import (
    CALIBRATION_POINTS,
    VALIDATION_POINTS,
    Tracker,
    fit_calibration,
    validation_summary,
)
from headgaze.stream import DEFAULT_PORT, GazeSender, OneEuroFilter

DATA = Path("data")  # gitignored: calibrations are personal
RESULTS = Path("results")
ESC, SPACE = 27, 32


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def screen_of(args) -> Screen:
    return Screen(args.screen_px[0], args.screen_px[1], args.screen_width_cm, args.distance_cm)


def on_screen_markers(args) -> ScreenMarkers:
    return ScreenMarkers.for_screen(*args.screen_px)


def paper_markers(args) -> ScreenMarkers:
    """Printed markers taped with their black square's outer corner on the screen corner."""
    side_px = args.marker_mm / (args.screen_width_cm * 10.0 / args.screen_px[0])
    return ScreenMarkers.for_screen(*args.screen_px, marker_size_px=side_px, inset_px=0.0)


def open_cameras(args) -> CameraPair:
    try:
        return CameraPair(args.eye, args.scene)
    except RuntimeError as exc:
        sys.exit(f"{exc}. Check --eye / --scene (camera index or video file).")


# ----------------------------------------------------------------- dot runs
def run_dots(args, title: str, points_norm, model=None):
    """Show the dots fullscreen over the marker screen; returns [(dot index, Sample)]."""
    width, height = args.screen_px
    markers = on_screen_markers(args)
    base = cv2.cvtColor(render_screen(markers, width, height), cv2.COLOR_GRAY2BGR)
    points_px = np.array(points_norm) * [width, height]
    tracker = Tracker(ScreenLocator(markers), model)
    cams, display = open_cameras(args), Display(width, height)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(points_px))
    records = []
    try:
        while True:  # instructions until SPACE, with a live status line
            pair = cams.read()
            status = "no camera frames"
            if pair is not None:
                s = tracker.process(pair.t, pair.eye, pair.scene)
                status = f"pupil {'OK' if s.pupil.valid else 'not found'} | markers {len(s.markers)}/4"
            lines = [
                title,
                "Keep your head still and follow each dot with your eyes.",
                "SPACE to start, ESC to cancel.",
                status,
            ]
            key = display.show(put_lines(base.copy(), lines))
            if key == ESC:
                return None, points_px
            if key == SPACE:
                break
        for index in order:
            onset = time.time()
            while (elapsed := time.time() - onset) < args.dot_s:
                if display.show(draw_dot(base, points_px[index], elapsed / 0.3)) == ESC:
                    return None, points_px
                pair = cams.read()
                if pair is None:
                    continue
                sample = tracker.process(pair.t, pair.eye, pair.scene)
                if elapsed >= args.discard_s:  # the eye is still travelling to the dot
                    records.append((int(index), sample))
    finally:
        cams.close()
        display.close()
    return records, points_px


def cmd_calibrate(args) -> None:
    records, points_px = run_dots(args, "Calibration", CALIBRATION_POINTS)
    if records is None:
        sys.exit("Cancelled.")
    try:
        model, report = fit_calibration(records, points_px)
    except ValueError as exc:
        sys.exit(f"Calibration failed: {exc}. Run `python -m headgaze preview` first.")
    counts = report["samples_per_dot"]
    print(f"usable samples per dot: {[counts.get(i, 0) for i in range(len(points_px))]}")
    if sum(1 for i in range(len(points_px)) if counts.get(i, 0) >= 5) < 7:
        sys.exit("Too few usable samples on too many dots; check light and camera placement.")
    out = Path(args.calibration)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"created": stamp(), "map": model.to_dict(), "report": report}, indent=2)
    )
    loo = report["leave_one_dot_out_scene_px"]
    print(f"saved {out}: {report['n_samples']} samples")
    print(
        f"leave-one-dot-out error in the scene image: mean {loo['mean']:.1f} px, max {loo['max']:.1f} px"
    )
    print("Accuracy in degrees comes from `python -m headgaze validate`.")


def load_model(args) -> EyeToSceneMap:
    path = Path(args.calibration)
    if not path.exists():
        sys.exit(f"{path} not found: run `python -m headgaze calibrate` first.")
    return EyeToSceneMap.from_dict(json.loads(path.read_text())["map"])


def cmd_validate(args) -> None:
    records, points_px = run_dots(args, "Validation", VALIDATION_POINTS, load_model(args))
    if records is None:
        sys.exit("Cancelled.")
    summary = validation_summary(records, points_px, screen_of(args))
    summary.update(
        created=stamp(),
        calibration=str(args.calibration),
        screen={
            "width_px": args.screen_px[0],
            "height_px": args.screen_px[1],
            "width_cm": args.screen_width_cm,
            "distance_cm": args.distance_cm,
        },
        note=args.note,
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"validation_{summary['created']}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"accuracy  {summary['accuracy_deg']:.2f} deg")
    print(f"precision {summary['precision_rms_s2s_deg']:.2f} deg RMS sample-to-sample")
    print(f"data loss {summary['data_loss_pct']:.1f} %")
    print(f"saved {out}")


# ------------------------------------------------------------- live views
def cmd_preview(args) -> None:
    width, height = args.screen_px
    markers = on_screen_markers(args)
    base = cv2.cvtColor(render_screen(markers, width, height), cv2.COLOR_GRAY2BGR)
    tracker = Tracker(ScreenLocator(markers))
    cams, display = open_cameras(args), Display(width, height)
    try:
        while True:
            pair = cams.read()
            if pair is None and not args.eye.isdigit():
                break  # end of a replayed recording
            frame = base.copy()
            if pair is not None:
                s = tracker.process(pair.t, pair.eye, pair.scene)
                eye = cv2.resize(draw_pupil(pair.eye, s.pupil), (480, 360))
                scene = pair.scene.copy()
                if s.markers:
                    ids = np.array(list(s.markers), dtype=np.int32).reshape(-1, 1)
                    corners = [
                        c.reshape(1, 4, 2).astype(np.float32) for c in s.markers.values()
                    ]
                    cv2.aruco.drawDetectedMarkers(scene, corners, ids)
                scene = cv2.resize(scene, (480, 360))
                y0, x0 = height // 2 - 180, width // 2 - 490
                frame[y0 : y0 + 360, x0 : x0 + 480] = eye
                frame[y0 : y0 + 360, x0 + 500 : x0 + 980] = scene
                status = (
                    f"pupil confidence {s.pupil.confidence:.2f} | markers {len(s.markers)}/4 | "
                    f"screen fit {s.reprojection_px:.1f} px"
                )
                put_lines(frame, [status, "ESC to quit"], origin=(x0, y0 + 400), scale=0.7)
            if display.show(frame) == ESC:
                break
    finally:
        cams.close()
        display.close()


def cmd_serve(args) -> None:
    width, height = args.screen_px
    markers = paper_markers(args) if args.markers == "paper" else on_screen_markers(args)
    tracker = Tracker(ScreenLocator(markers), load_model(args))
    cams = open_cameras(args)
    display = None
    if args.markers == "screen":  # the markers have to be on screen for the scene camera
        display = Display(width, height)
        marker_screen = render_screen(markers, width, height)
    sender, filt = GazeSender(port=args.port), OneEuroFilter()
    seq, n, n_valid, last_valid, t_report = 0, 0, 0, None, time.time()
    print(
        f"Streaming gaze to udp://127.0.0.1:{args.port} ({args.markers} markers). Ctrl+C to stop."
    )
    try:
        while True:
            if display is not None and display.show(marker_screen) == ESC:
                break
            pair = cams.read()
            if pair is None:
                if not args.eye.isdigit():
                    break  # end of a replayed recording
                continue
            s = tracker.process(pair.t, pair.eye, pair.scene)
            seq, n = seq + 1, n + 1
            if s.gaze_px is None:
                sender.send(s.t, None, None, seq)
            else:
                n_valid += 1
                if last_valid is None or s.t - last_valid > 0.5:
                    filt.reset()  # after a gap the old filter state is stale
                last_valid = s.t
                fx, fy = filt(s.t, *s.gaze_px)
                sender.send(
                    s.t, (fx / width, fy / height), tuple(s.gaze_px / [width, height]), seq
                )
            if time.time() - t_report >= 2.0:
                rate = n / (time.time() - t_report)
                print(f"\r{rate:5.1f} Hz, valid {100 * n_valid / n:5.1f} %", end="", flush=True)
                n, n_valid, t_report = 0, 0, time.time()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cams.close()
        sender.close()
        if display is not None:
            display.close()


def cmd_print_markers(args) -> None:
    """A 300 dpi page with the four markers at their real size, labelled by corner."""
    dpi = 300
    side = int(round(args.marker_mm / 25.4 * dpi))
    margin = int(round(10 / 25.4 * dpi))
    cell = side + 2 * margin
    page = np.full((2 * cell + 260, 2 * cell), 255, np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    names = {0: "0: top-left", 1: "1: top-right", 2: "2: bottom-right", 3: "3: bottom-left"}
    for marker_id, (row, col) in zip(range(4), [(0, 0), (0, 1), (1, 1), (1, 0)], strict=True):
        y, x = row * cell + margin, col * cell + margin
        page[y : y + side, x : x + side] = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, side
        )
        cv2.putText(
            page,
            names[marker_id],
            (x, y + side + margin // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            0,
            2,
            cv2.LINE_AA,
        )
    lines = [
        f"Black squares {args.marker_mm:g} mm. Print at 100 % scale.",
        "Tape each marker upright, its black square's",
        "outer corner on that corner of the screen.",
    ]
    for k, line in enumerate(lines):
        y = 2 * cell + 60 + 55 * k
        cv2.putText(page, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.imwrite(args.out, page)
    print(
        f"saved {args.out} ({dpi} dpi); check the printed black squares measure {args.marker_mm:g} mm"
    )


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--eye", default="1", help="eye camera index or video file")
    common.add_argument("--scene", default="2", help="scene camera index or video file")
    common.add_argument(
        "--screen-px", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H")
    )
    common.add_argument("--screen-width-cm", type=float, default=34.5)
    common.add_argument("--distance-cm", type=float, default=60.0)
    common.add_argument("--calibration", default=str(DATA / "calibration.json"))
    common.add_argument("--dot-s", type=float, default=1.5, help="seconds per dot")
    common.add_argument(
        "--discard-s", type=float, default=0.3, help="ignored start of each dot"
    )
    common.add_argument("--seed", type=int, default=0)

    p = argparse.ArgumentParser(
        prog="python -m headgaze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("preview", parents=[common]).set_defaults(func=cmd_preview)
    sub.add_parser("calibrate", parents=[common]).set_defaults(func=cmd_calibrate)
    v = sub.add_parser("validate", parents=[common])
    v.add_argument("--note", default="", help="e.g. 'head still', 'head moving', 're-seated 3'")
    v.set_defaults(func=cmd_validate)
    s = sub.add_parser("serve", parents=[common])
    s.add_argument("--markers", choices=["paper", "screen"], default="paper")
    s.add_argument("--marker-mm", type=float, default=30.0)
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.set_defaults(func=cmd_serve)
    m = sub.add_parser("print-markers", parents=[common])
    m.add_argument("--marker-mm", type=float, default=30.0)
    m.add_argument("--out", default="markers_print.png")
    m.set_defaults(func=cmd_print_markers)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
