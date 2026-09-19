"""Regenerate the figures in docs/figures/.

    python examples/make_figures.py

Same seed as the end-to-end example, so the pictures and the numbers in the
README describe the same run.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon

from headgaze.mapping import EyeToSceneMap, cross_validated_residuals
from headgaze.pupil import fit_ellipse_robust
from headgaze.screen import (
    ScreenLocator,
    ScreenMarkers,
    apply_homography,
    homography_from_points,
    reprojection_error,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"
SEED = 20260918
SCREEN_W, SCREEN_H = 1920, 1080

INK = "#1b1f24"
MUTED = "#8c959f"
RAW = "#bcc4cc"
ACCENT = "#0969da"
WARM = "#bc4c00"
GOOD = "#1a7f37"
BAD = "#cf222e"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    }
)


def _ellipse_points(centre, axes, angle, n=70):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local = np.column_stack([axes[0] * np.cos(t), axes[1] * np.sin(t)])
    c, s = np.cos(angle), np.sin(angle)
    return local @ np.array([[c, -s], [s, c]]).T + np.asarray(centre)


# --------------------------------------------------------------- figure 1


def figure_pupil_fit():
    """Why the ellipse fit is robust rather than least squares."""
    rng = np.random.default_rng(SEED)
    truth_centre = np.array([140.0, 110.0])
    truth_axes = (24.0, 17.0)
    truth_angle = 0.2

    boundary = _ellipse_points(truth_centre, truth_axes, truth_angle, n=70)
    boundary = boundary + rng.normal(0.0, 0.5, size=boundary.shape)

    # Eyelashes: a dark streak crossing the upper pupil, which survives any
    # threshold and lands in the boundary point set.
    lashes = np.column_stack(
        [np.linspace(128.0, 152.0, 9), np.full(9, 127.0) + np.linspace(0.0, 4.0, 9)]
    )
    contaminated = np.vstack([boundary, lashes])

    robust_c, robust_a, robust_ang = fit_ellipse_robust(
        contaminated, iterations=3, keep_fraction=0.85
    )
    plain_c, plain_a, plain_ang = fit_ellipse_robust(
        contaminated, iterations=1, keep_fraction=1.0
    )

    fig, (ax_img, ax_fit) = plt.subplots(1, 2, figsize=(10.2, 4.4))

    # ---- left: a synthetic visible-light eye crop ------------------------
    size = 220
    yy, xx = np.mgrid[0:size, 0:size]
    image = 150.0 + rng.normal(0.0, 6.0, size=(size, size))

    iris = ((xx - 140) ** 2 / 55**2 + (yy - 110) ** 2 / 48**2) < 1.0
    image[iris] = 95.0 + rng.normal(0.0, 5.0, size=iris.sum())

    pupil = (
        (xx - truth_centre[0]) ** 2 / truth_axes[0] ** 2
        + (yy - truth_centre[1]) ** 2 / truth_axes[1] ** 2
    ) < 1.0
    image[pupil] = 42.0 + rng.normal(0.0, 4.0, size=pupil.sum())

    for gx, gy in [(132, 100), (150, 118), (120, 92)]:
        glint = ((xx - gx) ** 2 + (yy - gy) ** 2) < 9
        image[glint] = 250.0

    image[126:131, 128:153] = 55.0  # the eyelash streak

    ax_img.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax_img.set_title("visible light: low contrast, glints, lashes", loc="left")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(False)

    ax_img.annotate(
        "corneal reflections",
        xy=(119, 90),
        xytext=(10, 26),
        fontsize=8,
        color="white",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.0},
    )
    ax_img.annotate(
        "eyelash",
        xy=(133, 131),
        xytext=(22, 206),
        fontsize=8,
        color="white",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.0},
    )

    # ---- right: the two fits ---------------------------------------------
    ax_fit.scatter(boundary[:, 0], boundary[:, 1], s=9, color=MUTED, label="pupil boundary")
    ax_fit.scatter(
        lashes[:, 0], lashes[:, 1], s=26, color=BAD, zorder=3, label="eyelash pixels"
    )

    ax_fit.add_patch(
        Ellipse(
            truth_centre,
            2 * truth_axes[0],
            2 * truth_axes[1],
            angle=np.degrees(truth_angle),
            fill=False,
            edgecolor=INK,
            lw=1.2,
            ls=(0, (4, 3)),
            label="true pupil",
        )
    )
    ax_fit.add_patch(
        Ellipse(
            plain_c,
            2 * plain_a[0],
            2 * plain_a[1],
            angle=np.degrees(plain_ang),
            fill=False,
            edgecolor=WARM,
            lw=1.8,
            label="plain least squares",
        )
    )
    ax_fit.add_patch(
        Ellipse(
            robust_c,
            2 * robust_a[0],
            2 * robust_a[1],
            angle=np.degrees(robust_ang),
            fill=False,
            edgecolor=ACCENT,
            lw=1.8,
            label="trimmed (this repo)",
        )
    )

    robust_error = float(np.linalg.norm(robust_c - truth_centre))
    plain_error = float(np.linalg.norm(plain_c - truth_centre))

    ax_fit.set_aspect("equal")
    ax_fit.set_xlim(100, 180)
    ax_fit.set_ylim(150, 70)
    ax_fit.set_xlabel("eye-camera x (px)")
    ax_fit.set_ylabel("eye-camera y (px)")
    ax_fit.legend(loc="lower left", ncol=2)
    ax_fit.set_title(
        f"centre error: trimmed {robust_error:.2f} px, plain {plain_error:.2f} px", loc="left"
    )

    fig.tight_layout()
    fig.savefig(OUT / "01_pupil_fit.png", bbox_inches="tight")
    plt.close(fig)

    return robust_error, plain_error


# --------------------------------------------------------------- figure 2


def figure_screen_mapping():
    """What the scene camera sees, and where gaze lands after the homography."""
    rng = np.random.default_rng(SEED)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    locator = ScreenLocator(markers, max_reprojection_px=8.0)

    pose = np.eye(3)
    pose[:2, :2] += np.array([[-0.12, 0.10], [-0.06, -0.09]])
    pose[:2, 2] = np.array([180.0, 60.0])
    pose[2, :2] = np.array([-4.5e-5, 2.0e-5])
    pose /= pose[2, 2]

    screen_outline = np.array(
        [[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]], dtype=float
    )
    seen_outline = apply_homography(pose, screen_outline)

    observed = {
        mid: apply_homography(pose, corners) + rng.normal(0.0, 0.4, size=(4, 2))
        for mid, corners in markers.corners_px.items()
    }
    homography, error = locator.locate(observed)

    gaze_targets = np.array([[420.0, 260.0], [1500.0, 300.0], [980.0, 820.0], [1600.0, 900.0]])
    gaze_in_scene = apply_homography(pose, gaze_targets)
    recovered = apply_homography(homography, gaze_in_scene)

    fig, (ax_scene, ax_screen) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    ax_scene.add_patch(
        Polygon(
            seen_outline, closed=True, facecolor=ACCENT, alpha=0.06, edgecolor=MUTED, lw=1.0
        )
    )
    for corners in observed.values():
        ax_scene.add_patch(
            Polygon(corners, closed=True, facecolor=INK, edgecolor=INK, lw=1.0, alpha=0.85)
        )
    ax_scene.scatter(
        gaze_in_scene[:, 0], gaze_in_scene[:, 1], s=70, color=WARM, zorder=4, label="gaze"
    )

    ax_scene.set_title("scene camera (head tilted)", loc="left")
    ax_scene.set_aspect("equal")
    ax_scene.invert_yaxis()
    ax_scene.set_xticks([])
    ax_scene.set_yticks([])
    ax_scene.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01))
    for spine in ax_scene.spines.values():
        spine.set_visible(False)

    ax_screen.add_patch(
        Polygon(
            screen_outline, closed=True, facecolor=GOOD, alpha=0.06, edgecolor=MUTED, lw=1.0
        )
    )
    for corners in markers.corners_px.values():
        ax_screen.add_patch(Polygon(corners, closed=True, facecolor=INK, alpha=0.85, lw=0))

    ax_screen.scatter(
        gaze_targets[:, 0],
        gaze_targets[:, 1],
        s=150,
        facecolors="none",
        edgecolors=GOOD,
        lw=1.8,
        label="where they looked",
        zorder=3,
    )
    ax_screen.scatter(
        recovered[:, 0], recovered[:, 1], s=40, color=WARM, zorder=4, label="recovered"
    )

    residual = float(np.mean(np.linalg.norm(recovered - gaze_targets, axis=1)))

    ax_screen.set_title(
        f"screen, after the homography (reprojection {error:.2f} px, gaze {residual:.2f} px)",
        loc="left",
    )
    ax_screen.set_aspect("equal")
    ax_screen.set_xlim(-60, SCREEN_W + 60)
    ax_screen.set_ylim(SCREEN_H + 60, -60)
    ax_screen.set_xticks([])
    ax_screen.set_yticks([])
    ax_screen.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2)
    for spine in ax_screen.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "02_screen_mapping.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3


def figure_overdetermination():
    """Marker centres cannot catch a bad detection; marker corners can.

    This is the whole reason the locator keeps every corner, so it is worth a
    picture rather than a sentence.
    """
    rng = np.random.default_rng(SEED)
    markers = ScreenMarkers.for_screen(SCREEN_W, SCREEN_H)
    ids = list(markers.ids)

    def pose(seed):
        r = np.random.default_rng(seed)
        h = np.eye(3)
        h[:2, :2] += r.normal(0.0, 0.10, size=(2, 2))
        h[:2, 2] = r.normal(0.0, 40.0, size=2)
        h[2, :2] = r.normal(0.0, 4e-5, size=2)
        return h / h[2, 2]

    DISPLACEMENT = np.array([120.0, -90.0])

    def run(n_markers, use_corners):
        errors = []
        for trial in range(80):
            p = pose(trial)
            source, destination = [], []

            for mid in ids[:n_markers]:
                screen_corners = markers.corners_px[mid]
                seen = apply_homography(p, screen_corners)
                seen = seen + rng.normal(0.0, 0.3, size=seen.shape)

                if mid == ids[0]:
                    seen = seen + DISPLACEMENT  # one misdetected marker

                if use_corners:
                    source.append(seen)
                    destination.append(screen_corners)
                else:
                    source.append(seen.mean(axis=0, keepdims=True))
                    destination.append(screen_corners.mean(axis=0, keepdims=True))

            src = np.vstack(source)
            dst = np.vstack(destination)
            try:
                h = homography_from_points(src, dst)
            except (ValueError, np.linalg.LinAlgError):
                continue
            errors.append(reprojection_error(h, src, dst))

        return float(np.median(errors))

    labels = [
        "4 marker centres\n(4 points, 8 equations)",
        "2 markers, corners\n(8 points, 16 equations)",
        "4 markers, corners\n(16 points, 32 equations)",
    ]
    values = [run(4, False), run(2, True), run(4, True)]
    colours = [BAD, ACCENT, ACCENT]

    fig, ax = plt.subplots(figsize=(8.4, 4.1))

    bars = ax.bar(np.arange(len(values)), values, width=0.55, color=colours)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:.2f} px" if value > 0.01 else f"{value:.1e} px",
            ha="center",
            fontsize=9,
            color=bar.get_facecolor(),
        )

    ax.axhline(8.0, color=GOOD, lw=1.2, ls=(0, (4, 3)))
    ax.text(-0.42, 8.35, "rejection threshold", fontsize=8, color=GOOD, ha="left")

    ax.annotate(
        "exactly determined:\nfits anything, so the\nmisdetection is invisible",
        xy=(0, values[0]),
        xytext=(0.15, max(values) * 0.55),
        fontsize=8.5,
        color=BAD,
        arrowprops={"arrowstyle": "->", "color": BAD, "lw": 1.0},
    )

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("reprojection error (px, median)")
    ax.set_title("One marker displaced by 150 px, in every case shown", loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "03_overdetermination.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4


def figure_calibration_honesty():
    """In-sample residual against leave-one-out, per calibration point."""
    rng = np.random.default_rng(SEED)

    side, margin = 4, 0.12
    coords = np.linspace(margin, 1 - margin, side)
    grid = np.array([(x, y) for y in coords for x in coords])
    screen = grid * np.array([SCREEN_W, SCREEN_H])

    u, v = grid[:, 0] - 0.5, grid[:, 1] - 0.5
    pupil = np.column_stack(
        [
            150.0 + 52.0 * u + 6.0 * u**3 + 3.0 * u * v,
            130.0 + 44.0 * v - 5.0 * v**3 - 2.0 * u * v,
        ]
    )
    pupil = pupil + rng.normal(0.0, 0.4, size=pupil.shape)

    scene = screen.copy()

    model = EyeToSceneMap(degree=2).fit(pupil, scene)
    in_sample = model.residuals_px(pupil, scene)
    held_out = cross_validated_residuals(pupil, scene)

    fig, ax = plt.subplots(figsize=(8.6, 3.9))

    idx = np.arange(len(in_sample))
    ax.bar(idx - 0.19, in_sample, width=0.36, color=RAW, label="scored on its own data")
    ax.bar(idx + 0.19, held_out, width=0.36, color=ACCENT, label="held out")

    ax.axhline(np.mean(in_sample), color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(np.nanmean(held_out), color=ACCENT, lw=1.0, ls=(0, (4, 3)))

    ax.text(
        len(idx) - 1.4,
        np.nanmean(held_out) * 1.07,
        f"mean {np.nanmean(held_out):.1f} px",
        fontsize=8,
        color=ACCENT,
        ha="right",
    )
    ax.text(
        len(idx) - 1.4,
        np.mean(in_sample) * 1.08,
        f"mean {np.mean(in_sample):.1f} px",
        fontsize=8,
        color=MUTED,
        ha="right",
    )

    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])
    ax.set_xlabel("calibration point")
    ax.set_ylabel("residual (px)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title(
        f"{len(idx)} points, {model.n_parameters} parameters: the in-sample number is "
        f"{np.nanmean(held_out) / np.mean(in_sample):.1f}x too kind",
        loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUT / "04_calibration_honesty.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_pupil_fit()
    figure_screen_mapping()
    figure_overdetermination()
    figure_calibration_honesty()
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
