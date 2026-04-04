"""
Cell Slicer — Segmentation Pipeline v3
========================================
Identifies and marks cell boundaries in microscopy video:
  - Red Blood Cells (RBCs)  → RED overlay
  - Neutrophil              → GREEN overlay

Temporal strategy:
─────────────────
RBCs are static:
  • Hough circle detections are accumulated across all frames into a stable
    density map (positions that recur in ≥20% of frames = confirmed RBC territory).
  • Per-frame Hough results supplement the stable map for RBCs that appear/
    disappear near stage moves.
  • Marker-controlled watershed expands stable RBC seeds into actual cell shapes.

Neutrophil moves:
  • Frame-to-frame difference (|frame_t - frame_{t-N}|) signals motion.
  • The motion signal is intersected with cell bodies and subtracted of all
    known RBC territory → confirmed motion region.
  • The confirmed motion region is used as a SEED; the full neutrophil body is
    recovered by growing that seed into the connected cell-body component
    that contains it (with aggressive morphological closing to merge pseudopods).

Stage jumps:
  • Inter-frame diff spike → flag frame, reset temporal model, skip segmentation.

Usage:
    python3 pipeline.py [--input PATH] [--output PATH] [--build-density-map] [options]

Two-pass workflow (recommended):
    python3 pipeline.py --build-density-map   # pass 1: ~4s, saves rbc_density.npy
    python3 pipeline.py                        # pass 2: uses cached density map
"""

import argparse
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from collections import deque
from scipy import ndimage


# ─── Colour constants (RGB) ────────────────────────────────────────────────────
COLOUR_RBC_RGB        = (220, 60,  60)
COLOUR_NEUTROPHIL_RGB = (40,  220, 40)
ALPHA_FILL            = 0.40


# ─── Stage jump ────────────────────────────────────────────────────────────────

def detect_stage_jump(prev_gray, curr_gray, threshold=18.0):
    if prev_gray is None:
        return False
    return float(cv2.absdiff(prev_gray, curr_gray).astype(np.float32).mean()) > threshold


# ─── Preprocessing ─────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(frame_rgb):
    gray     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    enhanced = _clahe.apply(gray)
    return gray, enhanced


# ─── Density map (pass 1) ──────────────────────────────────────────────────────

def build_density_map(frames, rbc_radius, density_path, dot_r=10):
    """
    Accumulate Hough circle positions across all frames.
    Returns a (H, W) float32 array of fraction-of-frames each pixel was near a seed.
    """
    h, w = frames.shape[1], frames.shape[2]
    counts  = np.zeros((h, w), np.int32)
    n_total = len(frames)

    print("Building RBC density map…")
    for frame in tqdm(frames, unit="frame"):
        _, enh = preprocess(frame)
        circles = _find_hough(enh, rbc_radius)
        layer   = np.zeros((h, w), np.int32)
        if circles is not None:
            for (cx, cy, cr) in circles:
                if 0 <= cy < h and 0 <= cx < w:
                    cv2.circle(layer, (cx, cy), dot_r, 1, -1)
        counts += layer

    density = counts.astype(np.float32) / n_total
    np.save(str(density_path), density)
    print(f"  Saved → {density_path}  (max={density.max():.3f})")
    return density


# ─── Hough ─────────────────────────────────────────────────────────────────────

def _find_hough(enhanced, rbc_radius):
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(rbc_radius * 1.4),
        param1=50, param2=13,
        minRadius=int(rbc_radius * 0.6),
        maxRadius=int(rbc_radius * 1.2),
    )
    return np.round(circles[0]).astype(int) if circles is not None else None


# ─── Cell body mask ────────────────────────────────────────────────────────────

def get_cell_bodies(enhanced, dark_thresh=100, close_r=9):
    """
    Binary mask covering all cell material (RBCs + neutrophil).
    Uses a more aggressive close to bridge neutrophil pseudopods.
    """
    _, dark = cv2.threshold(enhanced, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r, close_r))
    # Aggressive close to merge pseudopods and internal gaps
    closed  = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_big, iterations=2)
    filled  = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)
    return cleaned


# ─── RBC segmentation ──────────────────────────────────────────────────────────

def get_rbc_zone(density, hough_circles, rbc_radius, h, w,
                 density_thresh=0.20, hough_expand=1.3, stable_expand=26):
    """Combined RBC exclusion zone from stable density + current Hough."""
    rbc_stable   = (density >= density_thresh).astype(np.uint8) * 255
    k_stable     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (stable_expand*2+1, stable_expand*2+1))
    rbc_expanded = cv2.dilate(rbc_stable, k_stable)

    rbc_hough = np.zeros((h, w), np.uint8)
    if hough_circles is not None:
        for (cx, cy, cr) in hough_circles:
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(rbc_hough, (cx, cy), int(cr * hough_expand), 255, -1)

    return cv2.bitwise_or(rbc_hough, rbc_expanded)


def watershed_rbcs(gray, cell_bodies, density, hough_circles, rbc_radius):
    """Marker-controlled watershed for RBC boundaries using stable seeds."""
    h, w = gray.shape
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Seeds: stable density peaks only (reliable, no noise)
    seed_mask = (density >= 0.20).astype(np.uint8) * 255
    # Also seed from current Hough for cells that just appeared
    if hough_circles is not None:
        for (cx, cy, cr) in hough_circles:
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(seed_mask, (cx, cy), max(2, int(rbc_radius * 0.22)), 255, -1)

    # Individual seed blobs → markers
    n_seeds, seed_labels = cv2.connectedComponents(seed_mask)
    markers = np.zeros((h, w), dtype=np.int32)
    for lbl in range(1, n_seeds):
        markers[seed_labels == lbl] = lbl + 1

    # Sure background
    sure_bg = cv2.dilate(cell_bodies, k3, iterations=3)
    markers[sure_bg == 0] = 1

    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_bgr, markers)

    # Collect RBC-sized regions
    rbc_min = np.pi * (rbc_radius * 0.50) ** 2
    rbc_max = np.pi * (rbc_radius * 1.60) ** 2
    rbc_mask = np.zeros((h, w), np.uint8)
    for lbl in range(2, int(markers.max()) + 1):
        region = (markers == lbl).astype(np.uint8)
        area   = int(region.sum())
        if rbc_min <= area <= rbc_max:
            rbc_mask |= region * 255

    return rbc_mask


# ─── Neutrophil detection ──────────────────────────────────────────────────────

class MotionBuffer:
    """Stores recent grayscale frames for inter-frame diff."""
    def __init__(self, lag=8):
        self.lag    = lag
        self.buffer = deque(maxlen=lag + 1)

    def update(self, gray):
        self.buffer.append(gray.copy())

    def motion_diff(self):
        """Return |frame_t - frame_{t-lag}|, or None if not enough frames."""
        if len(self.buffer) < self.lag + 1:
            return None
        return cv2.absdiff(self.buffer[-1], self.buffer[0])

    def reset(self):
        self.buffer.clear()


def detect_neutrophil(enhanced, gray, cell_bodies, rbc_zone, motion_diff,
                      rbc_radius, h, w):
    """
    Find the neutrophil as the largest moving, non-RBC cell body region.
    Returns binary mask.
    """
    if motion_diff is None:
        return np.zeros((h, w), np.uint8)

    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Threshold motion
    _, fg = cv2.threshold(motion_diff, 15, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_big, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k3)

    # Constrain to cell bodies and remove RBC territory
    fg_cells  = cv2.bitwise_and(fg, cell_bodies)
    candidate = cv2.bitwise_and(fg_cells, cv2.bitwise_not(rbc_zone))

    # Pick the best seed blob (largest, away from image border)
    n_comp, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(candidate)
    margin       = 15
    min_seed_area = 80
    valid = [
        (stats[l, cv2.CC_STAT_AREA], l, centroids[l])
        for l in range(1, n_comp)
        if stats[l, cv2.CC_STAT_AREA] >= min_seed_area
        and centroids[l][0] > margin and centroids[l][0] < w - margin
        and centroids[l][1] > margin and centroids[l][1] < h - margin
    ]
    if not valid:
        return np.zeros((h, w), np.uint8)

    best_area, best_lbl, best_cent = max(valid, key=lambda x: x[0])
    seed_mask = ((comp_labels == best_lbl) * 255).astype(np.uint8)

    # Grow seed into connected cell_body components
    # (aggressive close merges pseudopods into one blob)
    seed_dilated   = cv2.dilate(seed_mask,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n_cb, cb_labels = cv2.connectedComponents(cell_bodies)
    touched         = set(np.unique(cb_labels[seed_dilated > 0])) - {0}

    if not touched:
        return np.zeros((h, w), np.uint8)

    neutro = np.zeros((h, w), np.uint8)
    for lbl in touched:
        neutro[cb_labels == lbl] = 255

    # Must be larger than an RBC
    min_neutro_area = np.pi * rbc_radius ** 2 * 1.5
    if neutro.sum() // 255 < min_neutro_area:
        return np.zeros((h, w), np.uint8)

    # Fill holes and return
    return ndimage.binary_fill_holes(neutro).astype(np.uint8) * 255


# ─── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame_rgb, rbc_mask, neutro_mask):
    result = frame_rgb.copy().astype(np.float32)

    def fill(img, mask, colour):
        for c, val in enumerate(colour):
            img[:, :, c] = np.where(
                mask > 0,
                img[:, :, c] * (1 - ALPHA_FILL) + val * ALPHA_FILL,
                img[:, :, c]
            )

    fill(result, rbc_mask,    COLOUR_RBC_RGB)
    fill(result, neutro_mask, COLOUR_NEUTROPHIL_RGB)
    result = np.clip(result, 0, 255).astype(np.uint8)

    def outlines(img, mask, colour, thickness):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colour, thickness)

    outlines(result, rbc_mask,    COLOUR_RBC_RGB,        1)
    outlines(result, neutro_mask, COLOUR_NEUTROPHIL_RGB, 2)
    return result


def draw_legend(frame, stage_jump=False, warming_up=False):
    items = [("Neutrophil", COLOUR_NEUTROPHIL_RGB), ("RBC", COLOUR_RBC_RGB)]
    h, w  = frame.shape[:2]
    x, y  = 5, 12
    for label, colour in items:
        cv2.circle(frame, (x + 4, y - 3), 4, colour, -1)
        cv2.putText(frame, label, (x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        y += 14
    if stage_jump:
        cv2.putText(frame, "STAGE MOVE", (w // 2 - 42, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    if warming_up:
        cv2.putText(frame, "warming up...", (w // 2 - 46, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 100), 1, cv2.LINE_AA)
    return frame


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           default="source-movie/chase-original.mp4")
    parser.add_argument("--output",          default="output/chase-segmented.mp4")
    parser.add_argument("--density-map",     default="output/rbc_density.npy",
                        help="Path to cached density map (created by --build-density-map)")
    parser.add_argument("--build-density-map", action="store_true",
                        help="Run pass 1: build and save RBC density map, then exit")
    parser.add_argument("--rbc-radius",      type=float, default=17.0)
    parser.add_argument("--dark-thresh",     type=int,   default=100)
    parser.add_argument("--motion-lag",      type=int,   default=8,
                        help="Frame lag for inter-frame motion diff (default: 8)")
    parser.add_argument("--stage-jump-threshold", type=float, default=18.0)
    parser.add_argument("--test",            type=int,   default=0)
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    density_path = Path(args.density_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    reader = iio.imopen(str(input_path), "r", plugin="pyav")
    fps    = reader.metadata().get("fps", 15.0)
    reader.close()
    frames = iio.imread(str(input_path), plugin="pyav", index=None)
    n_total = len(frames)

    # ── Pass 1: build density map ────────────────────────────────────────────
    if args.build_density_map:
        build_density_map(frames, args.rbc_radius, density_path)
        return

    # ── Load or build density map ────────────────────────────────────────────
    if density_path.exists():
        print(f"Loading density map: {density_path}")
        density = np.load(str(density_path))
    else:
        print("No density map found — building now (this takes ~4s)…")
        density = build_density_map(frames, args.rbc_radius, density_path)

    # ── Pass 2: segmentation ─────────────────────────────────────────────────
    n_frames = n_total
    if args.test > 0:
        frames, n_frames = frames[:args.test], args.test
        print(f"  [TEST] {n_frames} frames")
    print(f"  {n_frames} frames @ {fps:.1f}fps, {frames.shape[2]}×{frames.shape[1]}px")
    print(f"  RBC radius: {args.rbc_radius}px | motion lag: {args.motion_lag}f | dark_thresh: {args.dark_thresh}")

    h, w        = frames.shape[1], frames.shape[2]
    motion_buf  = MotionBuffer(lag=args.motion_lag)
    out_frames  = []
    prev_gray   = None

    for i in tqdm(range(n_frames), unit="frame"):
        frame = frames[i]
        gray, enhanced = preprocess(frame)

        stage_jump = detect_stage_jump(prev_gray, gray, args.stage_jump_threshold)
        prev_gray  = gray

        if stage_jump:
            motion_buf.reset()
            out = frame.copy()
            draw_legend(out, stage_jump=True)
            out_frames.append(out)
            continue

        motion_diff = motion_buf.motion_diff()   # before update → lag frames back
        motion_buf.update(gray)

        hough_circles = _find_hough(enhanced, args.rbc_radius)
        cell_bodies   = get_cell_bodies(enhanced, args.dark_thresh)
        rbc_zone      = get_rbc_zone(density, hough_circles, args.rbc_radius, h, w)
        rbc_mask      = watershed_rbcs(gray, cell_bodies, density, hough_circles, args.rbc_radius)
        neutro_mask   = detect_neutrophil(
            enhanced, gray, cell_bodies, rbc_zone, motion_diff,
            args.rbc_radius, h, w
        )

        out = draw_overlay(frame, rbc_mask, neutro_mask)
        draw_legend(out, warming_up=(motion_diff is None))
        out_frames.append(out)

    print(f"\nWriting: {output_path}")
    iio.imwrite(str(output_path), np.stack(out_frames), plugin="pyav",
                codec="h264", fps=int(round(fps)), out_pixel_format="yuv420p")
    print(f"Done → {output_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
