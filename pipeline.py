"""
Cell Slicer — Segmentation Pipeline v4
========================================
Identifies and marks cell boundaries in microscopy video:
  - Red Blood Cells (RBCs)  → RED overlay
  - Neutrophil              → GREEN overlay

Segmentation strategy:
──────────────────────
RBCs (all frames):
  Stable Hough circle density map (accumulated across all frames) + per-frame
  Hough → marker-controlled watershed for actual cell-boundary shapes.

Neutrophil — three tiers based on available information:

  Tier 1 — DIRECT (frames with correction points):
    GrabCut seeded by the annotated neutrophil/background/rbc points.
    Gives ~80-90% accuracy on corrected frames.

  Tier 2 — GUIDED (frames within propagation window of a correction):
    Nearest corrected frame's neutrophil centroid is used as a positional prior.
    GrabCut is seeded with a disc around the predicted position + background seeds
    from bright pixels and known-RBC positions. Falls back gracefully if it fails.

  Tier 3 — MOTION (all other frames):
    Inter-frame diff (|frame_t - frame_{t-lag}|) → motion signal → subtract
    RBC zone → largest moving cell-body region. Weakest but still useful.

Stage jumps: motion buffer reset, correction propagation unaffected.

Usage:
    python3 pipeline.py [--build-density-map] [--corrections PATH] [options]
"""

import argparse
import json
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from collections import deque
from scipy import ndimage


# ─── Constants ─────────────────────────────────────────────────────────────────
COLOUR_RBC_RGB        = (220, 60,  60)
COLOUR_NEUTROPHIL_RGB = (40,  220, 40)
ALPHA_FILL            = 0.40

CORRECTION_WINDOW     = 8   # frames either side of a correction that Tier-2 applies
GRABCUT_SEED_R        = 8   # radius of seed discs for GrabCut
GRABCUT_ITER          = 5   # GrabCut iterations
MOTION_LAG            = 8   # inter-frame lag for motion diff


# ─── Preprocessing ─────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(frame_rgb):
    gray     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    enhanced = _clahe.apply(gray)
    return gray, enhanced


# ─── Stage jump ────────────────────────────────────────────────────────────────

def detect_stage_jump(prev_gray, curr_gray, threshold=18.0):
    if prev_gray is None:
        return False
    return float(cv2.absdiff(prev_gray, curr_gray).astype(np.float32).mean()) > threshold


# ─── Corrections ───────────────────────────────────────────────────────────────

def load_corrections(path):
    with open(path) as f:
        data = json.load(f)
    valid = [m for m in data if {'frame','x','y','label'} <= set(m.keys())
             and m['label'] in ('neutrophil','rbc','background')]
    print(f"  Loaded {len(valid)} corrections from {path}")
    return valid


def corrections_for_frame(corrections, frame_idx, window=0):
    """Corrections exactly on frame_idx, or within ±window if window>0."""
    return [c for c in corrections if abs(c['frame'] - frame_idx) <= window]


def nearest_correction_frame(corrections, frame_idx):
    """Return (distance, frame_number) of nearest annotated frame, or None."""
    frames = sorted(set(c['frame'] for c in corrections))
    if not frames:
        return None
    nearest = min(frames, key=lambda f: abs(f - frame_idx))
    return abs(nearest - frame_idx), nearest


# ─── GrabCut neutrophil segmentation ───────────────────────────────────────────

def grabcut_neutrophil(frame_bgr, enhanced, neutro_seeds, bg_seeds, rbc_seeds,
                       seed_r=GRABCUT_SEED_R, n_iter=GRABCUT_ITER):
    """
    Run GrabCut with explicit seed points.
    neutro_seeds: list of (x, y) → definite foreground
    bg_seeds:     list of (x, y) → definite background
    rbc_seeds:    list of (x, y) → probable background
    Returns binary mask (255=neutrophil).
    """
    h, w = enhanced.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Bright pixels → definite background
    gc_mask[enhanced > 190] = cv2.GC_BGD

    # Seed circles
    for (x, y) in neutro_seeds:
        cv2.circle(gc_mask, (int(x), int(y)), seed_r, cv2.GC_FGD, -1)
    for (x, y) in bg_seeds:
        cv2.circle(gc_mask, (int(x), int(y)), seed_r, cv2.GC_BGD, -1)
    for (x, y) in rbc_seeds:
        cv2.circle(gc_mask, (int(x), int(y)), seed_r, cv2.GC_PR_BGD, -1)

    if not neutro_seeds:
        return np.zeros((h, w), dtype=np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame_bgr, gc_mask, None, bgd_model, fgd_model,
                    n_iter, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Post-process: fill holes, remove small blobs
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)

    return mask


# ─── Density map (pass 1) ──────────────────────────────────────────────────────

def build_density_map(frames, rbc_radius, density_path):
    h, w    = frames.shape[1], frames.shape[2]
    counts  = np.zeros((h, w), np.int32)
    dot_r   = 10
    n_total = len(frames)
    print("Building RBC density map…")
    for frame in tqdm(frames, unit="frame"):
        _, enh  = preprocess(frame)
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
        minDist=int(rbc_radius * 1.4), param1=50, param2=13,
        minRadius=int(rbc_radius * 0.6), maxRadius=int(rbc_radius * 1.2),
    )
    return np.round(circles[0]).astype(int) if circles is not None else None


# ─── RBC segmentation ──────────────────────────────────────────────────────────

def get_rbc_zone(density, hough_circles, rbc_radius, h, w):
    rbc_stable   = (density >= 0.20).astype(np.uint8) * 255
    k_stable     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))
    rbc_expanded = cv2.dilate(rbc_stable, k_stable)
    rbc_hough    = np.zeros((h, w), np.uint8)
    if hough_circles is not None:
        for (cx, cy, cr) in hough_circles:
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(rbc_hough, (cx, cy), int(cr * 1.3), 255, -1)
    return cv2.bitwise_or(rbc_hough, rbc_expanded)


def watershed_rbcs(gray, density, hough_circles, rbc_radius, h, w):
    """Marker-controlled watershed for RBC boundary shapes."""
    # Cell bodies for watershed extent (lighter close kernel this time)
    _, dark = cv2.threshold(_clahe.apply(gray), 100, 255, cv2.THRESH_BINARY_INV)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed      = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3, iterations=2)
    filled      = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    cell_bodies = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)

    seed_mask = (density >= 0.20).astype(np.uint8) * 255
    if hough_circles is not None:
        for (cx, cy, cr) in hough_circles:
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(seed_mask, (cx, cy), max(2, int(rbc_radius * 0.22)), 255, -1)

    n_seeds, seed_labels = cv2.connectedComponents(seed_mask)
    markers = np.zeros((h, w), dtype=np.int32)
    for lbl in range(1, n_seeds):
        markers[seed_labels == lbl] = lbl + 1

    sure_bg = cv2.dilate(cell_bodies, k3, iterations=3)
    markers[sure_bg == 0] = 1
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_bgr, markers)

    rbc_min = np.pi * (rbc_radius * 0.50) ** 2
    rbc_max = np.pi * (rbc_radius * 1.60) ** 2
    rbc_mask = np.zeros((h, w), np.uint8)
    for lbl in range(2, int(markers.max()) + 1):
        region = (markers == lbl).astype(np.uint8)
        area   = int(region.sum())
        if rbc_min <= area <= rbc_max:
            rbc_mask |= region * 255
    return rbc_mask


# ─── Neutrophil: Tier-1 (direct corrections) ──────────────────────────────────

def neutrophil_from_corrections(frame_bgr, enhanced, frame_corrs):
    """Tier 1: GrabCut directly seeded by the user's correction points."""
    neutro_seeds = [(c['x'], c['y']) for c in frame_corrs if c['label'] == 'neutrophil']
    bg_seeds     = [(c['x'], c['y']) for c in frame_corrs if c['label'] == 'background']
    rbc_seeds    = [(c['x'], c['y']) for c in frame_corrs if c['label'] == 'rbc']
    return grabcut_neutrophil(frame_bgr, enhanced, neutro_seeds, bg_seeds, rbc_seeds)


# ─── Neutrophil: Tier-2 (position-guided) ─────────────────────────────────────

def neutrophil_centroid(mask):
    """Return (cx, cy) of a binary mask's centroid, or None."""
    m = cv2.moments(mask)
    if m['m00'] < 10:
        return None
    return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])


def neutrophil_guided(frame_bgr, enhanced, prior_centroid, rbc_zone, h, w):
    """
    Tier 2: GrabCut seeded at the predicted neutrophil position.
    prior_centroid: (cx, cy) from the nearest corrected frame.
    """
    cx, cy = prior_centroid
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    # FG seeds: disc around predicted centroid (radius ~1 RBC)
    neutro_seeds = []
    r = 22
    for dx in range(-r, r+1, 8):
        for dy in range(-r, r+1, 8):
            if dx*dx + dy*dy <= r*r:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < w and 0 <= ny < h:
                    neutro_seeds.append((nx, ny))

    # BG seeds: bright background pixels (sampled)
    bg_ys, bg_xs = np.where(enhanced > 190)
    if len(bg_ys) > 20:
        idx = np.random.choice(len(bg_ys), 20, replace=False)
        bg_seeds = list(zip(bg_xs[idx].tolist(), bg_ys[idx].tolist()))
    else:
        bg_seeds = []

    # RBC seeds: known RBC zone pixels (sampled)
    rbc_ys, rbc_xs = np.where(rbc_zone > 0)
    if len(rbc_ys) > 20:
        idx = np.random.choice(len(rbc_ys), 20, replace=False)
        rbc_seeds = list(zip(rbc_xs[idx].tolist(), rbc_ys[idx].tolist()))
    else:
        rbc_seeds = []

    mask = grabcut_neutrophil(frame_bgr, enhanced, neutro_seeds, bg_seeds, rbc_seeds)

    # Sanity check: result must overlap predicted position
    if mask[cy, cx] == 0:
        # GrabCut drifted — fall back to a simple disc at the predicted position
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), 28, 255, -1)

    return mask


# ─── Neutrophil: Tier-3 (motion fallback) ─────────────────────────────────────

class MotionBuffer:
    def __init__(self, lag=MOTION_LAG):
        self.lag    = lag
        self.buffer = deque(maxlen=lag + 1)

    def update(self, gray): self.buffer.append(gray.copy())
    def reset(self):        self.buffer.clear()

    def motion_diff(self):
        if len(self.buffer) < self.lag + 1:
            return None
        return cv2.absdiff(self.buffer[-1], self.buffer[0])


def neutrophil_motion(gray, enhanced, motion_diff, rbc_zone, rbc_radius, h, w):
    """Tier 3: largest moving non-RBC blob."""
    if motion_diff is None:
        return np.zeros((h, w), np.uint8)

    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    _, fg = cv2.threshold(motion_diff, 15, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_big, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k3)

    # Cell bodies (light close only)
    _, dark = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k9, iterations=2)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    cell_bodies = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)

    candidate = cv2.bitwise_and(fg, cell_bodies)
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(rbc_zone))

    n_comp, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(candidate)
    margin = 15
    min_area = np.pi * rbc_radius ** 2 * 1.5
    valid = [
        (stats[l, cv2.CC_STAT_AREA], l, centroids[l])
        for l in range(1, n_comp)
        if stats[l, cv2.CC_STAT_AREA] >= min_area
        and centroids[l][0] > margin and centroids[l][0] < w - margin
        and centroids[l][1] > margin and centroids[l][1] < h - margin
    ]
    if not valid:
        return np.zeros((h, w), np.uint8)

    _, best_lbl, best_cent = max(valid, key=lambda x: x[0])
    seed = ((comp_labels == best_lbl) * 255).astype(np.uint8)
    seed_dil = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    n_cb, cb_labels = cv2.connectedComponents(cell_bodies)
    touched = set(np.unique(cb_labels[seed_dil > 0])) - {0}
    if not touched:
        return np.zeros((h, w), np.uint8)
    neutro = np.zeros((h, w), np.uint8)
    for lbl in touched:
        neutro[cb_labels == lbl] = 255
    return ndimage.binary_fill_holes(neutro).astype(np.uint8) * 255


# ─── Apply RBC corrections ─────────────────────────────────────────────────────

def apply_rbc_bg_corrections(rbc_mask, neutro_mask, enhanced, frame_corrs, h, w):
    """
    Apply rbc/background corrections directly.
    'rbc' corrections: force a disc at that point into rbc_mask, out of neutro.
    'background' corrections: erase a disc from both masks.
    """
    disc_r = 18
    for c in frame_corrs:
        px, py = int(np.clip(c['x'], 0, w-1)), int(np.clip(c['y'], 0, h-1))
        disc = np.zeros((h, w), np.uint8)
        cv2.circle(disc, (px, py), disc_r, 255, -1)
        if c['label'] == 'rbc':
            rbc_mask    = cv2.bitwise_or(rbc_mask, disc)
            neutro_mask = cv2.bitwise_and(neutro_mask, cv2.bitwise_not(disc))
        elif c['label'] == 'background':
            rbc_mask    = cv2.bitwise_and(rbc_mask,    cv2.bitwise_not(disc))
            neutro_mask = cv2.bitwise_and(neutro_mask, cv2.bitwise_not(disc))
    return rbc_mask, neutro_mask


# ─── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame_rgb, rbc_mask, neutro_mask):
    result = frame_rgb.copy().astype(np.float32)
    def fill(img, mask, colour):
        for c, val in enumerate(colour):
            img[:,:,c] = np.where(mask>0, img[:,:,c]*(1-ALPHA_FILL)+val*ALPHA_FILL, img[:,:,c])
    fill(result, rbc_mask,    COLOUR_RBC_RGB)
    fill(result, neutro_mask, COLOUR_NEUTROPHIL_RGB)
    result = np.clip(result, 0, 255).astype(np.uint8)
    def outlines(img, mask, colour, thickness):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colour, thickness)
    outlines(result, rbc_mask,    COLOUR_RBC_RGB,        1)
    outlines(result, neutro_mask, COLOUR_NEUTROPHIL_RGB, 2)
    return result


def draw_legend(frame, tier=None, stage_jump=False):
    items = [("Neutrophil", COLOUR_NEUTROPHIL_RGB), ("RBC", COLOUR_RBC_RGB)]
    h, w  = frame.shape[:2]
    x, y  = 5, 12
    for label, colour in items:
        cv2.circle(frame, (x+4, y-3), 4, colour, -1)
        cv2.putText(frame, label, (x+12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1, cv2.LINE_AA)
        y += 14
    if stage_jump:
        cv2.putText(frame, "STAGE MOVE", (w//2-42, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1, cv2.LINE_AA)
    if tier:
        tier_col = {1: (0,230,120), 2: (0,180,230), 3: (180,180,100)}.get(tier, (128,128,128))
        cv2.putText(frame, f"T{tier}", (w-20, h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, tier_col, 1, cv2.LINE_AA)
    return frame


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           default="source-movie/chase-original.mp4")
    parser.add_argument("--output",          default="output/chase-segmented.mp4")
    parser.add_argument("--density-map",     default="output/rbc_density.npy")
    parser.add_argument("--build-density-map", action="store_true")
    parser.add_argument("--corrections",     default=None)
    parser.add_argument("--rbc-radius",      type=float, default=17.0)
    parser.add_argument("--stage-jump-threshold", type=float, default=18.0)
    parser.add_argument("--test",            type=int, default=0)
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    density_path = Path(args.density_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    reader = iio.imopen(str(input_path), "r", plugin="pyav")
    fps    = reader.metadata().get("fps", 15.0)
    reader.close()
    frames  = iio.imread(str(input_path), plugin="pyav", index=None)
    n_total = len(frames)

    if args.build_density_map:
        build_density_map(frames, args.rbc_radius, density_path)
        return

    if density_path.exists():
        density = np.load(str(density_path))
    else:
        print("No density map — building now…")
        density = build_density_map(frames, args.rbc_radius, density_path)

    corrections = []
    if args.corrections and Path(args.corrections).exists():
        corrections = load_corrections(args.corrections)
    elif args.corrections:
        print(f"  WARNING: corrections file not found: {args.corrections}")

    n_frames = n_total
    if args.test > 0:
        frames, n_frames = frames[:args.test], args.test
        print(f"  [TEST] {n_frames} frames")
    print(f"  {n_frames} frames @ {fps:.1f}fps  |  RBC radius: {args.rbc_radius}px")

    h, w         = frames.shape[1], frames.shape[2]
    motion_buf   = MotionBuffer(lag=MOTION_LAG)
    out_frames   = []
    prev_gray    = None
    neutro_centroids = {}   # frame_idx → (cx, cy) for Tier-2 propagation

    print("Processing…")
    for i in tqdm(range(n_frames), unit="frame"):
        frame      = frames[i]
        gray, enh  = preprocess(frame)
        frame_bgr  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        stage_jump = detect_stage_jump(prev_gray, gray, args.stage_jump_threshold)
        prev_gray  = gray

        if stage_jump:
            motion_buf.reset()
            out = frame.copy()
            draw_legend(out, stage_jump=True)
            out_frames.append(out)
            continue

        motion_diff = motion_buf.motion_diff()
        motion_buf.update(gray)

        hough_circles = _find_hough(enh, args.rbc_radius)
        rbc_zone      = get_rbc_zone(density, hough_circles, args.rbc_radius, h, w)
        rbc_mask      = watershed_rbcs(gray, density, hough_circles, args.rbc_radius, h, w)

        # ── Determine neutrophil tier ────────────────────────────────────────
        direct_corrs = corrections_for_frame(corrections, i, window=0)
        tier = None

        if direct_corrs and any(c['label'] == 'neutrophil' for c in direct_corrs):
            # Tier 1: direct GrabCut
            neutro_mask = neutrophil_from_corrections(frame_bgr, enh, direct_corrs)
            tier = 1

        else:
            # Look for nearest correction frame within window
            nearby = corrections_for_frame(corrections, i, window=CORRECTION_WINDOW)
            nearby_neutro = [c for c in nearby if c['label'] == 'neutrophil']

            if nearby_neutro:
                # Tier 2: guided by nearest known centroid
                ref_result = nearest_correction_frame(corrections, i)
                if ref_result:
                    _, ref_frame = ref_result
                    if ref_frame in neutro_centroids:
                        prior = neutro_centroids[ref_frame]
                        neutro_mask = neutrophil_guided(frame_bgr, enh, prior, rbc_zone, h, w)
                        tier = 2
                    else:
                        # Fallback: seed from the nearby correction points directly
                        neutro_seeds = [(c['x'], c['y']) for c in nearby_neutro]
                        bg_seeds     = [(c['x'], c['y']) for c in nearby if c['label']=='background']
                        rbc_seeds_pt = [(c['x'], c['y']) for c in nearby if c['label']=='rbc']
                        neutro_mask  = grabcut_neutrophil(frame_bgr, enh, neutro_seeds, bg_seeds, rbc_seeds_pt)
                        tier = 2
                else:
                    neutro_mask = neutrophil_motion(gray, enh, motion_diff, rbc_zone, args.rbc_radius, h, w)
                    tier = 3
            else:
                # Tier 3: motion only
                neutro_mask = neutrophil_motion(gray, enh, motion_diff, rbc_zone, args.rbc_radius, h, w)
                tier = 3

        # Store centroid for Tier-2 propagation
        c = neutrophil_centroid(neutro_mask)
        if c:
            neutro_centroids[i] = c

        # Apply rbc/background corrections (disc-based, direct override)
        all_corrs = corrections_for_frame(corrections, i, window=CORRECTION_WINDOW)
        rbc_mask, neutro_mask = apply_rbc_bg_corrections(
            rbc_mask, neutro_mask, enh, all_corrs, h, w
        )

        out = draw_overlay(frame, rbc_mask, neutro_mask)
        draw_legend(out, tier=tier)
        out_frames.append(out)

    print(f"\nWriting: {output_path}")
    iio.imwrite(str(output_path), np.stack(out_frames), plugin="pyav",
                codec="h264", fps=int(round(fps)), out_pixel_format="yuv420p")
    print(f"Done → {output_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
