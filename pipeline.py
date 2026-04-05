"""
Cell Slicer — Segmentation Pipeline v6
========================================
Correction-driven segmentation:
  - Neutrophil: GrabCut seeded from ALL correction points on the frame
    (or interpolated from neighbouring corrected frames within the same segment)
  - RBCs: disc masks centred on RBC correction points + watershed for boundaries
  - Corrections never propagate across stage jump frames

Usage:
    python3 pipeline.py [--build-density-map] [--corrections PATH] [options]
"""

import argparse, json
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
GRABCUT_ITER          = 5
MOTION_LAG            = 8
STAGE_JUMP_THRESH     = 18.0
MIN_NEUTROPHIL_PX     = 300
RBC_DISC_R            = 18     # radius of RBC correction discs in px


# ─── Preprocessing ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return gray, _clahe.apply(gray)


# ─── Stage jump detection ──────────────────────────────────────────────────────

def find_stage_jumps(frames, threshold=STAGE_JUMP_THRESH):
    jumps = set()
    prev = None
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            if float(cv2.absdiff(gray, prev).astype(np.float32).mean()) > threshold:
                jumps.add(i)
        prev = gray
    return jumps


def segment_for_frame(frame_idx, jump_frames, n_frames):
    """Return (seg_start, seg_end) for frame_idx, bounded by jump frames."""
    jumps = sorted(jump_frames)
    start = 0
    for j in jumps:
        if j > frame_idx:
            return start, j - 1
        start = j + 1
    return start, n_frames - 1


# ─── Corrections ───────────────────────────────────────────────────────────────

def load_corrections(path):
    with open(path) as f:
        data = json.load(f)
    valid = [m for m in data
             if {'frame','x','y','label'} <= set(m.keys())
             and m['label'] in ('neutrophil','rbc','background')]
    print(f"  Loaded {len(valid)} corrections ({sum(1 for m in valid if m['label']=='neutrophil')} neutrophil, "
          f"{sum(1 for m in valid if m['label']=='rbc')} rbc, "
          f"{sum(1 for m in valid if m['label']=='background')} background)")
    return valid


def corrections_for_frame(corrections, frame_idx):
    return [c for c in corrections if c['frame'] == frame_idx]


def annotated_frames_in_segment(corrections, seg_start, seg_end, label=None):
    """Sorted list of annotated frame indices within [seg_start, seg_end]."""
    frames = set()
    for c in corrections:
        if seg_start <= c['frame'] <= seg_end:
            if label is None or c['label'] == label:
                frames.add(c['frame'])
    return sorted(frames)


# ─── Interpolation helpers ─────────────────────────────────────────────────────

def centroid_of_corrections(corrections, frame_idx, label):
    pts = [(c['x'], c['y']) for c in corrections
           if c['frame'] == frame_idx and c['label'] == label]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def interpolate_points(corrections, frame_idx, label, seg_start, seg_end):
    """
    Return interpolated seed points for `label` at `frame_idx`.
    Finds the nearest annotated frames on either side within the segment,
    and linearly shifts the correction points by the centroid displacement.
    """
    ann = annotated_frames_in_segment(corrections, seg_start, seg_end, label=label)
    if not ann:
        return []

    if frame_idx in ann:
        return [(c['x'], c['y']) for c in corrections
                if c['frame'] == frame_idx and c['label'] == label]

    before = [f for f in ann if f < frame_idx]
    after  = [f for f in ann if f > frame_idx]

    if before and after:
        f0, f1 = max(before), min(after)
        t = (frame_idx - f0) / (f1 - f0)
    elif before:
        f0, f1, t = max(before), max(before), 1.0
    else:
        f0, f1, t = min(after), min(after), 0.0

    # Reference frame = whichever is closer
    ref_frame = f0 if (frame_idx - f0 <= f1 - frame_idx) else f1
    ref_t     = 0.0 if ref_frame == f0 else 1.0

    c0 = centroid_of_corrections(corrections, f0, label)
    c1 = centroid_of_corrections(corrections, f1, label) if f0 != f1 else c0
    if c0 is None or c1 is None:
        return []

    # Interpolate centroid displacement
    interp_cx = c0[0] + t * (c1[0] - c0[0])
    interp_cy = c0[1] + t * (c1[1] - c0[1])
    ref_cx, ref_cy = (c0 if ref_frame == f0 else c1)

    dx = interp_cx - ref_cx
    dy = interp_cy - ref_cy

    # Shift all correction points from the reference frame
    ref_pts = [(c['x'], c['y']) for c in corrections
               if c['frame'] == ref_frame and c['label'] == label]
    return [(x + dx, y + dy) for x, y in ref_pts]


# ─── GrabCut ───────────────────────────────────────────────────────────────────

def grabcut_neutrophil(frame_bgr, enhanced, neutro_pts, bg_pts, rbc_pts,
                       seed_r=8, n_iter=GRABCUT_ITER):
    """
    GrabCut with full correction point sets as seeds.
    Returns single-largest-component binary mask.
    """
    h, w = enhanced.shape
    if not neutro_pts:
        return np.zeros((h, w), np.uint8)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[enhanced > 190] = cv2.GC_BGD

    for x, y in neutro_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        cv2.circle(gc, (xi, yi), seed_r, cv2.GC_FGD, -1)
    for x, y in bg_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        cv2.circle(gc, (xi, yi), seed_r, cv2.GC_BGD, -1)
    for x, y in rbc_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        cv2.circle(gc, (xi, yi), seed_r, cv2.GC_PR_BGD, -1)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame_bgr, gc, None, bgd, fgd, n_iter, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return np.zeros((h, w), np.uint8)

    raw = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Keep only largest connected component ≥ MIN_NEUTROPHIL_PX
    n_c, lbl, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n_c <= 1:
        return np.zeros((h, w), np.uint8)
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if stats[best, cv2.CC_STAT_AREA] < MIN_NEUTROPHIL_PX:
        return np.zeros((h, w), np.uint8)
    mask = ((lbl == best) * 255).astype(np.uint8)
    return ndimage.binary_fill_holes(mask).astype(np.uint8) * 255


# ─── Neutrophil segmentation ───────────────────────────────────────────────────

def segment_neutrophil(frame_bgr, enhanced, corrections, frame_idx,
                       seg_start, seg_end, density, h, w):
    """
    Tier 1 if direct neutrophil corrections exist for this frame.
    Tier 2 if interpolated corrections available within segment.
    Tier 3 (empty) otherwise.
    """
    # Gather seed points
    neutro_pts = interpolate_points(corrections, frame_idx, 'neutrophil', seg_start, seg_end)
    bg_pts     = interpolate_points(corrections, frame_idx, 'background', seg_start, seg_end)
    rbc_pts    = interpolate_points(corrections, frame_idx, 'rbc',        seg_start, seg_end)

    direct = any(c['frame'] == frame_idx and c['label'] == 'neutrophil'
                 for c in corrections)
    tier   = 1 if direct else (2 if neutro_pts else 3)

    if not neutro_pts:
        return np.zeros((h, w), np.uint8), 3

    # Supplement BG seeds with high-density RBC map
    ry, rx = np.where(density >= 0.20)
    if len(ry) > 30:
        idx = np.random.choice(len(ry), 30, replace=False)
        rbc_pts = rbc_pts + list(zip(rx[idx].tolist(), ry[idx].tolist()))

    mask = grabcut_neutrophil(frame_bgr, enhanced, neutro_pts, bg_pts, rbc_pts)

    # Sanity: mask should cover at least one seed point
    for x, y in neutro_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        if mask[yi, xi] > 0:
            return mask, tier

    # If not, fall back to painted seed discs filled in
    fallback = np.zeros((h, w), np.uint8)
    for x, y in neutro_pts:
        cv2.circle(fallback, (int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))), 12, 255, -1)
    fallback = ndimage.binary_fill_holes(fallback > 0).astype(np.uint8) * 255
    return fallback, tier


# ─── RBC segmentation ──────────────────────────────────────────────────────────

def build_density_map(frames, rbc_radius, density_path):
    h, w   = frames.shape[1], frames.shape[2]
    counts = np.zeros((h, w), np.int32)
    print("Building RBC density map…")
    for frame in tqdm(frames, unit="frame"):
        _, enh = preprocess(frame)
        circles = _find_hough(enh, rbc_radius)
        layer = np.zeros((h, w), np.int32)
        if circles is not None:
            for (cx, cy, cr) in circles:
                if 0 <= cy < h and 0 <= cx < w:
                    cv2.circle(layer, (cx, cy), 10, 1, -1)
        counts += layer
    density = counts.astype(np.float32) / len(frames)
    np.save(str(density_path), density)
    print(f"  Saved → {density_path}  (max={density.max():.3f})")
    return density


def _find_hough(enhanced, rbc_radius):
    bl = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    c  = cv2.HoughCircles(bl, cv2.HOUGH_GRADIENT, dp=1,
                          minDist=int(rbc_radius*1.4), param1=50, param2=13,
                          minRadius=int(rbc_radius*0.6), maxRadius=int(rbc_radius*1.2))
    return np.round(c[0]).astype(int) if c is not None else None


def segment_rbcs(gray, enhanced, corrections, frame_idx, seg_start, seg_end,
                 density, hough_circles, rbc_radius, h, w):
    """
    RBC mask: interpolated RBC correction discs + Hough watershed.
    Background corrections erase from the mask.
    """
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Cell bodies for watershed
    _, dark = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    closed  = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3, iterations=2)
    filled  = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    cell_bodies = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)

    # Seeds from stable density + current Hough
    seed_mask = (density >= 0.20).astype(np.uint8) * 255
    if hough_circles is not None:
        for (cx, cy, cr) in hough_circles:
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(seed_mask, (cx, cy), max(2, int(rbc_radius*0.22)), 255, -1)

    n_s, s_labels = cv2.connectedComponents(seed_mask)
    markers = np.zeros((h, w), np.int32)
    for lbl in range(1, n_s):
        markers[s_labels == lbl] = lbl + 1

    sure_bg = cv2.dilate(cell_bodies, k3, iterations=3)
    markers[sure_bg == 0] = 1
    cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)

    rbc_min = np.pi * (rbc_radius * 0.50) ** 2
    rbc_max = np.pi * (rbc_radius * 1.60) ** 2
    rbc_mask = np.zeros((h, w), np.uint8)
    for lbl in range(2, int(markers.max()) + 1):
        region = (markers == lbl).astype(np.uint8)
        if rbc_min <= region.sum() <= rbc_max:
            rbc_mask |= region * 255

    # Paint RBC correction discs (interpolated) onto mask
    rbc_pts = interpolate_points(corrections, frame_idx, 'rbc', seg_start, seg_end)
    for x, y in rbc_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        cv2.circle(rbc_mask, (xi, yi), RBC_DISC_R, 255, -1)

    # Erase background corrections
    bg_pts = interpolate_points(corrections, frame_idx, 'background', seg_start, seg_end)
    for x, y in bg_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        cv2.circle(rbc_mask, (xi, yi), RBC_DISC_R, 0, -1)

    return rbc_mask


# ─── Motion buffer (Tier-3 fallback) ───────────────────────────────────────────

class MotionBuffer:
    def __init__(self, lag=MOTION_LAG):
        self.lag = lag
        self.buf = deque(maxlen=lag+1)
    def update(self, gray): self.buf.append(gray.copy())
    def reset(self):        self.buf.clear()
    def diff(self):
        if len(self.buf) < self.lag+1: return None
        return cv2.absdiff(self.buf[-1], self.buf[0])


# ─── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame_rgb, rbc_mask, neutro_mask):
    out = frame_rgb.copy().astype(np.float32)
    def fill(img, mask, c):
        for ch, v in enumerate(c):
            img[:,:,ch] = np.where(mask > 0,
                                   img[:,:,ch]*(1-ALPHA_FILL) + v*ALPHA_FILL,
                                   img[:,:,ch])
    fill(out, rbc_mask,    COLOUR_RBC_RGB)
    fill(out, neutro_mask, COLOUR_NEUTROPHIL_RGB)
    out = np.clip(out, 0, 255).astype(np.uint8)
    def draw_contours(img, mask, c, t):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, c, t)
    draw_contours(out, rbc_mask,    COLOUR_RBC_RGB,        1)
    draw_contours(out, neutro_mask, COLOUR_NEUTROPHIL_RGB, 2)
    return out


def draw_legend(frame, tier=None, stage_jump=False):
    for i, (label, colour) in enumerate([
            ("Neutrophil", COLOUR_NEUTROPHIL_RGB), ("RBC", COLOUR_RBC_RGB)]):
        y = 12 + i*14
        cv2.circle(frame, (9, y-3), 4, colour, -1)
        cv2.putText(frame, label, (17, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    if stage_jump:
        cv2.putText(frame, "STAGE MOVE", (w//2-42, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1, cv2.LINE_AA)
    if tier:
        col = {1:(0,230,120), 2:(0,180,230), 3:(180,180,100)}.get(tier, (128,128,128))
        cv2.putText(frame, f"T{tier}", (w-20, h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",               default="source-movie/chase-original.mp4")
    parser.add_argument("--output",              default="output/chase-segmented.mp4")
    parser.add_argument("--density-map",         default="output/rbc_density.npy")
    parser.add_argument("--build-density-map",   action="store_true")
    parser.add_argument("--corrections",         default=None)
    parser.add_argument("--rbc-radius",          type=float, default=17.0)
    parser.add_argument("--stage-jump-threshold",type=float, default=STAGE_JUMP_THRESH)
    parser.add_argument("--test",                type=int,   default=0)
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    density_path = Path(args.density_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = iio.imopen(str(input_path), "r", plugin="pyav")
    fps    = reader.metadata().get("fps", 15.0)
    reader.close()
    frames  = iio.imread(str(input_path), plugin="pyav", index=None)
    n_total = len(frames)

    if args.build_density_map:
        build_density_map(frames, args.rbc_radius, density_path)
        return

    density = (np.load(str(density_path)) if density_path.exists()
               else build_density_map(frames, args.rbc_radius, density_path))

    corrections = []
    if args.corrections and Path(args.corrections).exists():
        corrections = load_corrections(args.corrections)
    elif args.corrections:
        print(f"  WARNING: {args.corrections} not found")

    n_frames = n_total
    if args.test > 0:
        frames, n_frames = frames[:args.test], args.test
        print(f"  [TEST] {n_frames} frames")

    h, w = frames.shape[1], frames.shape[2]
    print(f"  {n_frames} frames @ {fps:.1f}fps  |  RBC radius={args.rbc_radius}px")

    print("  Detecting stage jumps…")
    jump_frames = find_stage_jumps(frames, args.stage_jump_threshold)
    print(f"  {len(jump_frames)} stage jump(s): {sorted(jump_frames)}")

    motion_buf = MotionBuffer(lag=MOTION_LAG)
    out_frames = []

    print("  Processing…")
    for i in tqdm(range(n_frames), unit="frame"):
        frame     = frames[i]
        gray, enh = preprocess(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        is_jump   = i in jump_frames

        if is_jump:
            motion_buf.reset()
            out = frame.copy()
            draw_legend(out, stage_jump=True)
            out_frames.append(out)
            continue

        motion_buf.update(gray)
        hough = _find_hough(enh, args.rbc_radius)

        # Segment bounds (no cross-jump propagation)
        seg_start, seg_end = segment_for_frame(i, jump_frames, n_frames)

        neutro_mask, tier = segment_neutrophil(
            frame_bgr, enh, corrections, i, seg_start, seg_end, density, h, w)

        rbc_mask = segment_rbcs(
            gray, enh, corrections, i, seg_start, seg_end,
            density, hough, args.rbc_radius, h, w)

        # Prevent overlap: neutrophil wins over RBC
        rbc_mask = cv2.bitwise_and(rbc_mask, cv2.bitwise_not(neutro_mask))

        out = draw_overlay(frame, rbc_mask, neutro_mask)
        draw_legend(out, tier=tier)
        out_frames.append(out)

    print(f"\nWriting: {output_path}")
    iio.imwrite(str(output_path), np.stack(out_frames), plugin="pyav",
                codec="h264", fps=int(round(fps)), out_pixel_format="yuv420p")
    print(f"Done → {output_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
