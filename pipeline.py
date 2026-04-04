"""
Cell Slicer — Segmentation Pipeline v5
========================================
Identifies and marks cell boundaries in microscopy video:
  - Red Blood Cells (RBCs)  → RED overlay
  - Neutrophil              → GREEN overlay

Correction-first design
───────────────────────
Manual corrections (corrections.json) are the primary signal. The pipeline
explicitly uses them to drive segmentation. Image analysis fills the gaps.

Stage jumps partition the video into segments. Corrections NEVER propagate
across a stage jump — the viewer trailing effect is bounded by jump frames.

Neutrophil detection (per frame, in priority order):
  1. Tier 1 — DIRECT: frame has neutrophil correction points → GrabCut
  2. Tier 2 — GUIDED: nearest corrected frame in same segment → GrabCut
     guided by interpolated centroid + RBC density seeds
  3. Tier 3 — MOTION: inter-frame diff fallback (weakest, no corrections nearby)

RBC detection: stable Hough density map + per-frame Hough → watershed.

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
GRABCUT_SEED_R        = 8
GRABCUT_ITER          = 5
MOTION_LAG            = 8
STAGE_JUMP_THRESH     = 18.0
MIN_NEUTROPHIL_PX     = 500


# ─── Preprocessing ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return gray, _clahe.apply(gray)


# ─── Stage jump detection ──────────────────────────────────────────────────────

def find_stage_jumps(frames, threshold=STAGE_JUMP_THRESH):
    """Return set of frame indices that are stage jumps."""
    jumps = set()
    prev = None
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            if float(cv2.absdiff(gray, prev).astype(np.float32).mean()) > threshold:
                jumps.add(i)
        prev = gray
    return jumps


def segment_intervals(n_frames, jump_frames):
    """
    Partition [0, n_frames) into contiguous segments separated by jump frames.
    Returns list of (start, end) inclusive ranges with no jumps inside.
    """
    boundaries = sorted(jump_frames)
    segs = []
    start = 0
    for j in boundaries:
        if j > start:
            segs.append((start, j - 1))
        start = j + 1          # jump frame starts a new segment alone
        segs.append((j, j))    # the jump frame is its own singleton
    if start < n_frames:
        segs.append((start, n_frames - 1))
    return segs


def frame_segment(frame_idx, segments):
    """Return (start, end) of the segment containing frame_idx."""
    for s, e in segments:
        if s <= frame_idx <= e:
            return s, e
    return frame_idx, frame_idx


# ─── Corrections ───────────────────────────────────────────────────────────────

def load_corrections(path):
    with open(path) as f:
        data = json.load(f)
    valid = [m for m in data
             if {'frame','x','y','label'} <= set(m.keys())
             and m['label'] in ('neutrophil','rbc','background')]
    print(f"  Loaded {len(valid)} corrections from {path}")
    return valid


def corrections_in_segment(corrections, seg_start, seg_end):
    """Corrections whose frame falls within [seg_start, seg_end]."""
    return [c for c in corrections if seg_start <= c['frame'] <= seg_end]


def corrections_for_frame(corrections, frame_idx):
    return [c for c in corrections if c['frame'] == frame_idx]


# ─── Centroid helpers ──────────────────────────────────────────────────────────

def centroid_from_corrections(corrections, frame_idx):
    pts = [(c['x'], c['y']) for c in corrections
           if c['frame'] == frame_idx and c['label'] == 'neutrophil']
    if not pts:
        return None
    return int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))


def build_centroid_table(corrections, segments):
    """
    Pre-compute neutrophil centroids for every annotated frame,
    organised by segment. Returns {frame_idx: (cx,cy)}.
    """
    table = {}
    annotated = sorted(set(c['frame'] for c in corrections if c['label'] == 'neutrophil'))
    for af in annotated:
        c = centroid_from_corrections(corrections, af)
        if c:
            table[af] = c
    return table


def interpolate_centroid(centroid_table, frame_idx, seg_start, seg_end):
    """
    Linearly interpolate/extrapolate centroid for frame_idx, but only using
    known frames within [seg_start, seg_end].
    Returns (cx, cy) or None if no known frame in segment.
    """
    known = {f: c for f, c in centroid_table.items() if seg_start <= f <= seg_end}
    if not known:
        return None
    if frame_idx in known:
        return known[frame_idx]
    frames = sorted(known)
    before = [f for f in frames if f < frame_idx]
    after  = [f for f in frames if f > frame_idx]
    if before and after:
        f0, f1 = max(before), min(after)
        t = (frame_idx - f0) / (f1 - f0)
        x0, y0 = known[f0];  x1, y1 = known[f1]
        return int(x0 + t*(x1-x0)), int(y0 + t*(y1-y0))
    elif before:
        return known[max(before)]
    else:
        return known[min(after)]


# ─── GrabCut ───────────────────────────────────────────────────────────────────

def grabcut_neutrophil(frame_bgr, enhanced, neutro_seeds, bg_seeds, rbc_seeds):
    h, w = enhanced.shape
    if not neutro_seeds:
        return np.zeros((h, w), np.uint8)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[enhanced > 190] = cv2.GC_BGD

    for x, y in neutro_seeds:
        cv2.circle(gc, (int(x), int(y)), GRABCUT_SEED_R, cv2.GC_FGD, -1)
    for x, y in bg_seeds:
        cv2.circle(gc, (int(x), int(y)), GRABCUT_SEED_R, cv2.GC_BGD, -1)
    for x, y in rbc_seeds:
        cv2.circle(gc, (int(x), int(y)), GRABCUT_SEED_R, cv2.GC_PR_BGD, -1)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame_bgr, gc, None, bgd, fgd, GRABCUT_ITER, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return np.zeros((h, w), np.uint8)

    raw = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Keep single largest component only — neutrophil is always one region
    n_c, comp_labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n_c <= 1:
        return np.zeros((h, w), np.uint8)
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if stats[best, cv2.CC_STAT_AREA] < MIN_NEUTROPHIL_PX:
        return np.zeros((h, w), np.uint8)
    mask = ((comp_labels == best) * 255).astype(np.uint8)
    return ndimage.binary_fill_holes(mask).astype(np.uint8) * 255


# ─── Neutrophil: Tier-1 (direct corrections) ──────────────────────────────────

def neutrophil_tier1(frame_bgr, enhanced, frame_corrs):
    neutro = [(c['x'], c['y']) for c in frame_corrs if c['label'] == 'neutrophil']
    bg     = [(c['x'], c['y']) for c in frame_corrs if c['label'] == 'background']
    rbc    = [(c['x'], c['y']) for c in frame_corrs if c['label'] == 'rbc']
    return grabcut_neutrophil(frame_bgr, enhanced, neutro, bg, rbc)


# ─── Neutrophil: Tier-2 (centroid-guided) ─────────────────────────────────────

def neutrophil_tier2(frame_bgr, enhanced, centroid, density, h, w):
    cx, cy = int(np.clip(centroid[0], 0, w-1)), int(np.clip(centroid[1], 0, h-1))

    # Tight FG seeds around centroid
    neutro_seeds = [(cx+dx, cy+dy)
                    for dx in range(-6, 7, 6)
                    for dy in range(-6, 7, 6)
                    if 0 <= cx+dx < w and 0 <= cy+dy < h]

    # BG: bright pixels
    by, bx = np.where(enhanced > 185)
    if len(by) > 30:
        idx = np.random.choice(len(by), 30, replace=False)
        bg_seeds = list(zip(bx[idx].tolist(), by[idx].tolist()))
    else:
        bg_seeds = list(zip(bx.tolist(), by.tolist()))

    # BG: confirmed RBC positions from density map
    ry, rx = np.where(density >= 0.20)
    if len(ry) > 40:
        idx = np.random.choice(len(ry), 40, replace=False)
        rbc_seeds = list(zip(rx[idx].tolist(), ry[idx].tolist()))
    else:
        rbc_seeds = list(zip(rx.tolist(), ry.tolist()))

    mask = grabcut_neutrophil(frame_bgr, enhanced, neutro_seeds, bg_seeds, rbc_seeds)

    # Retry with wider seeds if centroid not covered
    if mask[cy, cx] == 0:
        wide = [(cx+dx, cy+dy)
                for dx in range(-14, 15, 7)
                for dy in range(-14, 15, 7)
                if 0 <= cx+dx < w and 0 <= cy+dy < h]
        mask = grabcut_neutrophil(frame_bgr, enhanced, wide, bg_seeds, rbc_seeds)
    if mask[cy, cx] == 0:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), 22, 255, -1)

    return mask


# ─── Neutrophil: Tier-3 (motion fallback) ─────────────────────────────────────

class MotionBuffer:
    def __init__(self, lag=MOTION_LAG):
        self.lag = lag
        self.buf = deque(maxlen=lag+1)
    def update(self, gray): self.buf.append(gray.copy())
    def reset(self):        self.buf.clear()
    def diff(self):
        if len(self.buf) < self.lag+1: return None
        return cv2.absdiff(self.buf[-1], self.buf[0])


def neutrophil_tier3(gray, enhanced, motion_diff, density, rbc_radius, h, w):
    if motion_diff is None:
        return np.zeros((h, w), np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    _, fg = cv2.threshold(motion_diff, 15, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k15, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k3)
    _, dark = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k9, iterations=2)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    cell_bodies = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)
    rbc_zone = (density >= 0.20).astype(np.uint8) * 255
    rbc_zone = cv2.dilate(rbc_zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(53,53)))
    cand = cv2.bitwise_and(fg, cell_bodies)
    cand = cv2.bitwise_and(cand, cv2.bitwise_not(rbc_zone))
    n_c, comp_labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    margin = 15
    min_area = np.pi * rbc_radius**2 * 1.5
    valid = [(stats[l,cv2.CC_STAT_AREA], l, cents[l])
             for l in range(1,n_c)
             if stats[l,cv2.CC_STAT_AREA] >= min_area
             and cents[l][0] > margin and cents[l][0] < w-margin
             and cents[l][1] > margin and cents[l][1] < h-margin]
    if not valid:
        return np.zeros((h,w), np.uint8)
    _, best_lbl, _ = max(valid, key=lambda x: x[0])
    seed = ((comp_labels==best_lbl)*255).astype(np.uint8)
    seed_d = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n_cb, cb_labels = cv2.connectedComponents(cell_bodies)
    touched = set(np.unique(cb_labels[seed_d>0])) - {0}
    if not touched: return np.zeros((h,w),np.uint8)
    out = np.zeros((h,w),np.uint8)
    for lbl in touched: out[cb_labels==lbl]=255
    return ndimage.binary_fill_holes(out).astype(np.uint8)*255


# ─── RBC segmentation ──────────────────────────────────────────────────────────

def build_density_map(frames, rbc_radius, density_path):
    h, w = frames.shape[1], frames.shape[2]
    counts = np.zeros((h,w), np.int32)
    dot_r  = 10
    print("Building RBC density map…")
    for frame in tqdm(frames, unit="frame"):
        _, enh = preprocess(frame)
        circles = _find_hough(enh, rbc_radius)
        layer = np.zeros((h,w), np.int32)
        if circles is not None:
            for (cx,cy,cr) in circles:
                if 0<=cy<h and 0<=cx<w:
                    cv2.circle(layer,(cx,cy),dot_r,1,-1)
        counts += layer
    density = counts.astype(np.float32) / len(frames)
    np.save(str(density_path), density)
    print(f"  Saved → {density_path}  (max={density.max():.3f})")
    return density


def _find_hough(enhanced, rbc_radius):
    bl = cv2.GaussianBlur(enhanced,(5,5),1.5)
    c  = cv2.HoughCircles(bl, cv2.HOUGH_GRADIENT, dp=1,
                          minDist=int(rbc_radius*1.4), param1=50, param2=13,
                          minRadius=int(rbc_radius*0.6), maxRadius=int(rbc_radius*1.2))
    return np.round(c[0]).astype(int) if c is not None else None


def watershed_rbcs(gray, density, hough_circles, rbc_radius, h, w):
    _, dark = cv2.threshold(_clahe.apply(gray), 100, 255, cv2.THRESH_BINARY_INV)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3, iterations=2)
    filled = ndimage.binary_fill_holes(closed>0).astype(np.uint8)*255
    cell_bodies = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)
    seed_mask = (density>=0.20).astype(np.uint8)*255
    if hough_circles is not None:
        for (cx,cy,cr) in hough_circles:
            if 0<=cy<h and 0<=cx<w:
                cv2.circle(seed_mask,(cx,cy),max(2,int(rbc_radius*0.22)),255,-1)
    n_s, s_labels = cv2.connectedComponents(seed_mask)
    markers = np.zeros((h,w), np.int32)
    for lbl in range(1,n_s): markers[s_labels==lbl] = lbl+1
    sure_bg = cv2.dilate(cell_bodies, k3, iterations=3)
    markers[sure_bg==0] = 1
    cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    rbc_min = np.pi*(rbc_radius*0.50)**2
    rbc_max = np.pi*(rbc_radius*1.60)**2
    rbc_mask = np.zeros((h,w), np.uint8)
    for lbl in range(2, int(markers.max())+1):
        region = (markers==lbl).astype(np.uint8)
        if rbc_min <= region.sum() <= rbc_max:
            rbc_mask |= region*255
    return rbc_mask


# ─── Apply RBC/BG corrections ──────────────────────────────────────────────────

def apply_rbc_bg_corrections(rbc_mask, neutro_mask, frame_corrs, h, w):
    for c in frame_corrs:
        px = int(np.clip(c['x'],0,w-1))
        py = int(np.clip(c['y'],0,h-1))
        disc = np.zeros((h,w), np.uint8)
        cv2.circle(disc,(px,py),18,255,-1)
        if c['label']=='rbc':
            rbc_mask    = cv2.bitwise_or(rbc_mask, disc)
            neutro_mask = cv2.bitwise_and(neutro_mask, cv2.bitwise_not(disc))
        elif c['label']=='background':
            rbc_mask    = cv2.bitwise_and(rbc_mask,    cv2.bitwise_not(disc))
            neutro_mask = cv2.bitwise_and(neutro_mask, cv2.bitwise_not(disc))
    return rbc_mask, neutro_mask


# ─── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame_rgb, rbc_mask, neutro_mask):
    out = frame_rgb.copy().astype(np.float32)
    def fill(img, mask, c):
        for ch,v in enumerate(c):
            img[:,:,ch] = np.where(mask>0, img[:,:,ch]*(1-ALPHA_FILL)+v*ALPHA_FILL, img[:,:,ch])
    fill(out, rbc_mask,    COLOUR_RBC_RGB)
    fill(out, neutro_mask, COLOUR_NEUTROPHIL_RGB)
    out = np.clip(out,0,255).astype(np.uint8)
    def contours(img, mask, c, t):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, c, t)
    contours(out, rbc_mask,    COLOUR_RBC_RGB,        1)
    contours(out, neutro_mask, COLOUR_NEUTROPHIL_RGB, 2)
    return out


def draw_legend(frame, tier=None, stage_jump=False):
    for i,(label,colour) in enumerate([("Neutrophil",COLOUR_NEUTROPHIL_RGB),("RBC",COLOUR_RBC_RGB)]):
        y = 12 + i*14
        cv2.circle(frame,(9,y-3),4,colour,-1)
        cv2.putText(frame,label,(17,y),cv2.FONT_HERSHEY_SIMPLEX,0.32,(255,255,255),1,cv2.LINE_AA)
    h,w = frame.shape[:2]
    if stage_jump:
        cv2.putText(frame,"STAGE MOVE",(w//2-42,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,255),1,cv2.LINE_AA)
    if tier:
        col = {1:(0,230,120),2:(0,180,230),3:(180,180,100)}.get(tier,(128,128,128))
        cv2.putText(frame,f"T{tier}",(w-20,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.3,col,1,cv2.LINE_AA)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",             default="source-movie/chase-original.mp4")
    parser.add_argument("--output",            default="output/chase-segmented.mp4")
    parser.add_argument("--density-map",       default="output/rbc_density.npy")
    parser.add_argument("--build-density-map", action="store_true")
    parser.add_argument("--corrections",       default=None)
    parser.add_argument("--rbc-radius",        type=float, default=17.0)
    parser.add_argument("--stage-jump-threshold", type=float, default=STAGE_JUMP_THRESH)
    parser.add_argument("--test",              type=int,   default=0)
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

    density = np.load(str(density_path)) if density_path.exists() else \
              build_density_map(frames, args.rbc_radius, density_path)

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
    print(f"  {n_frames} frames @ {fps:.1f}fps  |  RBC radius: {args.rbc_radius}px")

    # ── Detect stage jumps and build segments ────────────────────────────────
    print("  Detecting stage jumps…")
    jump_frames = find_stage_jumps(frames, args.stage_jump_threshold)
    segments    = segment_intervals(n_frames, jump_frames)
    print(f"  {len(jump_frames)} jump(s) → {len(segments)} segment(s)")

    # ── Pre-compute neutrophil centroids from corrections ────────────────────
    centroid_table = build_centroid_table(corrections, segments)
    if centroid_table:
        print(f"  Centroids for frames: {sorted(centroid_table.keys())}")

    motion_buf = MotionBuffer(lag=MOTION_LAG)
    out_frames = []

    print(f"  Processing…")
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

        motion_diff = motion_buf.diff()
        motion_buf.update(gray)

        hough   = _find_hough(enh, args.rbc_radius)
        rbc_mask = watershed_rbcs(gray, density, hough, args.rbc_radius, h, w)

        # ── Determine segment for this frame ─────────────────────────────────
        seg_start, seg_end = frame_segment(i, segments)
        direct_corrs = corrections_for_frame(corrections, i)
        tier = None

        if direct_corrs and any(c['label']=='neutrophil' for c in direct_corrs):
            # Tier 1: direct GrabCut from annotations on this exact frame
            neutro_mask = neutrophil_tier1(frame_bgr, enh, direct_corrs)
            tier = 1

        else:
            # Tier 2: nearest known centroid within THIS segment only
            prior = interpolate_centroid(centroid_table, i, seg_start, seg_end)
            if prior:
                neutro_mask = neutrophil_tier2(frame_bgr, enh, prior, density, h, w)
                tier = 2
            else:
                # Tier 3: motion fallback (no corrections in this segment)
                neutro_mask = neutrophil_tier3(gray, enh, motion_diff, density,
                                               args.rbc_radius, h, w)
                tier = 3

        # Apply rbc/background corrections from this exact frame only
        rbc_mask, neutro_mask = apply_rbc_bg_corrections(
            rbc_mask, neutro_mask, direct_corrs, h, w
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
