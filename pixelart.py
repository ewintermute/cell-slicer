#!/usr/bin/env python3
"""
pixelart.py  —  Cell Slicer pixel art generator
================================================
Produces a stylised 32×48 pixel art animation from the segmented masks.

Geometry:
  Block size : 5×5 source pixels per art pixel
  Crop       : centre 160 px of the 320-wide source  (x_offset = 80)
  Art canvas : 32 wide × 48 tall  (fits 240 px tall exactly)

Labels:  0 = background   1 = RBC interior   2 = RBC edge   3 = neutrophil

Output:  docs/pixel_art.json

Usage:
    python3 pixelart.py [--corrections PATH] [--output PATH]
"""

import argparse, json
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P

# ── Geometry ─────────────────────────────────────────────────────────────────
ART_W  = 32
ART_H  = 48
BLOCK  = 5
SRC_W  = 320
SRC_H  = 240
CROP_X = (SRC_W - ART_W * BLOCK) // 2   # 80
CROP_Y = (SRC_H - ART_H * BLOCK) // 2   # 0

# ── Labels ────────────────────────────────────────────────────────────────────
BG      = 0
RBC_INT = 1
RBC_EDG = 2
NEU     = 3

# ── RBC stylization parameters ────────────────────────────────────────────────
RBC_R        = 3   # fixed circle radius in art pixels (~28 art px ≈ full RBC)
RBC_MIN_AREA = 3   # min art-px component area to consider at all
RBC_STAB_WIN = 6   # frames each side for centroid stability window
RBC_STAB_MIN = 3   # centroid must match in ≥ this many window frames

# ── Neutrophil smoothing parameters ──────────────────────────────────────────
NEU_W     = 6     # temporal window radius (frames each side)
NEU_RATIO = 2.5   # area ratio vs neighbour median before smoothing kicks in


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_art(mask_u8: np.ndarray) -> np.ndarray:
    """
    Crop a 240×320 uint8 mask to 240×160 and downsample to 48×32 bool.
    Uses INTER_AREA (proper coverage averaging — majority-vote equivalent).
    """
    cropped = mask_u8[:, CROP_X: CROP_X + ART_W * BLOCK]   # 240×160
    small   = cv2.resize(cropped, (ART_W, ART_H), interpolation=cv2.INTER_AREA)
    return small > 127                                        # 48×32 bool


def get_centroids(art_bool: np.ndarray):
    """List of (cx, cy) for each qualifying connected component."""
    n_c, _, stats, _ = cv2.connectedComponentsWithStats(
        art_bool.astype(np.uint8) * 255)
    out = []
    for l in range(1, n_c):
        if stats[l, cv2.CC_STAT_AREA] >= RBC_MIN_AREA:
            out.append((
                int(stats[l, cv2.CC_STAT_LEFT] + stats[l, cv2.CC_STAT_WIDTH]  // 2),
                int(stats[l, cv2.CC_STAT_TOP]  + stats[l, cv2.CC_STAT_HEIGHT] // 2),
            ))
    return out


def near(ax, ay, bx, by, r):
    return (ax - bx) ** 2 + (ay - by) ** 2 <= r * r


def paint_circle(canvas: np.ndarray, cy: int, cx: int, r: int = RBC_R):
    """Fill a solid disc of radius r at art-pixel (cy, cx)."""
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < ART_H and 0 <= nx < ART_W:
                    canvas[ny, nx] = True


def seg_of(fi, jumps, n):
    return P.segment_for_frame(fi, jumps, n)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corrections', default='output/corrections.json')
    ap.add_argument('--output',      default='docs/pixel_art.json')
    ap.add_argument('--input',       default='source-movie/chase-original.mp4')
    ap.add_argument('--density-map', default='output/rbc_density.npy')
    args = ap.parse_args()

    frames  = iio.imread(args.input, plugin='pyav', index=None)
    N       = len(frames)
    H, W    = frames.shape[1], frames.shape[2]
    density = np.load(args.density_map)
    corr    = P.load_corrections(args.corrections)
    jumps   = P.find_stage_jumps(frames)
    print(f"  {N} frames  |  {len(jumps)} jump frames  |  "
          f"art grid {ART_W}×{ART_H}  block {BLOCK}px  crop_x={CROP_X}")

    # ── Pass 1: source-resolution segmentation masks ─────────────────────────
    print("\nPass 1: segmenting all frames…")
    raw_masks = []
    for i in tqdm(range(N), unit='frame'):
        if i in jumps:
            z = np.zeros((H, W), np.uint8)
            raw_masks.append((z, z))
            continue
        frame     = frames[i]
        _, enh    = P.preprocess(frame)
        fbgr      = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ss, se    = seg_of(i, jumps, N)
        hou       = P._find_hough(enh, 17.0)
        cb        = P.get_cell_bodies(enh)
        nm, _     = P.segment_neutrophil(fbgr, enh, corr, i, ss, se, density, H, W)
        rm        = P.segment_rbcs(enh, cb, corr, i, ss, se, density, hou, 17.0, H, W)
        rm        = cv2.bitwise_and(rm, cv2.bitwise_not(nm))
        raw_masks.append((nm, rm))

    # ── Pass 2: temporal smoothing on source masks ────────────────────────────
    print("\nPass 2: temporal smoothing…")
    smooth_masks = P.smooth_masks_temporally(raw_masks, jumps, N)

    # ── Downsample to art space ───────────────────────────────────────────────
    print("\nConverting to art space…")
    art_neu = []
    art_rbc = []
    for i in tqdm(range(N), unit='frame'):
        nm, rm = smooth_masks[i]
        art_neu.append(to_art(nm))
        art_rbc.append(to_art(rm))

    # ── RBC centroid extraction ───────────────────────────────────────────────
    print("\nExtracting RBC centroids…")
    raw_cents = [get_centroids(m) for m in art_rbc]

    # ── Centroid stability filter ─────────────────────────────────────────────
    # A centroid passes if it has a nearby match (within RBC_R+2 art px) in
    # ≥ RBC_STAB_MIN frames of its ±RBC_STAB_WIN window (same segment).
    # This kills single- and short-burst ghost RBCs that blip through stylization.
    print("\nFiltering unstable centroids…")
    MATCH_R = RBC_R + 2

    stable_cents = []
    for fi in tqdm(range(N), unit='frame'):
        if fi in jumps:
            stable_cents.append([])
            continue
        ss = seg_of(fi, jumps, N)[0]
        window = [j for d in range(1, RBC_STAB_WIN + 1)
                  for j in [fi - d, fi + d]
                  if 0 <= j < N and j not in jumps
                  and seg_of(j, jumps, N)[0] == ss]
        kept = []
        for cx, cy in raw_cents[fi]:
            hits = sum(1 for j in window
                       if any(near(cx, cy, nx, ny, MATCH_R)
                              for nx, ny in raw_cents[j]))
            if hits >= RBC_STAB_MIN:
                kept.append((cx, cy))
        stable_cents.append(kept)

    # ── Build stylised RBC art masks ──────────────────────────────────────────
    print("\nPainting stylised RBC circles…")
    sty_rbc = []
    for fi in tqdm(range(N), unit='frame'):
        canvas = np.zeros((ART_H, ART_W), bool)
        for cx, cy in stable_cents[fi]:
            paint_circle(canvas, cy, cx)
        sty_rbc.append(canvas)

    # ── Neutrophil temporal smoothing in art space ────────────────────────────
    # For frames where area == 0 OR area deviates > NEU_RATIO from the rolling
    # neighbour median, replace with a blended average of neighbour masks.
    print("\nSmoothing neutrophil…")
    neu_areas = [int(m.sum()) for m in art_neu]
    sty_neu   = list(art_neu)

    for fi in range(N):
        if fi in jumps:
            continue
        ss = seg_of(fi, jumps, N)[0]
        nbrs = [j for d in range(1, NEU_W + 1)
                for j in [fi - d, fi + d]
                if 0 <= j < N and j not in jumps
                and seg_of(j, jumps, N)[0] == ss]
        if len(nbrs) < 2:
            continue
        med = float(np.median([neu_areas[j] for j in nbrs]))
        if med < 1:
            continue
        ratio = neu_areas[fi] / med if med > 0 else 0
        if ratio > NEU_RATIO or ratio < 1.0 / NEU_RATIO or neu_areas[fi] == 0:
            stack   = np.stack([art_neu[j].astype(float) for j in nbrs])
            blended = stack.mean(axis=0)
            sty_neu[fi] = blended > 0.40

    # ── Compose final art frames ──────────────────────────────────────────────
    print("\nComposing…")
    k3 = np.ones((3, 3), np.uint8)
    out_frames = []
    for fi in range(N):
        art = np.full((ART_H, ART_W), BG, np.uint8)

        # RBC: fill then mark edges (eroded interior = everything else is edge)
        rbc = sty_rbc[fi]
        art[rbc] = RBC_INT
        eroded = cv2.erode(rbc.astype(np.uint8) * 255, k3)
        art[rbc & (eroded == 0)] = RBC_EDG

        # Neutrophil overwrites RBC entirely
        art[sty_neu[fi].astype(bool)] = NEU

        out_frames.append(art.flatten().tolist())

    # ── Stats ─────────────────────────────────────────────────────────────────
    non_bg = sum(sum(1 for v in f if v != BG) for f in out_frames)
    avg_rbc_discs = np.mean([len(c) for c in stable_cents])
    print(f"\n  Average stable RBC centroids/frame : {avg_rbc_discs:.1f}")
    print(f"  Non-background art pixels (total)  : {non_bg:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        'width':      ART_W,
        'height':     ART_H,
        'n_frames':   N,
        'crop_x':     CROP_X,
        'crop_y':     CROP_Y,
        'block_size': BLOCK,
        'fps':        15,
        'labels':     {'bg': BG, 'rbc_int': RBC_INT, 'rbc_edg': RBC_EDG, 'neu': NEU},
        'frames':     out_frames,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(',', ':')))
    size_kb = out.stat().st_size // 1024
    print(f"\nDone → {out}  ({N} frames × {ART_W}×{ART_H}, {size_kb} KB)")


if __name__ == '__main__':
    main()
