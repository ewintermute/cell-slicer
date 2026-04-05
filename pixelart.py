#!/usr/bin/env python3
"""
pixelart.py  —  Cell Slicer pixel art generator  v2
=====================================================
Produces a stylised 64×48 pixel art animation (landscape, full source frame).

Geometry:
  Block size : 5×5 source pixels per art pixel
  Crop       : none — full 320×240 source maps to 64×48 art pixels
  Jump frames: skipped entirely — not included in output

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
ART_W  = 64
ART_H  = 48
BLOCK  = 5
SRC_W  = 320
SRC_H  = 240

# ── Labels ────────────────────────────────────────────────────────────────────
BG      = 0
RBC_INT = 1
RBC_EDG = 2
NEU     = 3

# ── RBC stylization parameters ────────────────────────────────────────────────
RBC_R        = 3   # fixed circle radius in art pixels
RBC_MIN_AREA = 3   # min art-px component area to consider
RBC_STAB_WIN = 6   # frames each side for stability window
RBC_STAB_MIN = 3   # must match in ≥ this many window frames

# ── Neutrophil smoothing parameters ──────────────────────────────────────────
NEU_W     = 6
NEU_RATIO = 2.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_art(mask_u8: np.ndarray) -> np.ndarray:
    """Downsample 240×320 uint8 mask to 48×64 bool via area averaging."""
    small = cv2.resize(mask_u8, (ART_W, ART_H), interpolation=cv2.INTER_AREA)
    return small > 127


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
    N_total = len(frames)
    H, W    = frames.shape[1], frames.shape[2]
    density = np.load(args.density_map)
    corr    = P.load_corrections(args.corrections)
    jumps   = P.find_stage_jumps(frames)

    # Build ordered list of non-jump frame indices
    good_frames = [i for i in range(N_total) if i not in jumps]
    N = len(good_frames)

    print(f"  Source: {N_total} frames, {len(jumps)} jump frames skipped")
    print(f"  Art frames: {N}  |  grid {ART_W}×{ART_H}  block {BLOCK}px  (no crop)")

    # ── Pass 1: segmentation on all source frames (for temporal context) ──────
    print("\nPass 1: segmenting all frames…")
    raw_masks_all = []   # indexed by source frame
    for i in tqdm(range(N_total), unit='frame'):
        if i in jumps:
            z = np.zeros((H, W), np.uint8)
            raw_masks_all.append((z, z))
            continue
        frame     = frames[i]
        _, enh    = P.preprocess(frame)
        fbgr      = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ss, se    = seg_of(i, jumps, N_total)
        hou       = P._find_hough(enh, 17.0)
        cb        = P.get_cell_bodies(enh)
        nm, _     = P.segment_neutrophil(fbgr, enh, corr, i, ss, se, density, H, W)
        rm        = P.segment_rbcs(enh, cb, corr, i, ss, se, density, hou, 17.0, H, W)
        rm        = cv2.bitwise_and(rm, cv2.bitwise_not(nm))
        raw_masks_all.append((nm, rm))

    # ── Pass 2: temporal smoothing (on full source sequence) ──────────────────
    print("\nPass 2: temporal smoothing…")
    smooth_all = P.smooth_masks_temporally(raw_masks_all, jumps, N_total)

    # ── Select only non-jump frames ───────────────────────────────────────────
    smooth_masks = [smooth_all[i] for i in good_frames]
    src_indices  = good_frames   # source frame number for each art frame

    # ── Downsample to art space ───────────────────────────────────────────────
    print("\nConverting to art space…")
    art_neu = [to_art(m[0]) for m in tqdm(smooth_masks, unit='frame')]
    art_rbc = [to_art(m[1]) for m in smooth_masks]

    # ── RBC centroid extraction & stability filter ────────────────────────────
    print("\nExtracting and filtering RBC centroids…")
    raw_cents = [get_centroids(m) for m in art_rbc]
    MATCH_R = RBC_R + 2

    # Stability window operates in art-frame space (jumps already excluded)
    stable_cents = []
    for ai in tqdm(range(N), unit='frame'):
        window = list(range(max(0, ai - RBC_STAB_WIN), ai)) + \
                 list(range(ai + 1, min(N, ai + RBC_STAB_WIN + 1)))
        kept = []
        for cx, cy in raw_cents[ai]:
            hits = sum(1 for j in window
                       if any(near(cx, cy, nx, ny, MATCH_R)
                              for nx, ny in raw_cents[j]))
            if hits >= RBC_STAB_MIN:
                kept.append((cx, cy))
        stable_cents.append(kept)

    # ── Build stylised RBC art masks ──────────────────────────────────────────
    print("\nPainting stylised RBC circles…")
    sty_rbc = []
    for ai in tqdm(range(N), unit='frame'):
        canvas = np.zeros((ART_H, ART_W), bool)
        for cx, cy in stable_cents[ai]:
            paint_circle(canvas, cy, cx)
        sty_rbc.append(canvas)

    # ── Neutrophil temporal smoothing in art space ────────────────────────────
    print("\nSmoothing neutrophil…")
    neu_areas = [int(m.sum()) for m in art_neu]
    sty_neu   = list(art_neu)

    for ai in range(N):
        nbrs = list(range(max(0, ai - NEU_W), ai)) + \
               list(range(ai + 1, min(N, ai + NEU_W + 1)))
        if len(nbrs) < 2:
            continue
        med = float(np.median([neu_areas[j] for j in nbrs]))
        if med < 1:
            continue
        ratio = neu_areas[ai] / med if med > 0 else 0
        if ratio > NEU_RATIO or ratio < 1.0 / NEU_RATIO or neu_areas[ai] == 0:
            stack   = np.stack([art_neu[j].astype(float) for j in nbrs])
            sty_neu[ai] = stack.mean(axis=0) > 0.40

    # ── Compose final art frames ──────────────────────────────────────────────
    print("\nComposing…")
    k3 = np.ones((3, 3), np.uint8)
    out_frames = []
    for ai in range(N):
        art = np.full((ART_H, ART_W), BG, np.uint8)
        rbc = sty_rbc[ai]
        art[rbc] = RBC_INT
        eroded = cv2.erode(rbc.astype(np.uint8) * 255, k3)
        art[rbc & (eroded == 0)] = RBC_EDG
        art[sty_neu[ai].astype(bool)] = NEU
        out_frames.append(art.flatten().tolist())

    avg_rbc = np.mean([len(c) for c in stable_cents])
    print(f"\n  Art frames: {N}  |  avg stable RBCs/frame: {avg_rbc:.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        'width':         ART_W,
        'height':        ART_H,
        'n_frames':      N,
        'block_size':    BLOCK,
        'rbc_radius':    RBC_R,
        'fps':           15,
        'src_indices':   src_indices,     # source frame number per art frame
        'labels':        {'bg': BG, 'rbc_int': RBC_INT, 'rbc_edg': RBC_EDG, 'neu': NEU},
        'frames':        out_frames,
        'centroids':     stable_cents,    # [[cx,cy],...] per art frame
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(',', ':')))
    size_kb = out.stat().st_size // 1024
    print(f"Done → {out}  ({N} art frames × {ART_W}×{ART_H}, {size_kb} KB)")


if __name__ == '__main__':
    main()
