#!/usr/bin/env python3
"""
add_organelles.py — Add simulated cytoplasmic organelles to neutrophil interior.

Uses Electra2 color (label 6). Particles undergo:
  - Translation with the neutrophil centroid (they move with the cell)
  - Brownian diffusion (random walk, sigma per frame)
  - Slow rotational streaming (cytoplasmic flow, ~1 revolution per 200 frames)
  - Boundary enforcement: particles that exit the neutrophil are reflected
    back to the nearest interior pixel.

On stage-jump frames (large centroid displacement) particles are re-seeded.

Source:  ../docs/pixel_art_le.json   (must already have leading edge, label 5)
Output:  ../docs/pixel_art_org.json  (full 64×48 with organelles)
         ../docs/pixel_art_org_48x32.json (48×32 crop, regenerated)

Usage:
    python3 add_organelles.py [--n-particles N] [--diffusion D] [--rotation R]
                               [--seed S]

Defaults:
    --n-particles  20
    --diffusion    0.45   (px per frame, 1-sigma Gaussian random walk)
    --rotation     0.030  (radians per frame ~1 rev per 210 frames)
    --seed         42
"""

import json
import argparse
import numpy as np
from scipy import ndimage

LABEL_ORGANELLE = 6
STRUCT4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
JUMP_THRESHOLD = 4.0   # px — centroid delta above this = stage jump, re-seed

def interior_pixels(frame):
    """Return (ys, xs) arrays of neutrophil interior pixels (labels 3 and 5)."""
    neut = (frame == 3) | (frame == 5)
    ys, xs = np.where(neut)
    return ys, xs

def centroid_of(frame):
    ys, xs = np.where((frame == 3) | (frame == 5))
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())

def nearest_interior(px, py, int_ys, int_xs):
    """Snap (px, py) to nearest interior pixel."""
    dists = (int_xs - px)**2 + (int_ys - py)**2
    idx = np.argmin(dists)
    return float(int_xs[idx]), float(int_ys[idx])

def seed_particles(n, int_ys, int_xs, rng):
    """Uniformly seed n particles among interior pixels."""
    idx = rng.choice(len(int_ys), size=n, replace=(len(int_ys) < n))
    return int_xs[idx].astype(float), int_ys[idx].astype(float)

def run(n_particles, diffusion, rotation, source, microbe_path, output, crop_source, crop_output, seed):
    rng = np.random.default_rng(seed)

    print(f"Loading {source}...")
    with open(source) as f:
        data = json.load(f)

    W, H = data['width'], data['height']
    frames_in = data['frames']
    N = data['n_frames']

    print(f"  {N} frames, {W}×{H}, {n_particles} particles")
    print(f"  diffusion={diffusion}px/frame  rotation={rotation:.4f}rad/frame")

    frames_out = []

    # Initialise particles in frame 0
    f0 = np.array(frames_in[0]).reshape(H, W)
    int_ys, int_xs = interior_pixels(f0)
    px, py = seed_particles(n_particles, int_ys, int_xs, rng)   # float positions
    prev_cx, prev_cy = centroid_of(f0)

    for fi in range(N):
        frame = np.array(frames_in[fi]).reshape(H, W).copy()
        int_ys, int_xs = interior_pixels(frame)

        if len(int_ys) == 0:
            frames_out.append(frame.flatten().tolist())
            continue

        cx, cy = centroid_of(frame)

        # ── Stage jump: re-seed particles ─────────────────────────────────────
        delta = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
        if delta > JUMP_THRESHOLD:
            px, py = seed_particles(n_particles, int_ys, int_xs, rng)
        else:
            # ── Translate with centroid ────────────────────────────────────────
            dx_cell = cx - prev_cx
            dy_cell = cy - prev_cy
            px += dx_cell
            py += dy_cell

            # ── Rotational streaming (rotate around centroid) ──────────────────
            rel_x = px - cx
            rel_y = py - cy
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            px = cx + rel_x * cos_r - rel_y * sin_r
            py = cy + rel_x * sin_r + rel_y * cos_r

            # ── Brownian diffusion ────────────────────────────────────────────
            px += rng.normal(0, diffusion, n_particles)
            py += rng.normal(0, diffusion, n_particles)

            # ── Boundary enforcement ──────────────────────────────────────────
            # Build interior mask for fast lookup
            interior_mask = np.zeros((H, W), dtype=bool)
            interior_mask[int_ys, int_xs] = True

            for i in range(n_particles):
                xi, yi = int(round(px[i])), int(round(py[i]))
                if (0 <= yi < H and 0 <= xi < W and interior_mask[yi, xi]):
                    pass  # still inside
                else:
                    # Snap to nearest interior pixel
                    nx, ny = nearest_interior(px[i], py[i], int_ys, int_xs)
                    # Small jitter so they don't pile up on the boundary
                    px[i] = nx + rng.normal(0, 0.3)
                    py[i] = ny + rng.normal(0, 0.3)

        prev_cx, prev_cy = cx, cy

        # ── Paint organelles ──────────────────────────────────────────────────
        # Build interior mask
        interior_mask = np.zeros((H, W), dtype=bool)
        interior_mask[int_ys, int_xs] = True

        for i in range(n_particles):
            xi, yi = int(round(px[i])), int(round(py[i]))
            if 0 <= yi < H and 0 <= xi < W and interior_mask[yi, xi]:
                frame[yi, xi] = LABEL_ORGANELLE

        frames_out.append(frame.flatten().tolist())

    # ── Save full-frame output ─────────────────────────────────────────────────
    out_data = dict(data)
    out_data['frames'] = frames_out
    labels = dict(data.get('labels', {}))
    labels['organelle'] = LABEL_ORGANELLE
    out_data['labels'] = labels

    print(f"Writing {output}...")
    with open(output, 'w') as f:
        json.dump(out_data, f, separators=(',', ':'))

    # ── Regenerate 48×32 crop ─────────────────────────────────────────────────
    print(f"Regenerating crop {crop_output}...")
    with open(crop_source) as f:
        crop = json.load(f)
    src_frames = [np.array(fr).reshape(H, W) for fr in frames_out]
    n_cols, n_rows = crop['n_cols'], crop['n_rows']
    cx0_arr, cy0_arr = crop['crop_x0'], crop['crop_y0']
    crop_frames = []
    for fi in range(crop['n_frames']):
        x0, y0 = cx0_arr[fi], cy0_arr[fi]
        crop_frames.append(src_frames[fi][y0:y0+n_rows, x0:x0+n_cols].flatten().tolist())
    crop_out = dict(crop)
    crop_out['frames'] = crop_frames
    crop_labels = dict(crop.get('labels', {}))
    crop_labels['organelle'] = LABEL_ORGANELLE
    crop_out['labels'] = crop_labels
    with open(crop_output, 'w') as f:
        json.dump(crop_out, f, separators=(',', ':'))

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-particles', type=int,   default=20)
    parser.add_argument('--diffusion',   type=float, default=0.45)
    parser.add_argument('--rotation',    type=float, default=0.030)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--source',      default='../docs/pixel_art_le.json')
    parser.add_argument('--microbe',     default='../docs/microbe_positions.json')
    parser.add_argument('--output',      default='../docs/pixel_art_org.json')
    parser.add_argument('--crop-source', default='../docs/pixel_art_48x32.json')
    parser.add_argument('--crop-output', default='../docs/pixel_art_org_48x32.json')
    args = parser.parse_args()
    run(args.n_particles, args.diffusion, args.rotation,
        args.source, args.microbe, args.output,
        args.crop_source, args.crop_output, args.seed)
