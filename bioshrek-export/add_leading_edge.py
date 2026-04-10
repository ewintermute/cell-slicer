#!/usr/bin/env python3
"""
add_leading_edge.py — Add leading-edge label to neutrophil pixels in pixel_art.json.

Marks the forward-facing boundary of the moving neutrophil as label 5 (Azurite).
The motion direction is computed from the smoothed centroid trajectory.

Label 5 is added to the labels dict in the JSON and to each frame.
The base pixel_art.json is NOT modified — this writes a new file:
  ../docs/pixel_art_le.json

Usage:
    python3 add_leading_edge.py [--smooth <frames>] [--depth <px>] [--percentile <pct>]

Defaults:
    --smooth     10    (frames each side for centroid velocity)
    --depth       1    (boundary pixels thick)
    --percentile 60    (keep top 40% of boundary by dot product)
"""

import json
import argparse
import numpy as np
from scipy import ndimage

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_LEADING = 5   # new label for leading edge
STRUCT4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)

def compute_centroids(frames, W, H):
    centroids = []
    for fi in range(len(frames)):
        frame = np.array(frames[fi]).reshape(H, W)
        ys, xs = np.where(frame == 3)
        centroids.append((float(xs.mean()), float(ys.mean())) if len(ys) > 0 else None)
    return centroids

def motion_dir(centroids, fi, smooth):
    lo = max(0, fi - smooth)
    hi = min(len(centroids) - 1, fi + smooth)
    c0, c1 = centroids[lo], centroids[hi]
    if c0 is None or c1 is None:
        return None
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    mag = (dx**2 + dy**2) ** 0.5
    return (dx / mag, dy / mag) if mag >= 0.5 else None

def leading_edge_mask(frame, motion_dx, motion_dy, depth, percentile):
    neut = (frame == 3)
    if not neut.any():
        return np.zeros_like(frame, dtype=bool)
    ys, xs = np.where(neut)
    cx, cy = xs.mean(), ys.mean()
    # Single-pixel boundary (neutrophil pixels with at least one non-neutrophil 4-neighbor)
    eroded = ndimage.binary_erosion(neut, structure=STRUCT4)
    boundary = neut & ~eroded
    bys, bxs = np.where(boundary)
    if len(bys) == 0:
        return np.zeros_like(frame, dtype=bool)
    dots = (bxs - cx) * motion_dx + (bys - cy) * motion_dy
    threshold = np.percentile(dots, percentile)
    leading = np.zeros_like(frame, dtype=bool)
    for y, x, d in zip(bys, bxs, dots):
        if d >= threshold:
            leading[y, x] = True
    # Optionally grow inward
    for _ in range(depth - 1):
        leading = ndimage.binary_dilation(leading, structure=STRUCT4) & neut
    return leading

def run(smooth, depth, percentile, source, output):
    print(f"Loading {source}...")
    with open(source) as f:
        data = json.load(f)

    W, H = data['width'], data['height']
    frames_in = data['frames']
    N = data['n_frames']

    print(f"  {N} frames, {W}×{H}. Computing centroids...")
    centroids = compute_centroids(frames_in, W, H)

    frames_out = []
    leading_count = 0
    still_count = 0

    for fi in range(N):
        frame = np.array(frames_in[fi]).reshape(H, W).copy()
        md = motion_dir(centroids, fi, smooth)
        if md is not None:
            dx, dy = md
            le = leading_edge_mask(frame, dx, dy, depth, percentile)
            frame[le] = LABEL_LEADING
            leading_count += le.sum()
        else:
            still_count += 1
        frames_out.append(frame.flatten().tolist())

    print(f"  Leading edge pixels added: {leading_count} across {N - still_count} frames")
    print(f"  Frames with no detectable motion: {still_count}")

    # Update labels dict
    out_data = dict(data)
    out_data['frames'] = frames_out
    labels = dict(data.get('labels', {}))
    labels['leading'] = LABEL_LEADING
    out_data['labels'] = labels

    print(f"Writing {output}...")
    with open(output, 'w') as f:
        json.dump(out_data, f, separators=(',', ':'))
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth',      type=int,   default=10)
    parser.add_argument('--depth',       type=int,   default=1)
    parser.add_argument('--percentile',  type=float, default=60)
    parser.add_argument('--source',      default='../docs/pixel_art.json')
    parser.add_argument('--output',      default='../docs/pixel_art_le.json')
    args = parser.parse_args()
    run(args.smooth, args.depth, args.percentile, args.source, args.output)
