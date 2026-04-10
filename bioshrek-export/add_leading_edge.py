#!/usr/bin/env python3
"""
add_leading_edge.py — Add leading-edge label to neutrophil pixels in pixel_art.json.

Marks the forward-facing boundary of the neutrophil as label 5 (Azurite).

Direction is derived from the neutrophil→chased-bacterium vector, smoothed over a
wide window so it changes slowly and always points toward prey.

Outputs:
  ../docs/pixel_art_le.json        (full 64×48)
  ../docs/pixel_art_le_48x32.json  (48×32 jump-cut crop, regenerated from above)

Usage:
    python3 add_leading_edge.py [--smooth <frames>] [--depth <px>] [--percentile <pct>]

Defaults:
    --smooth     15    (uniform filter half-width for direction smoothing)
    --depth       1    (boundary pixels thick)
    --percentile 60    (keep top 40% of boundary by dot product — narrower stripe)
"""

import json
import argparse
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter1d

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_LEADING = 5
STRUCT4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)

def chased_microbe_centroids(mp_data, W, H):
    """Return per-frame (cx, cy) for the chased (leftmost) microbe component."""
    N = mp_data['n_frames']
    result = []
    for fi in range(N):
        entry = mp_data['frames'][fi]
        if not entry or not entry.get('pixels'):
            result.append(None); continue
        arr = np.zeros((H, W), dtype=bool)
        for px, py in entry['pixels']:
            if 0 <= py < H and 0 <= px < W: arr[py, px] = True
        labeled, n = ndimage.label(arr)
        if n == 0:
            result.append(None); continue
        best_cx, best_pos = 999, None
        for comp in range(1, n + 1):
            cys, cxs = np.where(labeled == comp)
            cx = cxs.mean()
            if cx < best_cx:
                best_cx = cx
                best_pos = (float(cxs.mean()), float(cys.mean()))
        result.append(best_pos)
    return result

def smooth_prey_directions(neu_centroids, microbe_centroids, smooth):
    """
    Compute per-frame direction unit vector: neutrophil → chased microbe,
    smoothed with a uniform filter of width 2*smooth+1.
    Returns arrays (dx, dy), each of length N.
    """
    N = len(neu_centroids)
    raw_dx = np.zeros(N)
    raw_dy = np.zeros(N)
    for fi in range(N):
        nc = neu_centroids[fi]
        mc = microbe_centroids[fi]
        if nc and mc:
            dx, dy = mc[0] - nc[0], mc[1] - nc[1]
            mag = (dx**2 + dy**2) ** 0.5
            if mag > 0:
                raw_dx[fi] = dx / mag
                raw_dy[fi] = dy / mag
    sm_dx = uniform_filter1d(raw_dx, size=smooth * 2 + 1, mode='nearest')
    sm_dy = uniform_filter1d(raw_dy, size=smooth * 2 + 1, mode='nearest')
    mags = np.sqrt(sm_dx**2 + sm_dy**2)
    mags = np.where(mags < 0.01, 1.0, mags)
    return sm_dx / mags, sm_dy / mags

def leading_edge_mask(frame, dir_dx, dir_dy, depth, percentile):
    neut = (frame == 3)
    if not neut.any():
        return np.zeros_like(frame, dtype=bool)
    ys, xs = np.where(neut)
    cx, cy = xs.mean(), ys.mean()
    eroded = ndimage.binary_erosion(neut, structure=STRUCT4)
    boundary = neut & ~eroded
    bys, bxs = np.where(boundary)
    if len(bys) == 0:
        return np.zeros_like(frame, dtype=bool)
    dots = (bxs - cx) * dir_dx + (bys - cy) * dir_dy
    threshold = np.percentile(dots, percentile)
    leading = np.zeros_like(frame, dtype=bool)
    for y, x, d in zip(bys, bxs, dots):
        if d >= threshold:
            leading[y, x] = True
    for _ in range(depth - 1):
        leading = ndimage.binary_dilation(leading, structure=STRUCT4) & neut
    return leading

def run(smooth, depth, percentile, source, microbe_path, output, crop_source, crop_output):
    print(f"Loading {source}...")
    with open(source) as f:
        data = json.load(f)
    with open(microbe_path) as f:
        mp = json.load(f)

    W, H = data['width'], data['height']
    frames_in = data['frames']
    N = data['n_frames']

    print(f"  {N} frames, {W}×{H}. Computing neutrophil & microbe centroids...")
    neu_c = []
    for fi in range(N):
        frame = np.array(frames_in[fi]).reshape(H, W)
        ys, xs = np.where(frame == 3)
        neu_c.append((float(xs.mean()), float(ys.mean())) if len(ys) > 0 else None)

    microbe_c = chased_microbe_centroids(mp, W, H)
    dir_x, dir_y = smooth_prey_directions(neu_c, microbe_c, smooth)

    angles = np.degrees(np.arctan2(dir_y, dir_x))
    diffs = np.abs(np.diff(angles))
    diffs = np.where(diffs > 180, 360 - diffs, diffs)
    print(f"  Direction smoothness — max Δangle/frame: {diffs.max():.1f}°, mean: {diffs.mean():.2f}°")

    frames_out = []
    leading_total = 0
    for fi in range(N):
        frame = np.array(frames_in[fi]).reshape(H, W).copy()
        le = leading_edge_mask(frame, dir_x[fi], dir_y[fi], depth, percentile)
        frame[le] = LABEL_LEADING
        leading_total += le.sum()
        frames_out.append(frame.flatten().tolist())

    print(f"  Leading edge pixels: {leading_total} total ({leading_total/N:.1f} avg/frame)")

    out_data = dict(data)
    out_data['frames'] = frames_out
    labels = dict(data.get('labels', {}))
    labels['leading'] = LABEL_LEADING
    out_data['labels'] = labels

    print(f"Writing {output}...")
    with open(output, 'w') as f:
        json.dump(out_data, f, separators=(',', ':'))

    # Regenerate cropped version
    print(f"Regenerating crop {crop_output}...")
    with open(crop_source) as f:
        crop = json.load(f)
    src_frames = [np.array(fr).reshape(H, W) for fr in frames_out]
    n_cols, n_rows = crop['n_cols'], crop['n_rows']
    cx0, cy0 = crop['crop_x0'], crop['crop_y0']
    crop_frames = []
    for fi in range(crop['n_frames']):
        x0, y0 = cx0[fi], cy0[fi]
        crop_frames.append(src_frames[fi][y0:y0+n_rows, x0:x0+n_cols].flatten().tolist())
    crop_out = dict(crop)
    crop_out['frames'] = crop_frames
    crop_labels = dict(crop.get('labels', {}))
    crop_labels['leading'] = LABEL_LEADING
    crop_out['labels'] = crop_labels
    with open(crop_output, 'w') as f:
        json.dump(crop_out, f, separators=(',', ':'))

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth',      type=int,   default=15)
    parser.add_argument('--depth',       type=int,   default=1)
    parser.add_argument('--percentile',  type=float, default=60)
    parser.add_argument('--source',         default='../docs/pixel_art.json')
    parser.add_argument('--microbe',        default='../docs/microbe_positions.json')
    parser.add_argument('--output',         default='../docs/pixel_art_le.json')
    parser.add_argument('--crop-source',    default='../docs/pixel_art_48x32.json')
    parser.add_argument('--crop-output',    default='../docs/pixel_art_le_48x32.json')
    args = parser.parse_args()
    run(args.smooth, args.depth, args.percentile,
        args.source, args.microbe, args.output,
        args.crop_source, args.crop_output)
