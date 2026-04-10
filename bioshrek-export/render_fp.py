#!/usr/bin/env python3
"""
render_fp.py — Render Cell Slicer pixel art using fluorescent protein colors.

Usage:
    python3 render_fp.py [--source <json>] [--output <mp4>] [--scale <int>]

Defaults:
    --source  ../docs/pixel_art.json         (64×48 full frame)
    --output  output/fp_fullframe.mp4
    --scale   10                              (640×480 output)

Other useful combos:
    python3 render_fp.py --source ../docs/pixel_art_48x32.json --output output/fp_48x32.mp4
"""

import json
import argparse
import os
import numpy as np
import imageio.v3 as iio

# ── Fluorescent protein color palette ─────────────────────────────────────────
# Label → (R, G, B)  [uint8]
FP_COLORS = {
    0: (17,   17,  17),   # Background       — no FP, near-black
    1: (255, 128,   0),   # RBC interior     — mKO2    #FF8000
    2: (255,   0,   0),   # RBC edge         — mRFP    #FF0000
    3: (  0, 200, 255),   # Neutrophil       — mTurquoise2  #00C8FF
    4: (170, 255,   0),   # Bacterium        — Venus   #AAFF00
}

def render(source_path, output_path, scale):
    print(f"Loading {source_path}...")
    with open(source_path) as f:
        data = json.load(f)

    W = data['n_cols'] if 'n_cols' in data else data['width']
    H = data['n_rows'] if 'n_rows' in data else data['height']
    fps = data.get('fps', 15)
    n_frames = data['n_frames']

    out_w = W * scale
    out_h = H * scale

    print(f"  {n_frames} frames, {W}×{H} art pixels → {out_w}×{out_h} output @ {fps} fps")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    frames_out = []
    for fi in range(n_frames):
        labels = np.array(data['frames'][fi], dtype=np.uint8).reshape(H, W)
        # Map labels to RGB
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for lbl, color in FP_COLORS.items():
            mask = labels == lbl
            rgb[mask] = color
        # Unknown labels → magenta (flag clearly)
        unknown = ~np.isin(labels, list(FP_COLORS.keys()))
        if unknown.any():
            rgb[unknown] = (255, 0, 255)

        # Nearest-neighbour upscale (no blur — crisp pixels)
        big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        frames_out.append(big)

    print(f"  Writing {output_path}...")
    iio.imwrite(
        output_path,
        frames_out,
        plugin='pyav',
        fps=fps,
        codec='libx264',
        out_pixel_format='yuv420p',
    )
    print(f"  Done → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='../docs/pixel_art.json')
    parser.add_argument('--output', default='output/fp_fullframe.mp4')
    parser.add_argument('--scale',  type=int, default=10)
    args = parser.parse_args()
    render(args.source, args.output, args.scale)
