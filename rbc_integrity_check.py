"""
rbc_integrity_check.py

Scan every frame in pixel_art.json and fill in missing pixels for RBC sprites
that are ALMOST complete — defined as: the canonical 29-pixel sprite is present
except for at most MAX_MISSING pixels (default 3).

Frame-edge clipping is handled: if a sprite centre is near the edge, some pixels
are legitimately absent; those don't count toward the missing-pixel budget.

Sprites with more than MAX_MISSING unclipped pixels absent are left untouched.
"""

import json, copy, sys, shutil
from pathlib import Path
import numpy as np
import cv2

# ── Canonical sprite ──────────────────────────────────────────────────────────
R = 3
ART_H, ART_W = 48, 64
MAX_MISSING   = 3   # only fix sprites missing ≤ this many pixels

DISC = [
    (dy, dx)
    for dy in range(-R, R+1)
    for dx in range(-R, R+1)
    if dy*dy + dx*dx <= R*R
]  # 29 offsets

k3 = np.ones((3,3), np.uint8)

def canonical_pixels(cy, cx):
    """
    Return dict (y,x)->label for the canonical sprite at (cy,cx),
    restricted to pixels that are within frame bounds.
    label: 1=fill, 2=edge
    """
    canvas = np.zeros((ART_H, ART_W), dtype=bool)
    in_bounds = []
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            canvas[ny, nx] = True
            in_bounds.append((ny, nx))
    eroded = cv2.erode(canvas.astype(np.uint8)*255, k3)
    result = {}
    for ny, nx in in_bounds:
        result[(ny, nx)] = 2 if eroded[ny, nx] == 0 else 1
    return result

# ── Load pixel art ────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path}...")
with open(art_path) as f:
    data = json.load(f)

n_frames = data['n_frames']

# ── Per-frame analysis ────────────────────────────────────────────────────────
total_sprites_fixed = 0
total_px_fixed      = 0
fixes_by_frame      = {}  # frame_idx -> list of (y, x, label)

for fi in range(n_frames):
    flat = data['frames'][fi]
    grid = np.array(flat, dtype=np.uint8).reshape(ART_H, ART_W)

    rbc_mask = np.isin(grid, [1, 2])
    if not rbc_mask.any():
        continue

    # Find connected components of RBC pixels
    rbc_bin = rbc_mask.astype(np.uint8)
    n_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)

    frame_fixes = []

    for comp in range(1, n_labels):
        cy_f, cx_f = centroids[comp]
        cy, cx = int(round(cy_f)), int(round(cx_f))

        # Try centre rounding candidates (±1)
        best_centre  = None
        best_missing = 999
        best_canon   = None

        for try_cy in [cy, cy-1, cy+1]:
            for try_cx in [cx, cx-1, cx+1]:
                canon = canonical_pixels(try_cy, try_cx)
                # How many in-bounds canonical pixels are absent?
                missing_count = sum(1 for pyx in canon if grid[pyx[0], pyx[1]] == 0)
                # How many in-bounds canonical pixels are present?
                present_count = sum(1 for pyx in canon if grid[pyx[0], pyx[1]] in (1, 2))

                if missing_count < best_missing and present_count > 0:
                    best_missing = missing_count
                    best_centre  = (try_cy, try_cx)
                    best_canon   = canon

        if best_missing == 0 or best_missing > MAX_MISSING:
            # Perfect or too many missing — skip
            continue

        # Verify: the present pixels must nearly fill the canon
        # (guards against a tiny fragment accidentally matching a shifted centre)
        present_count = sum(1 for pyx in best_canon if grid[pyx[0], pyx[1]] in (1, 2))
        canon_size    = len(best_canon)  # may be < 29 if clipped by frame edge
        if present_count < canon_size - MAX_MISSING:
            continue  # shouldn't happen given above logic, but be safe

        bcy, bcx = best_centre
        for pyx, lbl in best_canon.items():
            if grid[pyx[0], pyx[1]] == 0:
                frame_fixes.append((pyx[0], pyx[1], lbl))

    if frame_fixes:
        fixes_by_frame[fi] = frame_fixes
        total_sprites_fixed += 1   # approximate (one component = one sprite)
        total_px_fixed      += len(frame_fixes)

print(f"\nFrames with fixable incomplete sprites: {len(fixes_by_frame)}")
print(f"Total pixels to fill: {total_px_fixed}")

if total_px_fixed == 0:
    print("Nothing to fix!")
    sys.exit(0)

# Show distribution of fix sizes
from collections import Counter
size_dist = Counter(len(v) for v in fixes_by_frame.values())
print("\nFix size distribution (pixels per frame):")
for k in sorted(size_dist):
    print(f"  {k} px: {size_dist[k]} frames")

sample = sorted(fixes_by_frame.keys())[:15]
print("\nSample frames:")
for fi in sample:
    fixes = fixes_by_frame[fi]
    fills = sum(1 for _,_,l in fixes if l==1)
    edges = sum(1 for _,_,l in fixes if l==2)
    print(f"  Frame {fi:3d}: +{len(fixes)} px  (fill={fills}, edge={edges})")

# ── Apply fixes ───────────────────────────────────────────────────────────────
print("\nApplying fixes...")
fixed_data = copy.deepcopy(data)

for fi, fixes in fixes_by_frame.items():
    flat = list(fixed_data['frames'][fi])
    for (py, px, lbl) in fixes:
        idx = py * ART_W + px
        if flat[idx] == 0:  # never overwrite existing labels
            flat[idx] = lbl
    fixed_data['frames'][fi] = flat

backup = art_path.with_suffix('.json.prebake')
shutil.copy(art_path, backup)
print(f"Backup saved to {backup}")

with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',',':'))

print(f"Done. Filled {total_px_fixed} pixels across {len(fixes_by_frame)} frames → {art_path}")
