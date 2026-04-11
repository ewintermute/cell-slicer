"""
rbc_repair.py

Find all RBC components that are smaller than a full 29-pixel canonical sprite
(indicating damage from nearby erase operations) and restore the missing pixels.

For each under-sized component, we try all integer centres within ±2 of the
centroid, pick the one that best explains the existing pixels (fewest missing,
most present), then fill in only the gaps — never overwriting existing labels.
"""

import json, copy, shutil, numpy as np, cv2
from pathlib import Path

R      = 3
ART_H, ART_W = 48, 64
MAX_MISSING   = 6   # accept sprites missing up to 6 pixels (1px shift overlap damage)
MIN_PRESENT_FRAC = 0.70  # at least 70% of canon must already be there

DISC = [(dy,dx) for dy in range(-R,R+1) for dx in range(-R,R+1) if dy*dy+dx*dx<=R*R]
k3   = np.ones((3,3), np.uint8)

def canonical_pixels(cy, cx):
    canvas = np.zeros((ART_H, ART_W), dtype=bool)
    in_bounds = []
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            canvas[ny, nx] = True
            in_bounds.append((ny, nx))
    eroded = cv2.erode(canvas.astype(np.uint8)*255, k3)
    return {(ny,nx): (2 if eroded[ny,nx]==0 else 1) for ny,nx in in_bounds}

# ── Load ──────────────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)

N = data['n_frames']
fixed_data = copy.deepcopy(data)

total_sprites = 0
total_pixels  = 0
frames_fixed  = set()

for fi in range(N):
    flat = list(fixed_data['frames'][fi])
    grid = np.array(flat, dtype=np.uint8).reshape(ART_H, ART_W)

    rbc_bin = np.isin(grid, [1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)

    frame_changed = False
    for c in range(1, nl):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < 10 or area >= 29:
            continue  # full sprite or too small fragment

        cx_f, cy_f = cents[c]

        # Try all candidate centres in ±2 window
        best = None
        best_score = (-1, 999)  # (present_count, missing_count) — maximise present, minimise missing

        for try_cy in range(int(cy_f)-2, int(cy_f)+3):
            for try_cx in range(int(cx_f)-2, int(cx_f)+3):
                canon = canonical_pixels(try_cy, try_cx)
                canon_size = len(canon)
                present = sum(1 for pyx in canon if grid[pyx[0],pyx[1]] in (1,2))
                missing = canon_size - present

                if present < canon_size * MIN_PRESENT_FRAC:
                    continue
                if missing > MAX_MISSING:
                    continue

                score = (present, -missing)
                if score > best_score:
                    best_score = score
                    best = (try_cy, try_cx, canon, missing)

        if best is None:
            continue

        try_cy, try_cx, canon, missing = best
        if missing == 0:
            continue

        # Fill missing pixels only — never overwrite NEU(3) or microbe(4)
        px_added = 0
        for (py, px), lbl in canon.items():
            idx = py * ART_W + px
            if flat[idx] == 0:
                flat[idx] = lbl
                px_added += 1

        if px_added > 0:
            total_sprites += 1
            total_pixels  += px_added
            frame_changed  = True

    if frame_changed:
        fixed_data['frames'][fi] = flat
        frames_fixed.add(fi)

print(f"Repaired {total_sprites} damaged sprites across {len(frames_fixed)} frames")
print(f"Total pixels restored: {total_pixels}")

# ── Verify ────────────────────────────────────────────────────────────────────
bad = 0
for fi in frames_fixed:
    orig = data['frames'][fi]
    fixed = fixed_data['frames'][fi]
    for idx, (b, a) in enumerate(zip(orig, fixed)):
        if b != a:
            if b != 0 or a not in (1, 2):
                bad += 1
                print(f"  BAD fi={fi} idx={idx} {b}->{a}")
print(f"Bad changes: {bad}  {'✓' if bad==0 else '← PROBLEM'}")

# ── Save ──────────────────────────────────────────────────────────────────────
backup = art_path.with_suffix('.json.prerepair')
shutil.copy(art_path, backup)
print(f"Backup → {backup}")

with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',', ':'))
print(f"Done → {art_path}")
