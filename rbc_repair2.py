"""
rbc_repair2.py  —  Repair-only pass.

Find every RBC component that is missing pixels vs its best canonical centre.
ONLY add pixels (0→1 or 0→2). Never erase. Never touch NEU(3) or microbe(4).

Criteria:
  - Component area 10–28 (not full 29px)
  - Best canonical centre has ≥80% pixels already present
  - Missing ≤ 7 pixels
  - Only fill pixels that are currently 0
"""

import json, copy, shutil, numpy as np, cv2
from pathlib import Path

R      = 3
ART_H, ART_W = 48, 64
MIN_PRESENT  = 0.75
MAX_MISSING  = 7

DISC = [(dy,dx) for dy in range(-R,R+1) for dx in range(-R,R+1) if dy*dy+dx*dx<=R*R]
k3   = np.ones((3,3), np.uint8)

def canonical_pixels(cy, cx):
    canvas = np.zeros((ART_H, ART_W), dtype=bool)
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W: canvas[ny, nx] = True
    eroded = cv2.erode(canvas.astype(np.uint8)*255, k3)
    result = {}
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            result[(ny,nx)] = 2 if eroded[ny,nx]==0 else 1
    return result

art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)

N = data['n_frames']
fixed_data = copy.deepcopy(data)

repair_sprites = 0
repair_pixels  = 0
frames_touched = set()

for fi in range(N):
    flat = list(fixed_data['frames'][fi])
    grid = np.array(flat, dtype=np.uint8).reshape(ART_H, ART_W)
    rbc_bin = np.isin(grid,[1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)

    changed = False
    for c in range(1, nl):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < 10 or area >= 29: continue

        cx_f, cy_f = cents[c]
        best_miss, best_canon = 999, None

        for try_cy in range(int(cy_f)-2, int(cy_f)+3):
            for try_cx in range(int(cx_f)-2, int(cx_f)+3):
                canon = canonical_pixels(try_cy, try_cx)
                present = sum(1 for p in canon if grid[p[0],p[1]] in (1,2))
                missing = len(canon) - present
                if present >= len(canon) * MIN_PRESENT and missing < best_miss:
                    best_miss  = missing
                    best_canon = canon

        if best_canon is None or best_miss == 0 or best_miss > MAX_MISSING:
            continue

        px_added = 0
        for (py, px), lbl in best_canon.items():
            idx = py * ART_W + px
            if flat[idx] == 0:
                flat[idx] = lbl
                px_added += 1

        if px_added:
            repair_sprites += 1
            repair_pixels  += px_added
            changed = True

    if changed:
        fixed_data['frames'][fi] = flat
        frames_touched.add(fi)

print(f"Repaired {repair_sprites} sprites across {len(frames_touched)} frames")
print(f"Total pixels added: {repair_pixels}")

# Verify
bad = 0
for fi in frames_touched:
    orig = data['frames'][fi]
    fix  = fixed_data['frames'][fi]
    for idx, (b, a) in enumerate(zip(orig, fix)):
        if b != a:
            if b != 0:          bad += 1; print(f"  OVERWRITE fi={fi} idx={idx} {b}→{a}")
            if a in (3, 4):     bad += 1; print(f"  PROTECTED fi={fi} idx={idx} {b}→{a}")
print(f"Bad changes: {bad}  {'✓' if bad==0 else '← PROBLEM'}")

backup = art_path.with_suffix('.json.prerepair2')
shutil.copy(art_path, backup)
with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',',':'))
print(f"Done → {art_path}")
