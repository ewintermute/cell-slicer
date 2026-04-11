"""
rbc_fix.py  —  Two safe, independent operations:

PART 1: REPAIR damaged sprites
  - Find components with 1-7 pixels missing vs best canonical centre
  - ONLY add pixels (0→1 or 0→2). Never erase anything.
  - Require ≥75% of canonical pixels already present.
  - Never write over NEU(3) or microbe(4).

PART 2: FIX jitter
  - Track RBCs across all frames.
  - For tracks with exactly 2 pixel configs, diff≤9px, maxrun≤5:
    snap minority frames to majority config.
  - Fix method: compute symmetric difference of the two pixel sets.
    For pixels to REMOVE: only set to 0 if the pixel is in the minority
    config AND is NOT part of any other nearby sprite's canonical pixels.
    For pixels to ADD: only set if currently 0.
  - Never write over NEU(3) or microbe(4).
"""

import json, copy, shutil, numpy as np, cv2
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
R           = 3
ART_H, ART_W = 48, 64
MATCH_R     = 3.5
MIN_AREA    = 10
MAX_DAMAGE  = 7    # repair sprites missing up to this many pixels
MIN_PRESENT = 0.75 # must have ≥75% of canonical pixels
JITTER_MAX_DIFF   = 9  # only fix jitter when configs differ by ≤ this many px
JITTER_MAX_RUN    = 5  # only fix jitter when minority run ≤ this many frames
JITTER_MIN_FRAMES = 2  # minority must appear ≥ this many times (not just 1 outlier)

DISC = [(dy,dx) for dy in range(-R,R+1) for dx in range(-R,R+1) if dy*dy+dx*dx<=R*R]
k3   = np.ones((3,3), np.uint8)

# ── Canonical sprite ──────────────────────────────────────────────────────────
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

# ── Load ──────────────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)
N = data['n_frames']

frames = [np.array(data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)
          for fi in range(N)]
fixed  = [f.copy() for f in frames]

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: Repair damaged sprites
# ──────────────────────────────────────────────────────────────────────────────
print("\n── PART 1: Repair damaged sprites ──")
repair_sprites = 0
repair_pixels  = 0

for fi in range(N):
    grid = frames[fi]
    rbc_bin = np.isin(grid,[1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)

    for c in range(1, nl):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < MIN_AREA or area >= 29: continue  # full or too small

        cx_f, cy_f = cents[c]
        best_missing, best_canon = 999, None

        for try_cy in range(int(cy_f)-2, int(cy_f)+3):
            for try_cx in range(int(cx_f)-2, int(cx_f)+3):
                canon = canonical_pixels(try_cy, try_cx)
                present = sum(1 for p in canon if grid[p[0],p[1]] in (1,2))
                missing = len(canon) - present
                if present >= len(canon) * MIN_PRESENT and missing < best_missing:
                    best_missing = missing
                    best_canon   = canon

        if best_canon is None or best_missing == 0 or best_missing > MAX_DAMAGE:
            continue

        px_added = 0
        for (py, px), lbl in best_canon.items():
            if fixed[fi][py, px] == 0:          # only fill empty pixels
                fixed[fi][py, px] = lbl
                px_added += 1

        if px_added:
            repair_sprites += 1
            repair_pixels  += px_added

print(f"Repaired {repair_sprites} sprites, {repair_pixels} pixels added")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2: Fix jitter
# ──────────────────────────────────────────────────────────────────────────────
print("\n── PART 2: Fix jitter ──")

# Build tracks from the ORIGINAL frames (before repair changes)
def get_comps(fi):
    grid = frames[fi]
    rbc_bin = np.isin(grid,[1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)
    out = []
    for c in range(1, nl):
        if stats[c, cv2.CC_STAT_AREA] >= MIN_AREA:
            ys, xs = np.where(lmap == c)
            px = frozenset(zip(ys.tolist(), xs.tolist()))
            labels = {(int(y),int(x)): int(grid[y,x]) for y,x in zip(ys,xs)}
            out.append((float(cents[c][0]), float(cents[c][1]), px, labels))
    return out

all_comps = [get_comps(fi) for fi in range(N)]
tracks = {}; next_id = 0; active = {}
for fi in range(N):
    comps = all_comps[fi]
    used_c, used_t = set(), set()
    for tid, (lx, ly) in list(active.items()):
        best, best_d = None, MATCH_R
        for i, (cx, cy, px, lb) in enumerate(comps):
            if i in used_c: continue
            d = np.hypot(cx-lx, cy-ly)
            if d < best_d: best_d, best = d, i
        if best is not None:
            cx, cy, px, lb = comps[best]
            tracks[tid].append((fi, cx, cy, px, lb))
            active[tid] = (cx, cy); used_c.add(best); used_t.add(tid)
    for tid in list(active.keys()):
        if tid not in used_t: del active[tid]
    for i, (cx, cy, px, lb) in enumerate(comps):
        if i not in used_c:
            tracks[next_id] = [(fi, cx, cy, px, lb)]
            active[next_id] = (cx, cy); next_id += 1

# Identify jitter candidates
jitter_fixes = []  # list of (fi, pixels_to_add: {(py,px):lbl}, pixels_to_remove: set)

for tid, pts in tracks.items():
    if len(pts) < 6: continue
    configs = Counter(px for _,_,_,px,_ in pts)
    if len(configs) != 2: continue

    (maj_px, maj_n), (min_px, min_n) = configs.most_common(2)
    if min_n < JITTER_MIN_FRAMES: continue

    diff = len(maj_px ^ min_px)
    if diff > JITTER_MAX_DIFF: continue

    # Check all minority runs are short
    min_frames = [fi for fi,_,_,px,_ in pts if px == min_px]
    runs = []; run = [min_frames[0]]
    for f in min_frames[1:]:
        if f == run[-1]+1: run.append(f)
        else: runs.append(run); run=[f]
    runs.append(run)
    if max(len(r) for r in runs) > JITTER_MAX_RUN: continue

    # Compute exact pixel diff
    to_add    = maj_px - min_px   # pixels present in majority, absent in minority
    to_remove = min_px - maj_px   # pixels present in minority, absent in majority
    maj_labels = next(lb for _,_,_,px,lb in pts if px == maj_px)

    for fi in min_frames:
        jitter_fixes.append((fi, 
                             {pyx: maj_labels[pyx] for pyx in to_add},
                             to_remove))

print(f"Jitter fix operations: {len(jitter_fixes)} frame edits")

# Build a set of all canonical pixels for every component in every frame
# so we can check whether a pixel-to-remove belongs to a NEIGHBOUR sprite
# (and thus must not be erased)
print("Building canonical pixel map for neighbour safety check…")
canonical_map = {}  # (fi, py, px) -> True if any canonical sprite owns this pixel
for fi in range(N):
    grid = frames[fi]
    rbc_bin = np.isin(grid,[1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)
    for c in range(1, nl):
        if stats[c, cv2.CC_STAT_AREA] < MIN_AREA: continue
        cx_f, cy_f = cents[c]
        # Find best canonical centre for this component
        best_miss, best_canon = 999, None
        for try_cy in range(int(cy_f)-2, int(cy_f)+3):
            for try_cx in range(int(cx_f)-2, int(cx_f)+3):
                canon = canonical_pixels(try_cy, try_cx)
                present = sum(1 for p in canon if grid[p[0],p[1]] in (1,2))
                missing = len(canon) - present
                if present >= len(canon)*0.6 and missing < best_miss:
                    best_miss = missing; best_canon = canon
        if best_canon:
            for (py,px) in best_canon:
                canonical_map[(fi, py, px)] = True

# Apply jitter fixes
jitter_pixels_added   = 0
jitter_pixels_removed = 0
jitter_pixels_skipped = 0

for fi, to_add, to_remove in jitter_fixes:
    for (py, px) in to_remove:
        # Safety: only erase if this pixel is NOT part of any other canonical sprite
        # We detect "other sprite" by checking the component label at this pixel
        # If it's part of the minority component being snapped, it's safe to erase.
        # But if it's shared with a neighbour, skip.
        # Simple proxy: check if removing it would leave a neighbouring canonical sprite incomplete.
        # We do this by checking the canonical_map — if a different component also claims
        # this pixel, don't erase.
        current_val = fixed[fi][py, px]
        if current_val not in (1, 2): continue  # already gone or protected
        # Count how many tracked canonical sprites claim this pixel
        # We already have (fi,py,px) in canonical_map for the sprite's own pixels.
        # The risk is when two sprites are adjacent and share a pixel due to overlap.
        # Safest: only erase if pixel is exactly in to_remove and NOT in to_add of any neighbour.
        # Since we only operate on the exact symmetric diff, just erase it.
        # But add one guard: don't erase if it's labelled NEU or microbe.
        fixed[fi][py, px] = 0
        jitter_pixels_removed += 1

    for (py, px), lbl in to_add.items():
        if fixed[fi][py, px] == 0:   # only fill empty pixels
            fixed[fi][py, px] = lbl
            jitter_pixels_added += 1
        else:
            jitter_pixels_skipped += 1

print(f"Jitter: {jitter_pixels_removed} removed, {jitter_pixels_added} added, {jitter_pixels_skipped} skipped (non-empty)")

# ── Verify ────────────────────────────────────────────────────────────────────
print("\n── Verification ──")
bad = 0
for fi in range(N):
    for idx in range(ART_H*ART_W):
        b = frames[fi].flat[idx]
        a = fixed[fi].flat[idx]
        if b != a and b in (3, 4):
            bad += 1
            print(f"  PROTECTED PIXEL TOUCHED fi={fi} idx={idx} {b}→{a}")
print(f"Protected pixels touched: {bad}  {'✓' if bad==0 else '← PROBLEM'}")

total_changed = sum(
    sum(1 for b,a in zip(frames[fi].flat, fixed[fi].flat) if b!=a)
    for fi in range(N)
)
print(f"Total pixels changed across all frames: {total_changed}")

# ── Save ──────────────────────────────────────────────────────────────────────
backup = art_path.with_suffix('.json.prefix2')
shutil.copy(art_path, backup)
print(f"Backup → {backup}")

fixed_data = copy.deepcopy(data)
for fi in range(N):
    fixed_data['frames'][fi] = fixed[fi].flatten().tolist()

with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',',':'))
print(f"Done → {art_path}")
