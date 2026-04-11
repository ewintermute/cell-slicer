"""
rbc_dejitter.py

Remove back-and-forth jitter from RBC sprites.

Definition of jitter: an RBC that moves to a new position for a short run
(≤ MAX_EXCURSION_LEN consecutive frames) then returns within RETURN_RADIUS
of its pre-excursion position. The excursion is snapped back to the
pre-excursion centroid.

The key safeguard: we only snap if the RBC stays at the new position for
≤ MAX_EXCURSION_LEN frames before returning. Genuine drift is longer.

Algorithm:
  1. Build per-track centroid trajectories (same greedy nearest-neighbour
     tracker used in analysis).
  2. Detect excursions: a run of frames where centroid is displaced
     ≥ MIN_DISP from the "home" position, followed by a return.
  3. For each excursion frame, re-stamp the sprite at the home centroid
     instead.

Parameters (conservative):
  MAX_EXCURSION_LEN = 4   only snap runs of ≤4 frames
  MIN_DISP          = 0.8  trigger if centroid moves ≥0.8 art-px from home
  RETURN_RADIUS     = 0.6  "returned" if new pos within 0.6 art-px of home
  MAX_SNAP_DIST     = 3.0  never snap more than 3 art-px (safety cap)
"""

import json, copy, shutil, numpy as np, cv2
from pathlib import Path

# ── Params ────────────────────────────────────────────────────────────────────
MAX_EXCURSION_LEN = 4
MIN_DISP          = 0.8
RETURN_RADIUS     = 0.6
MAX_SNAP_DIST     = 3.0
MATCH_R           = 3.5
MIN_AREA          = 10

R       = 3
ART_H, ART_W = 48, 64
DISC = [(dy,dx) for dy in range(-R,R+1) for dx in range(-R,R+1) if dy*dy+dx*dx<=R*R]
k3   = np.ones((3,3), np.uint8)

# ── Canonical sprite ──────────────────────────────────────────────────────────
def stamp_sprite(canvas, cy, cx):
    """Paint canonical RBC sprite (fill+edge) at integer (cy, cx). Returns list of (y,x,label)."""
    tmp = np.zeros((ART_H, ART_W), dtype=bool)
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            tmp[ny, nx] = True
    eroded = cv2.erode(tmp.astype(np.uint8)*255, k3)
    pixels = []
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            lbl = 2 if eroded[ny,nx] == 0 else 1
            pixels.append((ny, nx, lbl))
    return pixels

# ── Load ──────────────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)
N = data['n_frames']

frames_np = [np.array(data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)
             for fi in range(N)]

# ── Build tracks ──────────────────────────────────────────────────────────────
def get_centroids(grid):
    rbc_bin = np.isin(grid, [1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)
    out = []
    for c in range(1, nl):
        if stats[c, cv2.CC_STAT_AREA] >= MIN_AREA:
            out.append((c, float(cents[c][0]), float(cents[c][1])))  # (comp_id, cx, cy)
    return out, lmap

all_data = [get_centroids(g) for g in frames_np]

tracks   = {}   # tid -> [(fi, cx, cy)]
next_id  = 0
active   = {}   # tid -> (cx, cy)

for fi in range(N):
    cents, _ = all_data[fi]
    used_c   = set()
    used_t   = set()
    for tid, (lx, ly) in list(active.items()):
        best, best_d = None, MATCH_R
        for i, (cid, cx, cy) in enumerate(cents):
            if i in used_c: continue
            d = np.hypot(cx-lx, cy-ly)
            if d < best_d:
                best_d, best = d, i
        if best is not None:
            _, cx, cy = cents[best]
            tracks[tid].append((fi, cx, cy))
            active[tid] = (cx, cy)
            used_c.add(best)
            used_t.add(tid)
    for tid in list(active.keys()):
        if tid not in used_t: del active[tid]
    for i, (cid, cx, cy) in enumerate(cents):
        if i not in used_c:
            tracks[next_id] = [(fi, cx, cy)]
            active[next_id] = (cx, cy)
            next_id += 1

print(f"Tracks: {next_id}  |  long (≥20fr): {sum(1 for t in tracks.values() if len(t)>=20)}")

# ── Detect excursions in each track ──────────────────────────────────────────
# Returns list of (fi, home_cx, home_cy) — frames to snap + where to snap them
snap_ops = []   # (fi, old_cx, old_cy, new_cx, new_cy)

for tid, pts in tracks.items():
    if len(pts) < 6:
        continue
    arr = pts  # list of (fi, cx, cy)
    i = 0
    while i < len(arr):
        fi0, x0, y0 = arr[i]
        # Look for start of excursion: next point is displaced ≥ MIN_DISP
        if i + 1 >= len(arr):
            break
        fi1, x1, y1 = arr[i+1]
        disp = np.hypot(x1-x0, y1-y0)
        if disp < MIN_DISP:
            i += 1
            continue
        # We're at the start of a potential excursion.
        # Find how long it lasts before returning to within RETURN_RADIUS of (x0,y0)
        excursion_end = None
        for j in range(i+1, min(i+1+MAX_EXCURSION_LEN, len(arr))):
            fj, xj, yj = arr[j]
            ret = np.hypot(xj-x0, yj-y0)
            if ret <= RETURN_RADIUS:
                excursion_end = j
                break
        if excursion_end is None:
            # Didn't return — genuine movement, skip
            i += 1
            continue
        # Snap the excursion frames back to home (x0, y0)
        snap_dist = np.hypot(x1-x0, y1-y0)
        if snap_dist > MAX_SNAP_DIST:
            i += 1
            continue
        for j in range(i+1, excursion_end):
            fj, xj, yj = arr[j]
            snap_ops.append((fj, xj, yj, x0, y0))
        i = excursion_end  # resume after the return

print(f"Snap operations: {len(snap_ops)} (frames to re-stamp)")

# ── Apply snaps ───────────────────────────────────────────────────────────────
fixed = [g.copy() for g in frames_np]
frames_changed = set()

for (fi, old_cx, old_cy, new_cx, new_cy) in snap_ops:
    grid = fixed[fi]
    old_icy, old_icx = int(round(old_cy)), int(round(old_cx))
    new_icy, new_icx = int(round(new_cy)), int(round(new_cx))

    # Erase old sprite pixels (only RBC labels, not NEU/microbe)
    old_pixels = stamp_sprite(grid, old_icy, old_icx)
    for py, px, _ in old_pixels:
        if grid[py, px] in (1, 2):
            grid[py, px] = 0

    # Stamp new sprite at home position (don't overwrite NEU/microbe)
    new_pixels = stamp_sprite(grid, new_icy, new_icx)
    for py, px, lbl in new_pixels:
        if grid[py, px] == 0:
            grid[py, px] = lbl

    frames_changed.add(fi)

print(f"Frames modified: {len(frames_changed)}")

# ── Verify no NEU/microbe pixels touched ─────────────────────────────────────
bad = 0
for fi in frames_changed:
    for idx in range(ART_H * ART_W):
        b = frames_np[fi].flat[idx]
        a = fixed[fi].flat[idx]
        if b != a and b in (3, 4):
            bad += 1
            print(f"  BAD fi={fi} idx={idx} {b}->{a}")
print(f"Protected pixels touched: {bad}  {'✓' if bad==0 else '← PROBLEM'}")

# ── Save ──────────────────────────────────────────────────────────────────────
backup = art_path.with_suffix('.json.prejitter')
shutil.copy(art_path, backup)
print(f"Backup → {backup}")

fixed_data = copy.deepcopy(data)
for fi in range(N):
    fixed_data['frames'][fi] = fixed[fi].flatten().tolist()

with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',', ':'))
print(f"Done → {art_path}")
