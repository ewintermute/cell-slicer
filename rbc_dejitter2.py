"""
rbc_dejitter2.py — Pixel-exact jitter detection and correction.

No centroid tracking. No stamp_sprite. No bounding-box erases.

Algorithm:
  1. For every frame, extract every RBC connected component as a frozenset
     of (y,x) pixels. Store its label map too.
  2. Collect all unique pixel sets across all frames.
  3. For each pair of pixel sets A and B that:
       - Overlap spatially (intersection non-empty, or bounding boxes nearby)
       - Differ by ≤ MAX_DIFF pixels (symmetric difference)
       - Both appear in at least MIN_APPEARANCES frames
       - Are never present in the SAME frame simultaneously (they're alternatives)
     -> These are jitter twins.
  4. For each twin pair, compute frame lists. The minority (fewer frames)
     is the jitter. Snap minority frames to the majority pixel set by:
       a. Setting to_remove pixels (in minority but not majority) to 0
          — but ONLY if they are currently part of this specific component
            and not shared with any other component in the same frame.
       b. Setting to_add pixels (in majority but not minority) to their
          label — ONLY if currently 0.

This is safe because:
  - We only ever touch the exact pixels in the two alternative sets.
  - We never infer positions from centroids.
  - We never erase pixels owned by a different component.
"""

import json, copy, shutil, numpy as np, cv2
from collections import defaultdict
from pathlib import Path

MAX_DIFF          = 9   # max pixels different between twin pixel sets
MIN_APPEARANCES   = 2   # each twin must appear in at least this many frames
MAX_MINORITY_RUN  = 8   # minority config max consecutive-frame run before we consider it real movement
MIN_AREA          = 10  # ignore tiny fragments

ART_H, ART_W = 48, 64

# ── Load ─────────────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)
N = data['n_frames']

# ── Extract all components per frame ─────────────────────────────────────────
# comp_info[fi] = list of (px_frozenset, label_dict, component_label_map_for_this_comp)
print("Extracting components …")
frame_comps = []   # frame_comps[fi] = list of (px_set, label_dict)
for fi in range(N):
    grid = np.array(data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)
    rbc_bin = np.isin(grid, [1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)
    comps = []
    for c in range(1, nl):
        if stats[c, cv2.CC_STAT_AREA] < MIN_AREA: continue
        ys, xs = np.where(lmap == c)
        px_set = frozenset(zip(ys.tolist(), xs.tolist()))
        lbl    = {(int(y),int(x)): int(grid[y,x]) for y,x in zip(ys,xs)}
        comps.append((px_set, lbl))
    frame_comps.append(comps)

# ── Build index: pixel_set -> list of frames where it appears ─────────────
print("Building pixel-set index …")
px_set_frames = defaultdict(list)  # frozenset -> [fi, ...]
for fi, comps in enumerate(frame_comps):
    for px_set, _ in comps:
        px_set_frames[px_set].append(fi)

all_sets = list(px_set_frames.keys())
print(f"Unique pixel configurations: {len(all_sets)}")

# ── Find jitter twin pairs ────────────────────────────────────────────────
print("Finding jitter twins …")
twins = []  # (set_a, set_b, frames_a, frames_b)

# For each pair of sets that have overlapping bounding boxes and small diff
# Optimise: bucket by approximate centroid
def bbox(px_set):
    ys = [p[0] for p in px_set]; xs = [p[1] for p in px_set]
    return min(ys), max(ys), min(xs), max(xs)

def overlapping(a, b):
    ay0,ay1,ax0,ax1 = bbox(a)
    by0,by1,bx0,bx1 = bbox(b)
    return not (ay1+2 < by0 or by1+2 < ay0 or ax1+2 < bx0 or bx1+2 < ax0)

# Only compare pairs where both have ≥ MIN_APPEARANCES frames
candidate_sets = [s for s in all_sets if len(px_set_frames[s]) >= MIN_APPEARANCES]
print(f"Candidate sets (≥{MIN_APPEARANCES} frames): {len(candidate_sets)}")

seen_pairs = set()
for i, set_a in enumerate(candidate_sets):
    frames_a = px_set_frames[set_a]
    for j, set_b in enumerate(candidate_sets):
        if j <= i: continue
        pair_key = (i, j)
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)

        frames_b = px_set_frames[set_b]

        # Must not co-exist in any frame
        if set(frames_a) & set(frames_b): continue

        # Must be spatially close
        if not overlapping(set_a, set_b): continue

        # Symmetric difference must be small
        diff = len(set_a ^ set_b)
        if diff > MAX_DIFF: continue

        # Check minority run length
        minority, minority_frames = (set_b, frames_b) if len(frames_b) <= len(frames_a) else (set_a, frames_a)
        sorted_min = sorted(minority_frames)
        runs = []; run = [sorted_min[0]]
        for f in sorted_min[1:]:
            if f == run[-1]+1: run.append(f)
            else: runs.append(run); run=[f]
        runs.append(run)
        max_run = max(len(r) for r in runs)
        if max_run > MAX_MINORITY_RUN: continue

        twins.append((set_a, set_b, frames_a, frames_b, diff, max_run))

print(f"Jitter twin pairs found: {len(twins)}")
for set_a, set_b, fa, fb, diff, mr in sorted(twins, key=lambda x: -len(x[2])+len(x[3])):
    maj_set, maj_fr = (set_a, fa) if len(fa)>=len(fb) else (set_b, fb)
    min_set, min_fr = (set_b, fb) if len(fa)>=len(fb) else (set_a, fa)
    ys=[p[0] for p in maj_set]; xs=[p[1] for p in maj_set]
    cx,cy = sum(xs)/len(xs), sum(ys)/len(ys)
    print(f"  approx({cx:.0f},{cy:.0f})  maj={len(maj_fr)}fr  min={len(min_fr)}fr  diff={diff}px  maxrun={mr}")

# ── Apply fixes ───────────────────────────────────────────────────────────
print("\nApplying fixes …")
fixed_data = copy.deepcopy(data)
px_removed = 0; px_added = 0

for set_a, set_b, frames_a, frames_b, diff, max_run in twins:
    # Majority = stable config, minority = jitter
    if len(frames_a) >= len(frames_b):
        stable_set, stable_frames = set_a, frames_a
        jitter_set, jitter_frames = set_b, frames_b
    else:
        stable_set, stable_frames = set_b, frames_b
        jitter_set, jitter_frames = set_a, frames_a

    # Get stable labels from first stable frame
    stable_labels = next(lb for px,lb in frame_comps[stable_frames[0]] if px == stable_set)

    to_remove = jitter_set - stable_set   # in jitter, not in stable -> erase
    to_add    = stable_set - jitter_set   # in stable, not in jitter -> fill

    for fi in jitter_frames:
        flat = list(fixed_data['frames'][fi])
        grid = np.array(data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)

        # Build component map for this frame to verify ownership
        rbc_bin = np.isin(grid,[1,2]).astype(np.uint8)
        nl, lmap, _, _ = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)

        # Find which component label belongs to this jitter set
        our_comp = None
        for c in range(1, nl):
            ys, xs = np.where(lmap == c)
            comp_px = frozenset(zip(ys.tolist(), xs.tolist()))
            overlap = len(comp_px & jitter_set)
            if overlap >= len(jitter_set) * 0.8:
                our_comp = c
                break

        for (py, px) in to_remove:
            # Only erase if this pixel belongs to OUR component
            if our_comp is not None and lmap[py, px] != our_comp:
                continue  # belongs to a neighbour — skip
            if flat[py*ART_W+px] in (1, 2):
                flat[py*ART_W+px] = 0
                px_removed += 1

        for (py, px) in to_add:
            if flat[py*ART_W+px] == 0:
                flat[py*ART_W+px] = stable_labels.get((py,px), 2)
                px_added += 1

        fixed_data['frames'][fi] = flat

print(f"Pixels removed: {px_removed}  added: {px_added}")

# ── Verify ───────────────────────────────────────────────────────────────
bad = 0
for fi in range(N):
    for idx,(b,a) in enumerate(zip(data['frames'][fi], fixed_data['frames'][fi])):
        if b!=a and b in (3,4):
            bad += 1; print(f"PROTECTED fi={fi} idx={idx} {b}→{a}")
print(f"Protected pixels touched: {bad} {'✓' if bad==0 else '← PROBLEM'}")

# ── Save ─────────────────────────────────────────────────────────────────
backup = art_path.with_suffix('.json.prejitter3')
shutil.copy(art_path, backup)
with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',',':'))
print(f"Done → {art_path}")
