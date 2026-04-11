"""
rbc_dejitter3.py — Canonical-aware pixel-exact jitter correction.

No centroid tracking. No stamp_sprite. No bounding-box erases.

Key improvement over rbc_dejitter2.py:
  - When a jitter twin pair is found, BOTH configs are scored against the
    canonical disc template at their shared centroid.
  - We snap ALL frames (majority AND minority) to whichever config scores
    BETTER against the canonical, not simply to the majority by frame count.
  - Before applying any erase, verify the result still forms a valid
    canonical-like disc (≥ MIN_CANON_SCORE of canonical pixels present).
    If the fix would break the sprite, skip it entirely.

Algorithm:
  1. For every frame, extract every RBC connected component as a frozenset
     of (y,x) pixels.
  2. Collect all unique pixel sets across all frames.
  3. For each pair of pixel sets A and B that:
       - Overlap spatially (bounding boxes nearby)
       - Differ by ≤ MAX_DIFF pixels (symmetric difference)
       - Both appear in at least MIN_APPEARANCES frames
       - Are never present in the SAME frame simultaneously
       - Have a minority-config max consecutive run ≤ MAX_MINORITY_RUN
     -> These are jitter twins.
  4. For each twin pair:
       a. Find the canonical disc centre (grid search near shared centroid).
       b. Score both pixel sets: how many canonical pixels does each cover?
       c. The BETTER-scoring set is the target config.
       d. Snap all frames with the worse config to the better config,
          touching only pixels in the symmetric difference.
       e. After applying: verify the resulting component still has
          ≥ MIN_CANON_SCORE fraction of canonical pixels — if not, skip.
"""

import json, copy, shutil, numpy as np, cv2
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MAX_DIFF          = 9    # max pixels different between twin pixel sets
MIN_APPEARANCES   = 2    # each twin must appear in at least this many frames
MAX_MINORITY_RUN  = 8    # minority config max consecutive-frame run
MIN_AREA          = 10   # ignore tiny fragments
MIN_CANON_SCORE   = 0.80 # target config must cover ≥ 80% of canonical pixels
R                 = 3    # RBC radius in art pixels

ART_H, ART_W = 48, 64
DISC = [(dy,dx) for dy in range(-R,R+1) for dx in range(-R,R+1) if dy*dy+dx*dx<=R*R]
k3   = np.ones((3,3), np.uint8)

# ── Canonical sprite ──────────────────────────────────────────────────────────
def canonical_pixels(cy, cx):
    """Return dict of (y,x)->label for the canonical disc at (cy,cx)."""
    canvas = np.zeros((ART_H, ART_W), dtype=bool)
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            canvas[ny, nx] = True
    eroded = cv2.erode(canvas.astype(np.uint8)*255, k3)
    result = {}
    for dy, dx in DISC:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < ART_H and 0 <= nx < ART_W:
            result[(ny,nx)] = 2 if eroded[ny,nx] == 0 else 1
    return result

def best_canonical(px_set):
    """
    Find the canonical disc centre that best matches px_set.
    Returns (canon_dict, score_fraction, centre_cy, centre_cx).
    """
    ys = [p[0] for p in px_set]; xs = [p[1] for p in px_set]
    cy_f = sum(ys)/len(ys); cx_f = sum(xs)/len(xs)
    best_score, best_canon, best_cy, best_cx = -1, None, None, None
    for try_cy in range(int(cy_f)-3, int(cy_f)+4):
        for try_cx in range(int(cx_f)-3, int(cx_f)+4):
            if not (0 <= try_cy < ART_H and 0 <= try_cx < ART_W):
                continue
            canon = canonical_pixels(try_cy, try_cx)
            present = sum(1 for p in canon if p in px_set)
            score = present / len(canon)
            if score > best_score:
                best_score = score
                best_canon = canon
                best_cy, best_cx = try_cy, try_cx
    return best_canon, best_score, best_cy, best_cx

# ── Load ──────────────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)
N = data['n_frames']

# ── Extract all components per frame ─────────────────────────────────────────
print("Extracting components …")
frame_comps = []
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

# ── Build index: pixel_set -> list of frames ──────────────────────────────────
print("Building pixel-set index …")
px_set_frames = defaultdict(list)
for fi, comps in enumerate(frame_comps):
    for px_set, _ in comps:
        px_set_frames[px_set].append(fi)

all_sets = list(px_set_frames.keys())
print(f"Unique pixel configurations: {len(all_sets)}")

# ── Find jitter twin pairs ────────────────────────────────────────────────────
print("Finding jitter twins …")

def bbox(px_set):
    ys = [p[0] for p in px_set]; xs = [p[1] for p in px_set]
    return min(ys), max(ys), min(xs), max(xs)

def overlapping(a, b):
    ay0,ay1,ax0,ax1 = bbox(a)
    by0,by1,bx0,bx1 = bbox(b)
    return not (ay1+2 < by0 or by1+2 < ay0 or ax1+2 < bx0 or bx1+2 < ax0)

def centroid(px_set):
    ys = [p[0] for p in px_set]; xs = [p[1] for p in px_set]
    return sum(ys)/len(ys), sum(xs)/len(xs)

MAX_CENTROID_DIST = 2.0  # twin centroids must be within this many pixels

candidate_sets = [s for s in all_sets if len(px_set_frames[s]) >= MIN_APPEARANCES]
print(f"Candidate sets (≥{MIN_APPEARANCES} frames): {len(candidate_sets)}")

twins = []  # (set_a, set_b, frames_a, frames_b, diff, max_run)
seen_pairs = set()
for i, set_a in enumerate(candidate_sets):
    frames_a = px_set_frames[set_a]
    for j, set_b in enumerate(candidate_sets):
        if j <= i: continue
        pair_key = (i, j)
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)

        frames_b = px_set_frames[set_b]
        if set(frames_a) & set(frames_b): continue  # co-exist → not twins
        if not overlapping(set_a, set_b): continue

        diff = len(set_a ^ set_b)
        if diff > MAX_DIFF: continue

        # Centroids must be nearly identical — guards against two distinct
        # nearby RBCs being mistakenly paired as one jittering RBC
        cy_a, cx_a = centroid(set_a)
        cy_b, cx_b = centroid(set_b)
        cdist = ((cy_a-cy_b)**2 + (cx_a-cx_b)**2)**0.5
        if cdist > MAX_CENTROID_DIST: continue

        # Both configs must be single connected components — a multi-blob
        # pixel set is two separate RBCs, not one jittering RBC
        def is_single_component(px_set):
            if not px_set: return False
            ys = [p[0] for p in px_set]; xs = [p[1] for p in px_set]
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            canvas = np.zeros((y1-y0+1, x1-x0+1), dtype=np.uint8)
            for py, px in px_set:
                canvas[py-y0, px-x0] = 1
            nl, _ = cv2.connectedComponents(canvas, connectivity=8)
            return nl == 2  # 1 background + 1 foreground
        if not is_single_component(set_a): continue
        if not is_single_component(set_b): continue

        # Check minority run length
        minority_frames = frames_b if len(frames_b) <= len(frames_a) else frames_a
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

# ── Score each twin pair against canonical ────────────────────────────────────
print("\nScoring twins against canonical disc …")
fixes = []   # (target_set, target_labels, frames_to_fix, to_remove, to_add, canon_score)
skipped_canon = 0

for set_a, set_b, frames_a, frames_b, diff, max_run in twins:
    canon_a, score_a, cy_a, cx_a = best_canonical(set_a)
    canon_b, score_b, cy_b, cx_b = best_canonical(set_b)

    # Pick the better-scoring config as the target
    if score_a >= score_b:
        target_set, target_frames = set_a, frames_a
        wrong_set,  wrong_frames  = set_b, frames_b
        target_score, target_canon = score_a, canon_a
        target_cy, target_cx       = cy_a, cx_a
    else:
        target_set, target_frames = set_b, frames_b
        wrong_set,  wrong_frames  = set_a, frames_a
        target_score, target_canon = score_b, canon_b
        target_cy, target_cx       = cy_b, cx_b

    # Gate: target must score well enough against canonical
    if target_score < MIN_CANON_SCORE:
        print(f"  SKIP approx({target_cx},{target_cy}): target score {target_score:.2f} < {MIN_CANON_SCORE}")
        skipped_canon += 1
        continue

    to_remove = wrong_set - target_set
    to_add    = target_set - wrong_set
    target_labels = next(
        lb for px, lb in frame_comps[target_frames[0]] if px == target_set
    )

    # Verify: would the resulting sprite still pass canonical check?
    # Simulate the fix on wrong_set and score it
    simulated = (wrong_set - to_remove) | to_add
    sim_score = sum(1 for p in target_canon if p in simulated) / len(target_canon)
    if sim_score < MIN_CANON_SCORE:
        print(f"  SKIP approx({target_cx},{target_cy}): post-fix sim score {sim_score:.2f} < {MIN_CANON_SCORE}")
        skipped_canon += 1
        continue

    maj_n = max(len(target_frames), len(wrong_frames))
    min_n = min(len(target_frames), len(wrong_frames))
    print(f"  FIX  approx({target_cx:2d},{target_cy:2d})  target_score={target_score:.2f}  "
          f"frames: {maj_n}→{len(target_frames)} right, {min_n}→{len(wrong_frames)} to fix  diff={diff}px")

    fixes.append((target_set, target_labels, wrong_frames, to_remove, to_add))

print(f"\nFixes to apply: {len(fixes)}  (skipped {skipped_canon} pairs — failed canonical gate)")

# ── Apply fixes ───────────────────────────────────────────────────────────────
print("\nApplying fixes …")
fixed_data = copy.deepcopy(data)
px_removed = 0; px_added = 0; px_skipped = 0

for target_set, target_labels, wrong_frames, to_remove, to_add in fixes:
    for fi in wrong_frames:
        flat = list(fixed_data['frames'][fi])
        grid = np.array(data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)

        # Build component map to verify pixel ownership
        rbc_bin = np.isin(grid, [1,2]).astype(np.uint8)
        nl, lmap, _, _ = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)

        # Find which component this wrong_set belongs to
        our_comp = None
        for c in range(1, nl):
            ys, xs = np.where(lmap == c)
            comp_px = frozenset(zip(ys.tolist(), xs.tolist()))
            if len(comp_px & (to_remove | (target_set & wrong_set))) >= len(to_remove | (target_set & wrong_set)) * 0.7:
                our_comp = c
                break

        for (py, px) in to_remove:
            # Only erase if this pixel belongs to OUR component, not a neighbour
            if our_comp is not None and lmap[py, px] != our_comp:
                px_skipped += 1
                continue
            # Never touch neutrophil or microbe
            if flat[py*ART_W+px] in (3, 4):
                px_skipped += 1
                continue
            if flat[py*ART_W+px] in (1, 2):
                flat[py*ART_W+px] = 0
                px_removed += 1

        for (py, px) in to_add:
            if flat[py*ART_W+px] in (3, 4):
                px_skipped += 1
                continue
            if flat[py*ART_W+px] == 0:
                flat[py*ART_W+px] = target_labels.get((py,px), 2)
                px_added += 1
            else:
                px_skipped += 1

        fixed_data['frames'][fi] = flat

print(f"Pixels removed: {px_removed}  added: {px_added}  skipped: {px_skipped}")

# ── Verify: no protected pixels touched ──────────────────────────────────────
print("\nVerifying protected pixels …")
bad = 0
for fi in range(N):
    for idx, (b, a) in enumerate(zip(data['frames'][fi], fixed_data['frames'][fi])):
        if b != a and b in (3, 4):
            bad += 1
            y, x = divmod(idx, ART_W)
            print(f"  PROTECTED fi={fi} ({x},{y}) {b}→{a}")
print(f"Protected pixels touched: {bad} {'✓' if bad==0 else '← PROBLEM'}")

# ── Verify: canonical integrity on changed sprites ────────────────────────────
print("\nVerifying canonical integrity of changed frames …")
canon_violations = 0
changed_frames = set()
for _, _, wrong_frames, _, _ in fixes:
    changed_frames.update(wrong_frames)

for fi in changed_frames:
    grid_after = np.array(fixed_data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)
    rbc_bin = np.isin(grid_after, [1,2]).astype(np.uint8)
    nl, lmap, stats, cents = cv2.connectedComponentsWithStats(rbc_bin, connectivity=8)
    for c in range(1, nl):
        if stats[c, cv2.CC_STAT_AREA] < MIN_AREA: continue
        ys, xs = np.where(lmap == c)
        px_set = frozenset(zip(ys.tolist(), xs.tolist()))
        _, score, cy, cx = best_canonical(px_set)
        if score < MIN_CANON_SCORE:
            canon_violations += 1
            print(f"  CANON FAIL fi={fi} comp@({cx},{cy}) score={score:.2f}")

print(f"Canonical violations in changed frames: {canon_violations} {'✓' if canon_violations==0 else '← PROBLEM'}")

total_changed = sum(
    sum(1 for b, a in zip(data['frames'][fi], fixed_data['frames'][fi]) if b != a)
    for fi in range(N)
)
print(f"\nTotal pixels changed across all frames: {total_changed}")

# ── Save ──────────────────────────────────────────────────────────────────────
backup = art_path.with_suffix('.json.prejitter3')
shutil.copy(art_path, backup)
with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',',':'))
print(f"Done → {art_path}")
print(f"Backup → {backup}")
