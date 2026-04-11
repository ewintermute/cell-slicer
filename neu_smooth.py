"""
neu_smooth.py

Light temporal smoothing on neutrophil pixels.

Strategy: per-frame majority vote over a ±HALF_WIN frame window.
  - A pixel stays ON  if it is neutrophil in >= THRESH of the window frames
  - A pixel turns OFF if it is neutrophil in <  THRESH of the window frames
  - Additions are conservative: a background pixel is only turned ON if it
    appears in ALL other window frames (not just majority) — we suppress
    drop-outs but do not hallucinate new territory
  - RBC and microbe pixels are never overwritten (NEU loses to them)

Parameters:
  HALF_WIN = 2  →  5-frame window  (safe given median centroid drift ~0.45px/frame)
  THRESH   = 3  →  majority of 5
"""

import json, copy, shutil
from pathlib import Path
import numpy as np

HALF_WIN  = 2
THRESH    = 3      # must be >= this many window frames to stay ON
NEU       = 3
RBC_LABELS = {1, 2}
MICROBE    = 4
ART_H, ART_W = 48, 64

# ── Load ──────────────────────────────────────────────────────────────────────
art_path = Path("docs/pixel_art.json")
print(f"Loading {art_path} …")
with open(art_path) as f:
    data = json.load(f)

n = data['n_frames']
frames = [np.array(data['frames'][fi], dtype=np.uint8).reshape(ART_H, ART_W)
          for fi in range(n)]

# ── Smooth ────────────────────────────────────────────────────────────────────
fixed = [f.copy() for f in frames]
pixels_added = 0
pixels_removed = 0

for fi in range(n):
    window = list(range(max(0, fi - HALF_WIN), min(n, fi + HALF_WIN + 1)))
    w = len(window)

    # Count how many window frames have NEU at each pixel
    vote = np.zeros((ART_H, ART_W), dtype=np.uint8)
    for wfi in window:
        vote += (frames[wfi] == NEU).astype(np.uint8)

    cur  = frames[fi]
    out  = fixed[fi]

    # Pixels currently NEU that lose majority → remove
    currently_neu = (cur == NEU)
    loses_majority = (vote < THRESH)
    remove_mask = currently_neu & loses_majority
    # Don't remove pixels that are protected by other labels (shouldn't be, but safe)
    out[remove_mask] = 0
    pixels_removed += int(remove_mask.sum())

    # Pixels currently background that appear in ALL other window frames → add
    other_frames = [frames[wfi] for wfi in window if wfi != fi]
    if other_frames:
        in_all_others = np.ones((ART_H, ART_W), dtype=bool)
        for of in other_frames:
            in_all_others &= (of == NEU)
        currently_bg   = (cur == 0)
        add_mask       = currently_bg & in_all_others
        # Never overwrite RBC or microbe
        protected      = np.isin(out, list(RBC_LABELS) + [MICROBE])
        add_mask      &= ~protected
        out[add_mask]  = NEU
        pixels_added  += int(add_mask.sum())

print(f"Pixels removed (flickers):  {pixels_removed}")
print(f"Pixels added   (drop-outs): {pixels_added}")
print(f"Net change:                 {pixels_added - pixels_removed:+d}")

# ── Verify no RBC/microbe pixels touched ─────────────────────────────────────
bad = 0
for fi in range(n):
    for idx in range(ART_H * ART_W):
        b = frames[fi].flat[idx]
        a = fixed[fi].flat[idx]
        if b != a and b in RBC_LABELS | {MICROBE}:
            bad += 1
print(f"Protected pixels overwritten: {bad}  {'(NONE — good)' if bad==0 else '← PROBLEM'}")

# ── Save ──────────────────────────────────────────────────────────────────────
backup = art_path.with_suffix('.json.preneu')
shutil.copy(art_path, backup)
print(f"Backup → {backup}")

fixed_data = copy.deepcopy(data)
for fi in range(n):
    fixed_data['frames'][fi] = fixed[fi].flatten().tolist()

with open(art_path, 'w') as f:
    json.dump(fixed_data, f, separators=(',', ':'))
print(f"Done → {art_path}")
