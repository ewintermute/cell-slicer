# Cell Slicer — Session Notes
**Last updated:** 2026-04-10

---

## Current state

The pixel art (`docs/pixel_art.json`, 412 frames, 64×48) is fully annotated with labels:
- 0 = background
- 1 = RBC fill (red)
- 2 = RBC edge (purple)
- 3 = neutrophil (green)
- 4 = microbe (yellow)

Clem has completed 12+ manual edit sessions covering all 412 frames. The microbe track (`docs/microbe_positions.json`) is also fully annotated.

---

## What we worked on today (2026-04-10)

### RBC jitter fix (rbc_dejitter3.py)
- Previous attempt (rbc_dejitter2.py) was already reverted.
- rbc_dejitter3.py uses canonical disc scoring to pick the better twin config.
- First run produced broken sprites: multi-component pixel sets (two separate adjacent RBCs) were being matched as "jitter twins."
- **Fix 1:** Added `MAX_CENTROID_DIST = 2.0` guard — didn't help (centroids were coincidentally close).
- **Fix 2:** Added single-connected-component guard — filtered the bad pair. Down to 2 pairs.
- Remaining broken sprite at frames 111–113, approx(46,1): pre-existing, not caused by jitter fix.
- **Programmatic fix:** Stamped canonical disc at (cx=46, cy=1) in frames 111–113. Committed `a9874ba`.
- **Jitter fix committed:** `a5bb927` (then reverted `eb6bef2` when broken sprites found), re-applied after fix as `a9874ba`.

### Edit session merge
- Clem pushed new `output/pixel_art_edited.json` and `output/microbe_positions.json`.
- Merged: 183 frames, 1926 pixels changed. Microbe positions unchanged.
- Committed `e6cace6`.

### 32×48 / 64×32 / 48×32 cropped viewers
- New sub-project: crop the 64×48 pixel map to various sizes with smooth pan or jump-cut.
- Key insight: `pixel_art.json` has `src_indices` — maps art frame index → source video frame. Must use this for video background sync (not `fi / fps`).
- Video sync bug fixed: closure-captured `seeked` handler so each seek draws the correct frame.
- Three viewers live:
  - https://ewintermute.github.io/cell-slicer/pixel_art_32.html (32×48, horizontal pan)
  - https://ewintermute.github.io/cell-slicer/pixel_art_64x32.html (64×32, vertical pan)
  - https://ewintermute.github.io/cell-slicer/pixel_art_48x32.html (48×32, 2-axis) ← current focus

### 48×32 jump-cut scheduler (current best)
- **Goal:** keep chased microbe in frame as long as possible per cut; tiebreak on neutrophil coverage.
- **Key insight:** label-4 has TWO connected components — one is the chased bacterium (leftmost x centroid), one is a second microbe far right. Algorithm uses only the leftmost component as the target.
- **Algorithm:** intersect valid windows frame-by-frame from each cut point; find the window that keeps the microbe in frame the longest; among tied windows pick the one with most neutrophil pixels.
- **Result: 3 cuts total:**
  - Frames 0–124: window (x0=0, y0=0)
  - Frames 125–295: window (x0=1, y0=16) — lower-left, captures neutrophil during chase
  - Frames 296–415: window (x0=0, y0=1)
- Frame 192 check: 256/256 neutrophil pixels in crop ✓, chased microbe centroid in frame ✓
- Committed `129f9ae`.

---

## Outstanding / next steps

1. **Review 48×32 jump-cut result** — Clem reviewing at https://ewintermute.github.io/cell-slicer/pixel_art_48x32.html
2. **Potential outputs** (not started):
   - Export as animated GIF or spritesheet
   - Rendered overlay video with microbe label
   - Analysis of neutrophil–microbe distance over time

---

## Key files

| Path | Description |
|------|-------------|
| `docs/pixel_art.json` | Main pixel art data (412 frames, 64×48) |
| `docs/pixel_art_48x32.json` | 48×32 jump-cut crop (3 cuts, microbe-centered) |
| `docs/pixel_art_32.json` | 32×48 smooth-pan crop |
| `docs/pixel_art_64x32.json` | 64×32 smooth-pan crop |
| `docs/microbe_positions.json` | Microbe centroid + pixels per frame |
| `docs/pixel_editor.html` | Browser editor (64×48) |
| `docs/pixel_art_48x32.html` | 48×32 viewer |
| `rbc_dejitter3.py` | RBC jitter fix (canonical-aware, single-component guard) |
| `source-movie/chase-original.mp4` | Source video (426 frames, 320×240, 15fps) |

---

## SSH setup (needed on container restart)

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp ~/workspace/credentials/github_ed25519 ~/.ssh/github_ed25519
chmod 600 ~/.ssh/github_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
cat > ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_ed25519
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```
