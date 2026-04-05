# Cell Slicer — Project Progress Notes

**Last updated:** 2026-04-05  
**Current pipeline:** `pipeline.py` (v7.1)  
**Output:** `output/chase-segmented.mp4`  
**Viewer:** https://ewintermute.github.io/cell-slicer/  
**Repo:** https://github.com/ewintermute/cell-slicer

---

## What this project does

Segments three cell types in a brightfield microscopy video (`source-movie/chase-original.mp4`, 426 frames, 320×240, 15fps):
- **Neutrophil** (green) — one large amoeboid cell that chases a microbe
- **Red blood cells / RBCs** (red) — many static round cells
- ~~Microbe~~ (dropped — too small to detect reliably at this resolution)

Output is an annotated MP4 with coloured semi-transparent overlays showing cell boundaries.

---

## Current state (v7.1)

**Metrics vs 1497 manual correction points across 160 annotated frames:**
- Neutrophil: Precision=100% Recall=74% F1=85%
- RBC: Precision=98% Recall=79% F1=87%

**Architecture:** Correction-driven, 3-pass pipeline:
1. Pass 1: compute raw per-frame masks using GrabCut (neutrophil) + watershed (RBCs)
2. Pass 2: temporal smoothing — suppress single-frame area spikes
3. Pass 3: render overlay with hole-fill and min-size filter applied last

---

## Manual correction system

The annotation viewer (`docs/index.html`) lets Clem click on the segmented video to mark:
- **Neutrophil** (N key) — click inside the neutrophil body
- **RBC** (R key) — click on an RBC that's misidentified or missing
- **Background** (B key) — click on a region wrongly labelled as a cell
- **Erase** (E key / right-click) — remove a nearby marker

Corrections saved to `output/corrections.json`. Import/Export from the viewer.  
To re-run with new corrections: `python3 pipeline.py --corrections output/corrections.json`

Corrections never propagate across stage jump frames.

---

## What works well

- **GrabCut seeded by all correction points** for neutrophil — biggest single improvement, jumped F1 from 27% to 85%
- **Watershed + correction-point region lookup** for RBCs — gives actual cell contours rather than blunt discs
- **Correction interpolation** between annotated frames (centroid-shifted point sets) — avoids needing to annotate every frame
- **Stage-jump isolation** — 10 detected jumps partition the video; corrections never bleed across them
- **Temporal smoothing** — suppresses single-frame mask spikes (e.g. neutrophil suddenly covers an RBC for 1 frame)
- **Hole filling** in RBC masks after smoothing — ensures solid interiors
- **Minimum RBC size** filter after smoothing — removes sub-threshold fragments
- **RBC density map** (`output/rbc_density.npy`) — accumulated Hough detections across all 426 frames; reliable seed positions for watershed

---

## Approaches that were tried and DIDN'T work

### Neutrophil detection without corrections

1. **Background subtraction (rolling median)** — Failed. The neutrophil is slow-moving and large, so it ends up in the median background and becomes invisible to the subtraction. RBC halos produced stronger signal than the neutrophil itself.

2. **Inter-frame diff (|frame_t - frame_{t-lag}|)** — Partially worked for motion signal but RBC halos produce sharp ring-shaped noise comparable to the neutrophil's diffuse signal. Could never cleanly separate neutrophil from RBC noise.

3. **Adaptive thresholding + largest blob** — The neutrophil's interior is mid-grey (~120-160 enhanced), same range as RBC interiors. Thresholding cannot separate them.

4. **Non-circular region after Hough exclusion** — Built a "stable RBC territory" map and subtracted it. The neutrophil region was never reliably the largest remaining blob because edge artifacts and merged cell blobs were larger.

5. **Cellpose (deep learning cell segmentation)** — Installed cellpose v3 with the `cyto3` model. Inference was ~30 seconds per frame on CPU → ~3.5 hours for 426 frames. Not viable in this environment. The `cpsam` model (cellpose v4) was even heavier (1.15GB). Abandoned.

6. **GrabCut with only centroid seed (Tier 2 approach)** — Used centroid of the neutrophil (from corrections) as sole GrabCut seed. Result: leaked onto nearby RBCs because the seed disc covered multiple cells, and GrabCut's colour model couldn't distinguish.

### RBC detection

7. **Hough Circle Transform alone** — Detected ~20 circles per frame but missed edge cells entirely (partially off-screen). Also: RBC diameter was initially miscalibrated (tried 16px, actual is ~37px diameter / 17px radius). After fixing scale, Hough gave ~21 circles per frame but density was too low for reliable seeding.

8. **Blunt disc painting at correction points** — First approach: paint an 18px disc wherever Clem clicked. Result: circles bore no relation to actual cell contours. Replaced by watershed region lookup.

9. **Flood-fill from correction point on cell_bodies mask** — Tried flood-filling from the correction point to get the cell contour. Failed: the `cell_bodies` mask used an aggressive morphological close (k=9) to bridge neutrophil pseudopods, which merged neighbouring RBCs into one blob. The flood fill then covered multiple cells.

### Image quality

10. **CLAHE enhancement** — Works, kept throughout.

11. **DoG / HOUGH_GRADIENT_ALT** — Tried for better RBC ring detection. Only found 3 circles even with low sensitivity. Abandoned.

---

## Key technical facts about the video

- **Frame size:** 320×240, 15fps, 426 frames
- **RBC diameter:** ~37px / radius ~17px
- **Neutrophil interior:** mid-grey, 115–166 enhanced intensity — NOT detectable by simple thresholding (same range as RBC interiors and background)
- **RBC central pallor:** bright centre (140–220 enhanced), dark rim (~40–100) — the rim IS detectable but merges with neighbours
- **Stage jumps:** 10 frames [8, 43, 70, 85, 118, 132, 166, 255, 356, 403] where the microscope stage moved. The frame AFTER a jump cluster is normal (don't mark it as "stage moving")
- **Consecutive jump detection bug:** A frame with high diff from the previous frame may just be comparing against a displaced frame, not itself a jump. Only the FIRST frame of a consecutive high-diff run is a real jump.
- **No GPU in this environment** — all inference is CPU-only

---

## File structure

```
2026-04-04 cell slicer/
├── pipeline.py              # main segmentation script
├── PROGRESS.md              # this file
├── source-movie/
│   └── chase-original.mp4  # input video
├── output/
│   ├── chase-segmented.mp4  # current output
│   ├── corrections.json     # manual annotations (1497 markers, 160 frames)
│   └── rbc_density.npy      # accumulated Hough density map (all 426 frames)
└── docs/                    # GitHub Pages viewer
    ├── index.html
    ├── chase-original.mp4
    └── chase-segmented.mp4
```

---

## How to continue

**Re-run with updated corrections:**
```bash
cd "/home/node/workspace/2026-04-04 cell slicer"
PATH=$PATH:/home/node/.local/bin python3 pipeline.py --corrections output/corrections.json
```

**Rebuild density map (only needed if you change RBC radius):**
```bash
python3 pipeline.py --build-density-map
```

**Score current output vs corrections:**
```python
import json, numpy as np, cv2, imageio.v3 as iio
with open('output/corrections.json') as f: corrections = json.load(f)
out_vid = iio.imread('output/chase-segmented.mp4', plugin='pyav', index=None)
# Then loop over annotated frames checking is_green / is_red at each point
```

---

## What to try next to improve recall

- The 26% neutrophil recall gap is mostly annotation points at the **outer edges** of the neutrophil body that GrabCut doesn't grow out to. Adding more correction points near the cell boundary of the neutrophil in a few frames will expand the GrabCut FG region.
- The 21% RBC recall gap is largely cells at **image borders** (x<10, y<10, etc.) that the Hough/watershed baseline misses, and cells in segments with no nearby correction frames.
- Temporal smoothing window is set to ±5 frames (`TEMPORAL_WINDOW = 5`). Increasing to 7 might help stability but could blur real slow biological changes.
- For the longer video segments without corrections (frames 75–84, 86–117, etc.), adding even 2–3 correction points per segment would enable Tier-2 interpolation instead of falling back to Tier-3 (motion-only).
