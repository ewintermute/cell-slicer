# Edited Cell Chase — Export Package

This folder contains the final exported videos and frames for **Project Bioshrek**,
derived from a brightfield microscopy recording of a neutrophil chasing a bacterium
through a field of red blood cells.

---

## Source Material

**Original video:** `chase-original.mp4`
- 426 frames, 320×240 px, 15 fps
- Brightfield microscopy, single channel (grayscale rendered as RGB)
- Contains 10 stage-jump frames where the microscope stage moved abruptly

**Stage-jump frames (excluded from all outputs):**
Frame indices 8, 43, 70, 85, 118, 132, 166, 255, 356, 403
These frames show a discontinuous displacement of the field of view and were removed,
leaving **416 usable frames**.

---

## Output Files

### 1. `uncropped.mp4` — Full-frame, jump-frames removed
- **Resolution:** 320×240 px (original source resolution)
- **Duration:** 416 frames @ 15 fps ≈ 27.7 s
- **How it was made:**
  The 416 non-jump source frames were extracted in order and re-encoded as a
  continuous video. No spatial cropping or colour correction was applied.
  Frame `i` in this video corresponds to source frame `src_indices[i]` in the
  original 426-frame recording.

---

### 2. `cropped.mp4` — Jump-cut crop, following the action
- **Resolution:** 240×160 px (48×32 art pixels × 5 px/block)
- **Duration:** 416 frames @ 15 fps ≈ 27.7 s
- **How it was made:**
  A 48×32 art-pixel crop window (= 240×160 source pixels) was applied to each frame.
  The window position was chosen by an automated scheduler to keep the chased bacterium
  in frame for as long as possible, with neutrophil coverage as a tiebreaker.
  The crop window changes at **2 jump-cut transitions**:

  | Segment         | Art frames | x0 (art px) | y0 (art px) | Source px offset    |
  |-----------------|-----------|-------------|-------------|---------------------|
  | Cut 1 (intro)   | 0 – 41    | 0           | 9 – 11 *    | x=0, y=45–55 px     |
  | Cut 2 (chase)   | 42 – 222  | 0           | 13          | x=0, y=65 px        |
  | Cut 3 (pursuit) | 223 – 415 | 0           | 3           | x=0, y=15 px        |

  \* Cut 1 y0 tracks the neutrophil top row per frame (9, 10, or 11) to keep the
  cell fully in frame as it moves slightly.

  All coordinates are in **art pixels** (1 art px = 5×5 source pixels).
  To convert: `source_x = art_x * 5`, `source_y = art_y * 5`.

---

### 3. `pixel-art.mp4` — Fluorescent protein pixel art animation
- **Resolution:** 480×320 px (48×32 art pixels × 10 px/block)
- **Duration:** 416 frames @ 15 fps ≈ 27.7 s
- **How it was made:**
  Each frame of the annotated pixel art (48×32 grid, same crop window as above)
  was rendered using fluorescent protein colours matching the agar pixel art palette.
  Pixels were scaled 10× with nearest-neighbour interpolation (no blending).

  **Colour / label mapping:**

  | Label | Cell type              | Fluorescent protein | Hex colour |
  |-------|------------------------|---------------------|------------|
  | 0     | Background             | — (no FP)           | `#111111`  |
  | 1     | RBC interior           | mKO2 (565 nm)       | `#FF8000`  |
  | 2     | RBC edge               | mRFP (607 nm)       | `#FF0000`  |
  | 3     | Neutrophil cytoplasm   | mTurquoise2 (474 nm)| `#00C8FF`  |
  | 4     | Chased bacterium       | Venus (528 nm)      | `#AAFF00`  |
  | 5     | Neutrophil leading edge| Azurite (448 nm)    | `#3333FF`  |
  | 6     | Cytoplasmic organelles | Electra2 (456 nm)   | `#4747FF`  |

  **Leading edge (label 5):**
  The forward-facing boundary of the neutrophil is marked in Azurite. The direction
  is computed as the vector from the neutrophil centroid toward the chased bacterium,
  smoothed with a 31-frame uniform filter (max change: 4.6°/frame). The outermost
  boundary pixels of the neutrophil in this direction (top 40% by dot product) are
  labelled as leading edge.

  **Cytoplasmic organelles (label 6):**
  20 simulated organelle particles (1 art px each) are placed inside the neutrophil
  cytoplasm. Each frame they undergo:
  - **Translation** with the neutrophil centroid (they move with the cell)
  - **Rotational streaming** at 0.030 rad/frame (~1 revolution per 210 frames),
    simulating actin-driven cytoplasmic flow
  - **Brownian diffusion** σ = 0.45 px/frame (random walk)
  - Particles that exit the neutrophil are snapped back to the nearest interior pixel.
  - On stage-jump frames, particles are re-seeded uniformly inside the cell.

---

### 4. `pixel-art-frames/` — Individual PNG frames
- **Resolution:** 48×32 px (native art resolution, no upscaling)
- **Format:** PNG, RGB
- **Files:** `frame_0000.png` … `frame_0415.png` (416 files)
- Same FP colour mapping as `pixel-art.mp4`.
- Suitable for import into sprite editors, further compositing, or GIF generation.

---

## Reproducing from the Original Video

To reproduce these outputs from `chase-original.mp4`:

1. **Determine skip frames** — detect stage jumps by frame-to-frame absolute
   difference; frames 8, 43, 70, 85, 118, 132, 166, 255, 356, 403 exceed the
   threshold. Exclude them to get 416 frames.

2. **Uncropped movie** — re-encode the 416 remaining frames in order at 15 fps.

3. **Cropped movie** — for each of the 416 frames, extract the 240×160 px region
   starting at the source pixel offset given in the cut table above, scaled by the
   per-frame x0/y0 art-pixel offsets in `pixel_art_org_48x32.json`.

4. **Pixel art** — segment each 5×5 source pixel block into one of the 6 cell labels
   using the annotation data in `pixel_art_org_48x32.json`, then render using the
   FP colour table above. The annotation was produced semi-automatically
   (GrabCut + watershed segmentation, followed by extensive manual correction)
   and is stored as a flat array of 48×32 label values per frame.

---

*Generated 2026-04-10. Source repo: https://github.com/ewintermute/cell-slicer*
