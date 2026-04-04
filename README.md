# Cell Slicer 🔬

AI-assisted image analysis pipeline for segmenting and tracking cells in brightfield/phase-contrast microscopy video.

## What it does

Identifies and marks boundaries of 2 cell types across all frames of a microscopy video:

| Cell type | Colour | Detection method |
|-----------|--------|-----------------|
| **Neutrophil** | 🟢 Green | Largest unseeded cell body region after RBC watershed |
| **Red Blood Cells** | 🔴 Red | Hough circle seeds → marker-controlled watershed on dark-border mask |

Output is an MP4 with semi-transparent coloured masks and outlines overlaid on the original footage.

### Detection strategy

1. **Cell body mask** — threshold dark regions (cell borders), fill holes (handles RBC central pallor), remove sub-cell noise → solid binary mask of all cell material
2. **RBC seeds** — Hough Circle Transform places one seed per RBC centre; these become watershed markers
3. **Marker-controlled watershed** — expands seeds into actual cell body shapes; prevents the RBC central pallor creating spurious internal regions
4. **Neutrophil** — any large connected component in the cell body mask *not* claimed by a Hough seed; the neutrophil is non-round and never gets a Hough seed

## Source material

`source-movie/chase-original.mp4` — 426 frames @ 15fps, 320×240px
Depicts a neutrophil chasing and catching a microbe in a whole-blood field.

## Running

```bash
python3 pipeline.py
```

Default I/O:
- Input:  `source-movie/chase-original.mp4`
- Output: `output/chase-segmented.mp4`

### Options

```
--input PATH              Source video (default: source-movie/chase-original.mp4)
--output PATH             Output video (default: output/chase-segmented.mp4)
--rbc-radius FLOAT        Expected RBC radius in pixels (default: 17 → ~34px diameter)
--dark-thresh INT         Intensity threshold for dark cell borders (default: 100)
--stage-jump-threshold    Mean pixel diff to flag stage movement (default: 18)
--test INT                Process only first N frames (for quick testing)
```

### Example

```bash
# Quick test on first 30 frames
python3 pipeline.py --test 30

# Full run with custom RBC diameter
python3 pipeline.py --diameter-rbc 38
```

## Dependencies

```bash
pip install opencv-python-headless imageio[pyav] scikit-image tqdm numpy
```

## Pipeline details

### Stage jump detection
Frame-to-frame mean absolute difference is computed. If it exceeds the threshold, the frame is marked "STAGE MOVE" and segmentation is skipped for that frame (avoids false detections on blurry/repositioning frames).

### RBC detection
Hough Circle Transform on CLAHE-enhanced grayscale. Parameters are tuned for the ~37px RBC diameter in this dataset.

### Neutrophil detection
Rolling median background model (25-frame window) combined with adaptive thresholding. The largest qualifying foreground blob (>2.5× RBC area) is selected as the neutrophil.

### Microbe detection
Image inversion highlights dark objects. After suppressing known-cell regions (dilated RBC + neutrophil masks), small blobs in the microbe size range are detected and dilated for visibility.

## Notes

- Pure OpenCV pipeline — no GPU required, runs at ~18fps on CPU
- Hough circle parameters (`param2=13`) are tuned for this specific dataset; adjust for different imaging conditions
- The microbe detector may produce false positives on debris; increasing the exclusion zone dilation or raising the threshold reduces these
