# Cell Slicer 🔬

AI-assisted image analysis pipeline for segmenting and tracking cells in brightfield/phase-contrast microscopy video.

## What it does

Identifies and marks boundaries of 3 cell types across all frames of a microscopy video:

| Cell type | Colour | Detection method |
|-----------|--------|-----------------|
| **Neutrophil** | 🟢 Green | Background subtraction + adaptive threshold → largest foreground blob |
| **Microbe** | 🟡 Yellow | Dark-blob detection on inverted image, size-filtered |
| **Red Blood Cells** | 🔴 Red | Hough Circle Transform (tuned for RBC diameter and central pallor) |

Output is an MP4 with semi-transparent coloured masks and outlines overlaid on the original footage.

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
--diameter-rbc FLOAT      Expected RBC diameter in pixels (default: 37)
--bg-history INT          Background model history window in frames (default: 25)
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
