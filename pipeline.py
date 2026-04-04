"""
Cell Slicer — Segmentation Pipeline v2
========================================
Identifies and marks boundaries of 2 cell types in microscopy video:
  - Red Blood Cells (RBCs)  → RED overlay
  - Neutrophil              → GREEN overlay

Strategy:
  1. Threshold dark regions → fill holes → get all cell bodies
  2. Use Hough circles to place one seed per RBC centre
  3. Marker-controlled watershed: expand seeds into cell bodies
     (this prevents the RBC central pallor from creating spurious internal regions)
  4. Any large unseeded region in the cell body mask = neutrophil

Stage jump detection: skip segmentation on blurry/repositioned frames.

Usage:
    python3 pipeline.py [--input PATH] [--output PATH] [options]
"""

import argparse
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage


# ─── Colour constants (RGB) ────────────────────────────────────────────────────
COLOUR_RBC_RGB        = (220, 60,  60)   # red
COLOUR_NEUTROPHIL_RGB = (40,  220, 40)   # green
ALPHA_FILL            = 0.40


# ─── Stage jump ────────────────────────────────────────────────────────────────

def detect_stage_jump(prev_gray, curr_gray, threshold=18.0):
    if prev_gray is None:
        return False
    return float(cv2.absdiff(prev_gray, curr_gray).astype(np.float32).mean()) > threshold


# ─── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return gray, enhanced


# ─── Cell body mask ────────────────────────────────────────────────────────────

def get_cell_bodies(enhanced, dark_thresh=100):
    """
    Binary mask of all cell regions (RBCs + neutrophil).
    Uses dark-border thresholding + hole fill.
    """
    _, dark = cv2.threshold(enhanced, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Close gaps in cell borders
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3, iterations=2)
    # Fill interior holes (RBC pallor etc.)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    # Remove sub-RBC noise
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)
    return cleaned


# ─── RBC seed finding (Hough) ──────────────────────────────────────────────────

def find_rbc_seeds(enhanced, rbc_radius):
    """
    Use Hough Circle Transform to locate RBC centres.
    Returns array of (cx, cy, cr) or None.
    """
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(rbc_radius * 1.4),
        param1=50,
        param2=13,
        minRadius=int(rbc_radius * 0.6),
        maxRadius=int(rbc_radius * 1.2),
    )
    if circles is not None:
        return np.round(circles[0]).astype(int)
    return None


# ─── Marker-controlled watershed ───────────────────────────────────────────────

def watershed_segment(gray, cell_bodies, rbc_seeds, rbc_radius):
    """
    Segment cell bodies using marker-controlled watershed.
    Seeds are RBC circle centres from Hough.
    Returns a label image (0=background, 1=uncertain, 2+=cells).
    """
    h, w = gray.shape
    markers = np.zeros((h, w), dtype=np.int32)

    # Plant one seed per Hough circle
    if rbc_seeds is not None:
        for idx, (cx, cy, cr) in enumerate(rbc_seeds):
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(markers, (cx, cy), max(2, cr // 3), idx + 2, -1)

    # Sure background: pixels clearly outside all cell bodies
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(cell_bodies, k3, iterations=3)
    markers[sure_bg == 0] = 1  # label 1 = background

    # Run watershed on the grayscale image (BGR input required)
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_bgr, markers)

    return markers


# ─── Classify watershed regions ────────────────────────────────────────────────

def classify_regions(markers, cell_bodies, rbc_radius):
    """
    Split watershed labels into RBC mask and neutrophil mask.

    - Seeded regions (label >= 2 from Hough) in RBC size range → RBC
    - Large unseeded contiguous regions in cell_bodies not covered by watershed → neutrophil
    """
    h, w = markers.shape
    rbc_min_area  = np.pi * (rbc_radius * 0.55) ** 2
    rbc_max_area  = np.pi * (rbc_radius * 1.5) ** 2
    neutro_min_area = rbc_max_area * 1.5

    rbc_mask    = np.zeros((h, w), dtype=np.uint8)
    neutro_mask = np.zeros((h, w), dtype=np.uint8)

    # Label seeded regions by size
    for label in range(2, int(markers.max()) + 1):
        region = (markers == label).astype(np.uint8)
        area = int(region.sum())
        if rbc_min_area <= area <= rbc_max_area:
            rbc_mask |= region * 255

    # Neutrophil: large connected components in cell_bodies NOT covered by rbc_mask
    # (the Hough missed it because it's not round)
    unseeded = cv2.bitwise_and(cell_bodies, cv2.bitwise_not(
        cv2.dilate(rbc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ))
    n_comp, comp_labels = cv2.connectedComponentsWithAlgorithm(
        unseeded, 8, cv2.CV_32S, cv2.CCL_DEFAULT
    )
    for lbl in range(1, n_comp):
        region = (comp_labels == lbl).astype(np.uint8)
        area = int(region.sum())
        if area >= neutro_min_area:
            neutro_mask |= region * 255

    # Clean up masks
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rbc_mask    = cv2.morphologyEx(rbc_mask,    cv2.MORPH_CLOSE, k3)
    neutro_mask = cv2.morphologyEx(neutro_mask, cv2.MORPH_CLOSE, k3)

    return rbc_mask, neutro_mask


# ─── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame_rgb, rbc_mask, neutro_mask):
    result = frame_rgb.copy().astype(np.float32)

    def fill(img, mask, colour):
        for c, val in enumerate(colour):
            img[:, :, c] = np.where(mask > 0, img[:, :, c] * (1 - ALPHA_FILL) + val * ALPHA_FILL, img[:, :, c])

    fill(result, rbc_mask,    COLOUR_RBC_RGB)
    fill(result, neutro_mask, COLOUR_NEUTROPHIL_RGB)
    result = np.clip(result, 0, 255).astype(np.uint8)

    def outlines(img, mask, colour, thickness):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colour, thickness)

    outlines(result, rbc_mask,    COLOUR_RBC_RGB,        1)
    outlines(result, neutro_mask, COLOUR_NEUTROPHIL_RGB, 2)

    return result


def draw_legend(frame, stage_jump=False):
    items = [
        ("Neutrophil", COLOUR_NEUTROPHIL_RGB),
        ("RBC",        COLOUR_RBC_RGB),
    ]
    h, w = frame.shape[:2]
    x, y = 5, 12
    for label, colour in items:
        cv2.circle(frame, (x + 4, y - 3), 4, colour, -1)
        cv2.putText(frame, label, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1, cv2.LINE_AA)
        y += 14
    if stage_jump:
        cv2.putText(frame, "STAGE MOVE", (w // 2 - 42, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    return frame


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="source-movie/chase-original.mp4")
    parser.add_argument("--output", default="output/chase-segmented.mp4")
    parser.add_argument("--rbc-radius", type=float, default=17.0,
                        help="Expected RBC radius in pixels (default: 17 → ~34px diameter)")
    parser.add_argument("--dark-thresh", type=int, default=100,
                        help="Intensity threshold for dark cell borders (default: 100)")
    parser.add_argument("--stage-jump-threshold", type=float, default=18.0)
    parser.add_argument("--test", type=int, default=0,
                        help="Process only first N frames")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    reader = iio.imopen(str(input_path), "r", plugin="pyav")
    fps = reader.metadata().get("fps", 15.0)
    reader.close()

    frames = iio.imread(str(input_path), plugin="pyav", index=None)
    n_frames = frames.shape[0]
    if args.test > 0:
        frames, n_frames = frames[:args.test], args.test
        print(f"  [TEST] {n_frames} frames")
    print(f"  {n_frames} frames @ {fps:.1f}fps, {frames.shape[2]}×{frames.shape[1]}px")
    print(f"  RBC radius: {args.rbc_radius}px  dark_thresh: {args.dark_thresh}")

    out_frames = []
    prev_gray  = None

    for i in tqdm(range(n_frames), unit="frame"):
        frame = frames[i]
        gray, enhanced = preprocess(frame)

        stage_jump = detect_stage_jump(prev_gray, gray, args.stage_jump_threshold)
        prev_gray  = gray

        if stage_jump:
            out = frame.copy()
            draw_legend(out, stage_jump=True)
            out_frames.append(out)
            continue

        cell_bodies = get_cell_bodies(enhanced, args.dark_thresh)
        rbc_seeds   = find_rbc_seeds(enhanced, args.rbc_radius)
        markers     = watershed_segment(gray, cell_bodies, rbc_seeds, args.rbc_radius)
        rbc_mask, neutro_mask = classify_regions(markers, cell_bodies, args.rbc_radius)

        out = draw_overlay(frame, rbc_mask, neutro_mask)
        draw_legend(out)
        out_frames.append(out)

    print(f"\nWriting: {output_path}")
    iio.imwrite(str(output_path), np.stack(out_frames), plugin="pyav",
                codec="h264", fps=int(round(fps)), out_pixel_format="yuv420p")
    print(f"Done → {output_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
