"""
Cell Slicer — Segmentation Pipeline
====================================
Identifies and marks boundaries of 3 cell types in microscopy video:
  - Neutrophil  → large, amoeboid, most prominent moving object (GREEN overlay)
  - Microbe     → tiny, very dark (YELLOW overlay)
  - Red Blood Cells → many spherical cells with central pallor (RED overlay)

Method:
  - RBCs: Hough Circle Transform (consistent round shape, central pallor)
  - Neutrophil: Background model + largest-foreground-blob tracking
  - Microbe: Dark-blob detection, size-filtered, excluding RBC/neutrophil regions
  - Stage jump: Frame-difference spike detection

Output: MP4 with semi-transparent colored masks + outlines per frame.

Usage:
    python3 pipeline.py [--input PATH] [--output PATH] [options]
"""

import argparse
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from collections import deque


# ─── Colour constants (RGB, since we render in RGB space) ──────────────────────
COLOUR_NEUTROPHIL_RGB = (40,  220, 40)    # green
COLOUR_MICROBE_RGB    = (220, 220, 0)     # yellow
COLOUR_RBC_RGB        = (220, 60,  60)    # red
ALPHA_FILL            = 0.38             # fill transparency


# ─── Stage jump detection ───────────────────────────────────────────────────────

def detect_stage_jump(prev_gray, curr_gray, threshold=18.0):
    """Return True if mean abs frame difference exceeds threshold."""
    if prev_gray is None:
        return False
    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
    return float(diff.mean()) > threshold


# ─── Pre-processing ─────────────────────────────────────────────────────────────

def preprocess(frame_rgb):
    """Return (gray, enhanced_gray) from an RGB frame."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return gray, enhanced


# ─── RBC detection ──────────────────────────────────────────────────────────────

def detect_rbcs(enhanced, diameter_rbc):
    """
    Detect RBCs using Hough Circle Transform.
    Returns a binary mask with filled circles and the raw circles array.

    RBCs appear as round cells ~35-40px diameter with central pallor.
    They are extremely numerous and don't move much.
    """
    h, w = enhanced.shape
    r = diameter_rbc / 2.0

    # Blur to smooth texture before edge detection
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(diameter_rbc * 0.85),   # prevent duplicate detections on same cell
        param1=50,                           # Canny edge upper threshold
        param2=13,                           # accumulator threshold (lower = more sensitive)
        minRadius=int(r * 0.65),
        maxRadius=int(r * 1.35),
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        circles_int = np.round(circles[0]).astype(int)
        for (cx, cy, cr) in circles_int:
            cv2.circle(mask, (cx, cy), cr, 255, -1)

    return mask, circles


# ─── Neutrophil detection ────────────────────────────────────────────────────────

class BackgroundModel:
    """
    Rolling median background model for foreground extraction.
    """
    def __init__(self, history=30):
        self.history = history
        self.frames = deque(maxlen=history)
        self.bg = None

    def update(self, gray):
        self.frames.append(gray.astype(np.float32))
        if len(self.frames) >= 5:
            self.bg = np.median(np.stack(self.frames, axis=0), axis=0).astype(np.uint8)

    def foreground(self, gray, threshold=15):
        if self.bg is None:
            return np.zeros_like(gray)
        diff = cv2.absdiff(gray, self.bg)
        _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return fg


def detect_neutrophil(enhanced, gray, bg_model, diameter_rbc):
    """
    Detect the neutrophil as the largest foreground blob.
    The neutrophil is amoeboid, much larger than RBCs, and actively moving.
    Returns a binary mask.
    """
    # Foreground from background subtraction
    fg = bg_model.foreground(gray, threshold=15)

    # Adaptive threshold to catch the neutrophil's distinct texture
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=4
    )

    # Combine both signals
    combined = cv2.bitwise_or(fg, adaptive)

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel_open)

    # Find contours; neutrophil must be > 2.5× the area of a single RBC
    rbc_area_ref    = np.pi * (diameter_rbc / 2) ** 2
    neutro_min_area = rbc_area_ref * 2.5

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the single largest qualifying contour
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= neutro_min_area and area > best_area:
            best = cnt
            best_area = area

    mask = np.zeros_like(enhanced, dtype=np.uint8)
    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, -1)

    return mask


# ─── Microbe detection ───────────────────────────────────────────────────────────

def detect_microbe(enhanced, neutrophil_mask, rbc_mask, diameter_rbc):
    """
    Detect the microbe as a very small, very dark blob.
    Excludes regions already claimed by neutrophil or RBCs.
    The microbe is much smaller than an RBC (~2-4px).
    """
    # Exclusion zone: dilated union of known cells
    combined_cells = np.clip(
        neutrophil_mask.astype(np.int32) + rbc_mask.astype(np.int32), 0, 255
    ).astype(np.uint8)
    exclusion = cv2.dilate(combined_cells, np.ones((9, 9), np.uint8), iterations=2)

    # Invert (dark → bright) and suppress known-cell regions
    inv = cv2.bitwise_not(enhanced)
    inv[exclusion > 0] = 0

    # Smooth and threshold
    blurred = cv2.GaussianBlur(inv, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 185, 255, cv2.THRESH_BINARY)

    # Remove single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Size filter: microbe must be smaller than 0.35× RBC radius in diameter
    max_microbe_area = np.pi * (diameter_rbc * 0.35) ** 2
    min_microbe_area = 1.5

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    microbe_mask = np.zeros_like(enhanced, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_microbe_area <= area <= max_microbe_area:
            cv2.drawContours(microbe_mask, [cnt], -1, 255, -1)

    # Dilate slightly for visibility in output
    microbe_mask = cv2.dilate(microbe_mask, np.ones((3, 3), np.uint8), iterations=1)

    return microbe_mask


# ─── Overlay rendering ──────────────────────────────────────────────────────────

def apply_colour_fill(base_rgb, mask, colour_rgb, alpha):
    """Blend a single-colour mask onto base_rgb with given alpha. In-place."""
    for c, val in enumerate(colour_rgb):
        channel = base_rgb[:, :, c].astype(np.float32)
        base_rgb[:, :, c] = np.where(
            mask > 0,
            np.clip(channel * (1 - alpha) + val * alpha, 0, 255),
            channel
        ).astype(np.uint8)


def draw_overlay(frame_rgb, neutrophil_mask, rbc_mask, microbe_mask):
    """Blend semi-transparent coloured fills + opaque outlines onto the frame."""
    out = frame_rgb.copy().astype(np.float32)
    result = frame_rgb.copy()

    # Fill layers (applied in order: RBC first, then neutrophil on top, then microbe)
    apply_colour_fill(result, rbc_mask,        COLOUR_RBC_RGB,        ALPHA_FILL)
    apply_colour_fill(result, neutrophil_mask, COLOUR_NEUTROPHIL_RGB, ALPHA_FILL)
    apply_colour_fill(result, microbe_mask,    COLOUR_MICROBE_RGB,    ALPHA_FILL)

    # Outlines (opaque, drawn on top)
    def draw_outlines(image, mask, colour_rgb, thickness=1):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colour_rgb, thickness)

    draw_outlines(result, rbc_mask,        COLOUR_RBC_RGB,        1)
    draw_outlines(result, neutrophil_mask, COLOUR_NEUTROPHIL_RGB, 2)
    draw_outlines(result, microbe_mask,    COLOUR_MICROBE_RGB,    1)

    return result


def draw_legend(frame, stage_jump=False):
    """Draw a small colour legend in the top-left corner."""
    items = [
        ("Neutrophil", COLOUR_NEUTROPHIL_RGB),
        ("Microbe",    COLOUR_MICROBE_RGB),
        ("RBC",        COLOUR_RBC_RGB),
    ]
    h, w = frame.shape[:2]
    x, y = 5, 12
    for label, colour_rgb in items:
        cv2.circle(frame, (x + 4, y - 3), 4, colour_rgb, -1)
        cv2.putText(frame, label, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1, cv2.LINE_AA)
        y += 14

    if stage_jump:
        cv2.putText(frame, "STAGE MOVE", (w // 2 - 42, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    return frame


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cell Slicer segmentation pipeline")
    parser.add_argument("--input",  default="source-movie/chase-original.mp4")
    parser.add_argument("--output", default="output/chase-segmented.mp4")
    parser.add_argument("--diameter-rbc", type=float, default=37.0,
                        help="Expected RBC diameter in pixels (default: 37)")
    parser.add_argument("--bg-history", type=int, default=25,
                        help="Background model history window in frames (default: 25)")
    parser.add_argument("--stage-jump-threshold", type=float, default=18.0,
                        help="Mean pixel diff to flag stage movement (default: 18)")
    parser.add_argument("--test", type=int, default=0,
                        help="If >0, only process this many frames (for testing)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading video: {input_path}")
    reader = iio.imopen(str(input_path), "r", plugin="pyav")
    meta = reader.metadata()
    fps = meta.get("fps", 15.0)
    reader.close()

    frames = iio.imread(str(input_path), plugin="pyav", index=None)
    n_frames = frames.shape[0]
    if args.test > 0:
        frames = frames[:args.test]
        n_frames = args.test
        print(f"  [TEST MODE] Processing {n_frames} frames only")

    print(f"  {n_frames} frames @ {fps:.2f} fps, {frames.shape[2]}×{frames.shape[1]} px")
    print(f"  RBC diameter: {args.diameter_rbc:.0f}px | BG history: {args.bg_history}f")

    bg_model = BackgroundModel(history=args.bg_history)
    out_frames = []
    prev_gray = None

    print("Processing frames…")
    for i in tqdm(range(n_frames), unit="frame"):
        frame = frames[i]  # HxWx3 RGB
        gray, enhanced = preprocess(frame)

        stage_jump = detect_stage_jump(prev_gray, gray, args.stage_jump_threshold)
        bg_model.update(gray)
        prev_gray = gray

        if stage_jump:
            out = frame.copy()
            draw_legend(out, stage_jump=True)
            out_frames.append(out)
            continue

        # Detect cells
        rbc_mask, _     = detect_rbcs(enhanced, args.diameter_rbc)
        neutrophil_mask = detect_neutrophil(enhanced, gray, bg_model, args.diameter_rbc)
        microbe_mask    = detect_microbe(enhanced, neutrophil_mask, rbc_mask, args.diameter_rbc)

        out = draw_overlay(frame, neutrophil_mask, rbc_mask, microbe_mask)
        draw_legend(out)
        out_frames.append(out)

    print(f"\nWriting output: {output_path}")
    out_array = np.stack(out_frames, axis=0)
    iio.imwrite(
        str(output_path),
        out_array,
        plugin="pyav",
        codec="h264",
        fps=int(round(fps)),
        out_pixel_format="yuv420p",
    )
    print(f"Done! → {output_path}  ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
