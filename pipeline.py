"""
Cell Slicer — Segmentation Pipeline v7
========================================
Correction-driven segmentation with temporal smoothing.

Key design:
- Correction points are seeds for actual image-based segmentation (not blunt discs)
- RBCs: correction point → flood-fill to cell contour via watershed
- Neutrophil: GrabCut seeded by all interpolated correction points
- Temporal smoothing: large frame-to-frame mask changes are suppressed
  (physical cells don't disappear for one frame then reappear)
- Stage jumps partition the video; corrections never propagate across jumps

Usage:
    python3 pipeline.py [--build-density-map] [--corrections PATH] [options]
"""

import argparse, json
import numpy as np
import cv2
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from collections import deque
from scipy import ndimage


# ─── Constants ─────────────────────────────────────────────────────────────────
COLOUR_RBC_RGB        = (220, 60,  60)
COLOUR_NEUTROPHIL_RGB = (40,  220, 40)
ALPHA_FILL            = 0.40
GRABCUT_ITER          = 5
STAGE_JUMP_THRESH     = 18.0
MIN_NEUTROPHIL_PX     = 300

# Temporal smoothing: if area ratio > FLICKER_RATIO vs rolling median, blend
FLICKER_RATIO         = 2.0
TEMPORAL_WINDOW       = 5   # frames each side for smoothing look-ahead/look-back


# ─── Preprocessing ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return gray, _clahe.apply(gray)


# ─── Stage jump detection ──────────────────────────────────────────────────────

def find_stage_jumps(frames, threshold=STAGE_JUMP_THRESH):
    """
    Detect stage jump frames. Only the FIRST frame of each consecutive run
    of high-diff frames is marked as a jump. Subsequent frames in a run
    have high diff only because the preceding frame was displaced — they are
    themselves normal (settled) frames and should not be skipped.
    """
    prev = None
    raw_jumps = []
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            if float(cv2.absdiff(gray, prev).astype(np.float32).mean()) > threshold:
                raw_jumps.append(i)
        prev = gray

    # Keep only the first frame of each consecutive cluster
    jumps = set()
    prev_j = -2
    for j in raw_jumps:
        if j != prev_j + 1:
            jumps.add(j)
        prev_j = j
    return jumps


def segment_for_frame(frame_idx, jump_frames, n_frames):
    """(seg_start, seg_end) for frame_idx, not crossing any jump frame."""
    jumps = sorted(jump_frames)
    start = 0
    for j in jumps:
        if j > frame_idx:
            return start, j - 1
        start = j + 1
    return start, n_frames - 1


# ─── Corrections ───────────────────────────────────────────────────────────────

def load_corrections(path):
    with open(path) as f:
        data = json.load(f)
    valid = [m for m in data
             if {'frame','x','y','label'} <= set(m.keys())
             and m['label'] in ('neutrophil','rbc','background')]
    counts = {l: sum(1 for m in valid if m['label']==l) for l in ('neutrophil','rbc','background')}
    print(f"  Loaded {len(valid)} corrections: {counts}")
    return valid


def corrections_for_frame(corrections, frame_idx):
    return [c for c in corrections if c['frame'] == frame_idx]


def annotated_frames_in_segment(corrections, seg_start, seg_end, label=None):
    frames = set()
    for c in corrections:
        if seg_start <= c['frame'] <= seg_end:
            if label is None or c['label'] == label:
                frames.add(c['frame'])
    return sorted(frames)


# ─── Interpolation helpers ─────────────────────────────────────────────────────

def centroid_of(corrections, frame_idx, label):
    pts = [(c['x'], c['y']) for c in corrections
           if c['frame'] == frame_idx and c['label'] == label]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def interpolate_points(corrections, frame_idx, label, seg_start, seg_end):
    """
    Return correction points for `label` at `frame_idx`, either directly
    (if this frame is annotated) or shifted from the nearest annotated frame
    by the inter-frame centroid displacement.
    Never crosses segment boundaries (stage jumps).
    """
    ann = annotated_frames_in_segment(corrections, seg_start, seg_end, label=label)
    if not ann:
        return []

    # Direct match
    direct = [(c['x'], c['y']) for c in corrections
              if c['frame'] == frame_idx and c['label'] == label]
    if direct:
        return direct

    # Find bracketing annotated frames
    before = [f for f in ann if f < frame_idx]
    after  = [f for f in ann if f > frame_idx]

    if before and after:
        f0, f1 = max(before), min(after)
        t = (frame_idx - f0) / (f1 - f0)
    elif before:
        f0, f1, t = max(before), max(before), 1.0
    else:
        f0, f1, t = min(after), min(after), 0.0

    # Use closer frame as reference, shift by interpolated centroid delta
    ref_frame = f0 if abs(frame_idx - f0) <= abs(frame_idx - f1) else f1
    c0 = centroid_of(corrections, f0, label)
    c1 = centroid_of(corrections, f1, label) if f0 != f1 else c0
    if c0 is None or c1 is None:
        return []

    interp_cx = c0[0] + t * (c1[0] - c0[0])
    interp_cy = c0[1] + t * (c1[1] - c0[1])
    ref_c = c0 if ref_frame == f0 else c1
    dx, dy = interp_cx - ref_c[0], interp_cy - ref_c[1]

    ref_pts = [(c['x'], c['y']) for c in corrections
               if c['frame'] == ref_frame and c['label'] == label]
    return [(x + dx, y + dy) for x, y in ref_pts]


# ─── Cell body mask ────────────────────────────────────────────────────────────

def get_cell_bodies(enhanced, dark_thresh=100):
    """Binary mask of all cell material (not background)."""
    _, dark = cv2.threshold(enhanced, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3, iterations=2)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, k5)


# ─── RBC segmentation ──────────────────────────────────────────────────────────

def build_density_map(frames, rbc_radius, density_path):
    h, w   = frames.shape[1], frames.shape[2]
    counts = np.zeros((h, w), np.int32)
    print("Building RBC density map…")
    for frame in tqdm(frames, unit="frame"):
        _, enh = preprocess(frame)
        circles = _find_hough(enh, rbc_radius)
        layer = np.zeros((h, w), np.int32)
        if circles is not None:
            for (cx, cy, cr) in circles:
                if 0 <= cy < h and 0 <= cx < w:
                    cv2.circle(layer, (cx, cy), 10, 1, -1)
        counts += layer
    density = counts.astype(np.float32) / len(frames)
    np.save(str(density_path), density)
    print(f"  Saved → {density_path}  (max={density.max():.3f})")
    return density


def _find_hough(enhanced, rbc_radius):
    bl = cv2.GaussianBlur(enhanced, (5,5), 1.5)
    c  = cv2.HoughCircles(bl, cv2.HOUGH_GRADIENT, dp=1,
                          minDist=int(rbc_radius*1.4), param1=50, param2=13,
                          minRadius=int(rbc_radius*0.6), maxRadius=int(rbc_radius*1.2))
    return np.round(c[0]).astype(int) if c is not None else None


def rbc_component_at(cell_bodies, x, y, h, w, rbc_radius):
    """
    Return the cell-body connected component touching point (x,y).
    Uses this as the actual RBC shape rather than a blunt disc.
    """
    xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))

    # If point is not on a cell body, search nearby
    if cell_bodies[yi, xi] == 0:
        found = False
        for r in [4, 8, 14, 20]:
            for dy in range(-r, r+1, 4):
                for dx in range(-r, r+1, 4):
                    ny, nx = yi+dy, xi+dx
                    if 0 <= ny < h and 0 <= nx < w and cell_bodies[ny, nx] > 0:
                        xi, yi = nx, ny
                        found = True
                        break
                if found: break
            if found: break
        if not found:
            return None

    # Get connected component at this point
    # Use flood fill from seed point within cell_bodies
    mask = np.zeros((h+2, w+2), np.uint8)
    img  = cell_bodies.copy()
    cv2.floodFill(img, mask, (xi, yi), 128)
    comp = (img == 128).astype(np.uint8) * 255

    # Size filter: must be RBC-sized
    area = comp.sum() // 255
    rbc_min = np.pi * (rbc_radius * 0.40) ** 2
    rbc_max = np.pi * (rbc_radius * 1.70) ** 2
    if area < rbc_min or area > rbc_max:
        return None
    return comp


def segment_rbcs(enhanced, cell_bodies, corrections, frame_idx, seg_start, seg_end,
                 density, hough_circles, rbc_radius, h, w):
    """
    RBC mask: watershed separates cells, correction points select which regions.
    This gives contour-accurate boundaries, not blunt discs.
    Background corrections erase the region they fall in.
    """
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # Watershed to separate touching cells into labelled regions
    seed_mask = (density >= 0.20).astype(np.uint8) * 255
    if hough_circles is not None:
        for (cx, cy, cr) in hough_circles:
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(seed_mask, (cx, cy), max(2, int(rbc_radius*0.22)), 255, -1)

    n_s, s_labels = cv2.connectedComponents(seed_mask)
    markers = np.zeros((h, w), np.int32)
    for lbl in range(1, n_s):
        markers[s_labels == lbl] = lbl + 1
    sure_bg = cv2.dilate(cell_bodies, k3, iterations=3)
    markers[sure_bg == 0] = 1
    cv2.watershed(np.stack([enhanced]*3, axis=-1), markers)

    rbc_min = np.pi * (rbc_radius * 0.50) ** 2
    rbc_max = np.pi * (rbc_radius * 1.80) ** 2

    # Build set of watershed labels that are RBC-sized
    rbc_labels = set()
    for lbl in range(2, int(markers.max()) + 1):
        area = int((markers == lbl).sum())
        if rbc_min <= area <= rbc_max:
            rbc_labels.add(lbl)

    # Pre-screen: remove watershed labels whose centroid falls near a background
    # correction point. This prevents the density/Hough-seeded initial mask from
    # including regions Clem explicitly marked as non-cell.
    # Use ALL background corrections from the segment (not interpolated) —
    # background points are static spatial vetoes, not moving cell positions.
    BG_SEED_EXCL_R = 22  # px
    bg_pts_pre = [(c['x'], c['y']) for c in corrections
                  if c['label'] == 'background'
                  and seg_start <= c['frame'] <= seg_end]
    if bg_pts_pre:
        to_remove = set()
        for lbl in rbc_labels:
            region = (markers == lbl)
            ys, xs = np.where(region)
            if len(xs) == 0: continue
            cxr, cyr = int(xs.mean()), int(ys.mean())
            if any((cxr - bx)**2 + (cyr - by)**2 <= BG_SEED_EXCL_R**2
                   for bx, by in bg_pts_pre):
                to_remove.add(lbl)
        rbc_labels -= to_remove

    # Start with all Hough/density-seeded RBC regions
    rbc_mask = np.zeros((h, w), np.uint8)
    for lbl in rbc_labels:
        rbc_mask |= (markers == lbl).astype(np.uint8) * 255

    # For each RBC correction point, ensure its watershed region is included
    rbc_pts = interpolate_points(corrections, frame_idx, 'rbc', seg_start, seg_end)
    for x, y in rbc_pts:
        xi = int(np.clip(x, 0, w-1))
        yi = int(np.clip(y, 0, h-1))

        # Search for an RBC-sized watershed region near this point
        found_lbl = None
        for r in [0, 3, 6, 10, 15]:
            step = max(1, r)
            for dy in range(-r, r+1, step):
                for dx in range(-r, r+1, step):
                    ny, nx = yi+dy, xi+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        lbl = int(markers[ny, nx])
                        if lbl in rbc_labels:
                            found_lbl = lbl
                            break
                        elif lbl >= 2:
                            # Check if this region is in size range
                            area = int((markers == lbl).sum())
                            if rbc_min <= area <= rbc_max:
                                rbc_labels.add(lbl)
                                found_lbl = lbl
                                break
                if found_lbl: break
            if found_lbl: break

        if found_lbl:
            rbc_mask |= (markers == found_lbl).astype(np.uint8) * 255
        else:
            # Accept any watershed region at/near the correction point, even small ones
            # (the annotator clicked on a real cell, so trust them over size filter)
            for r2 in [0, 3, 6, 10, 15, 20]:
                step2 = max(1, r2)
                inner = False
                for dy2 in range(-r2, r2+1, step2):
                    for dx2 in range(-r2, r2+1, step2):
                        ny2, nx2 = yi+dy2, xi+dx2
                        if 0<=ny2<h and 0<=nx2<w:
                            lbl2 = int(markers[ny2, nx2])
                            if lbl2 >= 2:
                                region2 = (markers == lbl2).astype(np.uint8)
                                if region2.sum() >= 40:
                                    rbc_mask |= region2 * 255
                                    inner = True; break
                    if inner: break
                if inner: break
            else:
                # Try a Hough circle centred near this point
                best_c = None
                if hough_circles is not None:
                    for (hcx, hcy, hcr) in hough_circles:
                        if abs(hcx-xi)<=rbc_radius*1.5 and abs(hcy-yi)<=rbc_radius*1.5:
                            best_c = (hcx, hcy, hcr)
                            break
                if best_c:
                    cv2.circle(rbc_mask, (best_c[0], best_c[1]), best_c[2], 255, -1)
                else:
                    cv2.circle(rbc_mask, (xi, yi), int(rbc_radius * 0.85), 255, -1)

    # Erase background corrections — remove ALL watershed regions whose centroid
    # falls within BG_ERASE_R px of the annotation point, plus the exact label.
    # A fixed radius handles frames where the watershed draws a different boundary
    # than in the annotated frame (the label at the point may differ).
    BG_ERASE_R = 22  # px — ~1.3x RBC radius; wide enough to catch boundary shifts
    # Use all background corrections in segment — static vetoes, not interpolated
    bg_pts = [(c['x'], c['y']) for c in corrections
              if c['label'] == 'background'
              and seg_start <= c['frame'] <= seg_end]
    for x, y in bg_pts:
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        # Always erase exact label at the point
        lbl = int(markers[yi, xi])
        if lbl >= 2:
            rbc_mask[markers == lbl] = 0
        # Also erase any RBC label whose centroid is within BG_ERASE_R
        for erase_lbl in list(rbc_labels):
            region = (markers == erase_lbl)
            ys, xs = np.where(region)
            if len(xs) == 0: continue
            cxr, cyr = int(xs.mean()), int(ys.mean())
            if (cxr - xi)**2 + (cyr - yi)**2 <= BG_ERASE_R**2:
                rbc_mask[region] = 0
                rbc_labels.discard(erase_lbl)
        # Fallback disc erase centred on annotation point
        cv2.circle(rbc_mask, (xi, yi), BG_ERASE_R, 0, -1)

    # Final guaranteed pass: any RBC correction point not yet covered by the
    # mask gets a disc. The annotator clicked on a real cell — trust them.
    for x, y in rbc_pts:
        xi = int(np.clip(x, 0, w-1))
        yi = int(np.clip(y, 0, h-1))
        if rbc_mask[yi, xi] == 0:
            # Try to find a Hough circle near this point for a better fit
            best = None
            if hough_circles is not None:
                dists = [((hcx-xi)**2+(hcy-yi)**2, hcx, hcy, hcr)
                         for hcx,hcy,hcr in hough_circles]
                dists.sort()
                if dists and dists[0][0] <= (rbc_radius*2)**2:
                    _, hx, hy, hr = dists[0]
                    best = (hx, hy, hr)
            if best:
                cv2.circle(rbc_mask, (best[0], best[1]), best[2], 255, -1)
            else:
                cv2.circle(rbc_mask, (xi, yi), int(rbc_radius * 0.85), 255, -1)

    return rbc_mask


# ─── Neutrophil segmentation ───────────────────────────────────────────────────

def grabcut_neutrophil(frame_bgr, enhanced, neutro_pts, bg_pts, rbc_pts,
                       seed_r=8, n_iter=GRABCUT_ITER):
    h, w = enhanced.shape
    if not neutro_pts:
        return np.zeros((h, w), np.uint8)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[enhanced > 190] = cv2.GC_BGD

    for x, y in neutro_pts:
        cv2.circle(gc, (int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))), seed_r, cv2.GC_FGD, -1)
    for x, y in bg_pts:
        cv2.circle(gc, (int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))), seed_r, cv2.GC_BGD, -1)
    for x, y in rbc_pts:
        cv2.circle(gc, (int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))), seed_r, cv2.GC_PR_BGD, -1)

    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(frame_bgr, gc, None, bgd, fgd, n_iter, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return np.zeros((h, w), np.uint8)

    raw = np.where((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Keep only the single largest component
    n_c, lbl, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n_c <= 1:
        return np.zeros((h, w), np.uint8)
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if stats[best, cv2.CC_STAT_AREA] < MIN_NEUTROPHIL_PX:
        return np.zeros((h, w), np.uint8)
    mask = ((lbl == best) * 255).astype(np.uint8)
    return ndimage.binary_fill_holes(mask).astype(np.uint8) * 255


def segment_neutrophil(frame_bgr, enhanced, corrections, frame_idx,
                       seg_start, seg_end, density, h, w):
    neutro_pts = interpolate_points(corrections, frame_idx, 'neutrophil', seg_start, seg_end)
    bg_pts     = interpolate_points(corrections, frame_idx, 'background', seg_start, seg_end)
    rbc_pts    = interpolate_points(corrections, frame_idx, 'rbc',        seg_start, seg_end)

    direct = any(c['frame']==frame_idx and c['label']=='neutrophil' for c in corrections)
    tier   = 1 if direct else (2 if neutro_pts else 3)

    if not neutro_pts:
        return np.zeros((h, w), np.uint8), 3

    # Add density RBC positions as background seeds
    ry, rx = np.where(density >= 0.20)
    if len(ry) > 30:
        idx = np.random.choice(len(ry), 30, replace=False)
        rbc_pts = rbc_pts + list(zip(rx[idx].tolist(), ry[idx].tolist()))

    mask = grabcut_neutrophil(frame_bgr, enhanced, neutro_pts, bg_pts, rbc_pts)

    # Directly paint all neutrophil correction points as definite foreground.
    # GrabCut boundary may not reach every annotation — painting ensures coverage.
    # Use a moderate radius (6px) to catch the local cell boundary, then close
    # to merge with the GrabCut region.
    seed_layer = np.zeros((h, w), np.uint8)
    for x, y in neutro_pts:
        cv2.circle(seed_layer, (int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))), 6, 255, -1)

    # Merge: union of GrabCut result and painted seeds
    combined = cv2.bitwise_or(mask, seed_layer)
    # Close to connect seeds to the main body (bridges small gaps)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    # Keep only the largest connected component (no leaked fragments)
    n_c, lbl, stats, _ = cv2.connectedComponentsWithStats(combined)
    if n_c > 1:
        best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        combined = ((lbl == best) * 255).astype(np.uint8)
    mask = ndimage.binary_fill_holes(combined > 0).astype(np.uint8) * 255

    return mask, tier


# ─── Temporal smoothing ────────────────────────────────────────────────────────

def smooth_masks_temporally(masks, jump_frames, n_frames, ratio_thresh=FLICKER_RATIO):
    """
    Suppress single-frame mask spikes using area-based outlier detection.
    
    If a frame's mask area is an outlier (>ratio_thresh times the rolling
    median of its neighbours within the same segment), replace it with the
    nearest non-outlier frame's mask. This handles:
    - Neutrophil suddenly expanding for 1 frame (frame 70 type error)
    - RBC regions blinking in/out for 1 frame
    
    Uses the actual mask from the best neighbouring frame (not pixel median)
    so the replacement is always a real, spatially coherent mask.
    """
    W = TEMPORAL_WINDOW

    def area(m):
        return int(m.sum()) // 255

    def seg_of(fi):
        return segment_for_frame(fi, jump_frames, n_frames)

    smoothed = [list(m) for m in masks]

    for mask_idx in range(2):  # 0=neutro, 1=rbc
        for fi in range(n_frames):
            if fi in jump_frames:
                continue
            seg_s, seg_e = seg_of(fi)
            cur_area = area(masks[fi][mask_idx])

            # Collect neighbour areas/frames in same segment
            neighbours = []
            for delta in range(-W, W+1):
                if delta == 0: continue
                nfi = fi + delta
                if nfi < 0 or nfi >= n_frames: continue
                if seg_of(nfi)[0] != seg_s: continue
                if nfi in jump_frames: continue
                na = area(masks[nfi][mask_idx])
                neighbours.append((abs(delta), nfi, na))

            if len(neighbours) < 3:
                continue

            areas = [na for _, _, na in neighbours]
            median_area = float(np.median(areas))
            if median_area < 50:
                continue

            # Check if current frame is an outlier
            ratio = (cur_area / median_area) if median_area > 0 else 1.0
            if ratio > ratio_thresh or ratio < 1.0 / ratio_thresh:
                # Replace with mask from the nearest non-outlier neighbour
                candidates = sorted(neighbours, key=lambda x: x[0])
                for _, nfi, na in candidates:
                    nratio = (na / median_area) if median_area > 0 else 1.0
                    if 1.0/ratio_thresh <= nratio <= ratio_thresh:
                        smoothed[fi][mask_idx] = masks[nfi][mask_idx].copy()
                        break

    return [tuple(m) for m in smoothed]


# ─── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame_rgb, rbc_mask, neutro_mask):
    out = frame_rgb.copy().astype(np.float32)
    def fill(img, mask, c):
        for ch, v in enumerate(c):
            img[:,:,ch] = np.where(mask>0, img[:,:,ch]*(1-ALPHA_FILL)+v*ALPHA_FILL, img[:,:,ch])
    fill(out, rbc_mask,    COLOUR_RBC_RGB)
    fill(out, neutro_mask, COLOUR_NEUTROPHIL_RGB)
    out = np.clip(out, 0, 255).astype(np.uint8)
    def draw_cnts(img, mask, c, t):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, c, t)
    draw_cnts(out, rbc_mask,    COLOUR_RBC_RGB,        1)
    draw_cnts(out, neutro_mask, COLOUR_NEUTROPHIL_RGB, 2)
    return out


def draw_legend(frame, tier=None, stage_jump=False):
    for i, (label, colour) in enumerate([
            ("Neutrophil", COLOUR_NEUTROPHIL_RGB), ("RBC", COLOUR_RBC_RGB)]):
        y = 12 + i*14
        cv2.circle(frame, (9, y-3), 4, colour, -1)
        cv2.putText(frame, label, (17, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    if stage_jump:
        cv2.putText(frame, "STAGE MOVING", (w//2-50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "STAGE MOVING", (w//2-50, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1, cv2.LINE_AA)
    if tier:
        col = {1:(0,230,120), 2:(0,180,230), 3:(180,180,100)}.get(tier, (128,128,128))
        cv2.putText(frame, f"T{tier}", (w-20, h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",                default="source-movie/chase-original.mp4")
    parser.add_argument("--output",               default="output/chase-segmented.mp4")
    parser.add_argument("--density-map",          default="output/rbc_density.npy")
    parser.add_argument("--build-density-map",    action="store_true")
    parser.add_argument("--corrections",          default=None)
    parser.add_argument("--rbc-radius",           type=float, default=17.0)
    parser.add_argument("--stage-jump-threshold", type=float, default=STAGE_JUMP_THRESH)
    parser.add_argument("--test",                 type=int,   default=0)
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    density_path = Path(args.density_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = iio.imopen(str(input_path), "r", plugin="pyav")
    fps    = reader.metadata().get("fps", 15.0)
    reader.close()
    frames  = iio.imread(str(input_path), plugin="pyav", index=None)
    n_total = len(frames)

    if args.build_density_map:
        build_density_map(frames, args.rbc_radius, density_path)
        return

    density = (np.load(str(density_path)) if density_path.exists()
               else build_density_map(frames, args.rbc_radius, density_path))

    corrections = []
    if args.corrections and Path(args.corrections).exists():
        corrections = load_corrections(args.corrections)
    elif args.corrections:
        print(f"  WARNING: {args.corrections} not found")

    n_frames = n_total
    if args.test > 0:
        frames, n_frames = frames[:args.test], args.test
        print(f"  [TEST] {n_frames} frames")

    h, w = frames.shape[1], frames.shape[2]
    print(f"  {n_frames} frames @ {fps:.1f}fps  |  RBC radius={args.rbc_radius}px")

    print("  Detecting stage jumps…")
    jump_frames = find_stage_jumps(frames, args.stage_jump_threshold)
    print(f"  {len(jump_frames)} jump(s): {sorted(jump_frames)}")

    # ── Pass 1: compute raw per-frame masks ──────────────────────────────────
    print("  Pass 1: computing masks…")
    raw_masks  = []  # list of (neutro_mask, rbc_mask)
    tiers      = []
    frame_bgrs = []

    for i in tqdm(range(n_frames), unit="frame"):
        frame     = frames[i]
        gray, enh = preprocess(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgrs.append(frame_bgr)

        if i in jump_frames:
            raw_masks.append((np.zeros((h,w),np.uint8), np.zeros((h,w),np.uint8)))
            tiers.append(0)
            continue

        hough = _find_hough(enh, args.rbc_radius)
        seg_s, seg_e = segment_for_frame(i, jump_frames, n_frames)
        cell_bodies  = get_cell_bodies(enh)

        neutro_mask, tier = segment_neutrophil(
            frame_bgr, enh, corrections, i, seg_s, seg_e, density, h, w)

        rbc_mask = segment_rbcs(
            enh, cell_bodies, corrections, i, seg_s, seg_e,
            density, hough, args.rbc_radius, h, w)

        # Neutrophil wins over RBC
        rbc_mask = cv2.bitwise_and(rbc_mask, cv2.bitwise_not(neutro_mask))

        raw_masks.append((neutro_mask, rbc_mask))
        tiers.append(tier)

    # ── Pass 2: temporal smoothing ───────────────────────────────────────────
    print("  Pass 2: temporal smoothing…")
    smooth_masks = smooth_masks_temporally(raw_masks, jump_frames, n_frames)

    # ── Pass 2b: small-fragment flicker suppression ──────────────────────────
    # Kill small RBC components that don't persist bilaterally across time.
    #
    # Only targets fragments below FLICKER_MAX_AREA (well below true RBC size).
    # Real RBCs are ~900px (π·17²). Watershed noise fragments are typically
    # <200px. We only apply the flicker check to these small fragments, leaving
    # all substantial components untouched.
    #
    # A fragment "persists" if any neighbour frame (within ±FLICKER_WINDOW)
    # has a component centroid within FLICKER_CENTROID_R px — on BOTH sides.
    #
    FLICKER_WINDOW    = 4    # frames each side
    FLICKER_MAX_AREA  = 200  # px — only check fragments smaller than this
    FLICKER_CENTROID_R = 25  # px — centroid match radius

    print("  Pass 2b: flicker suppression…")
    smooth_masks = list(smooth_masks)

    def seg_of(fi):
        return segment_for_frame(fi, jump_frames, n_frames)

    for i in range(n_frames):
        if i in jump_frames:
            continue
        neutro_mask, rbc_mask = smooth_masks[i]
        if rbc_mask is None or rbc_mask.max() == 0:
            continue

        seg_s, seg_e = seg_of(i)

        # Collect before/after neighbour frames
        before_frames, after_frames = [], []
        for delta in range(1, FLICKER_WINDOW + 1):
            for sign, bucket in [(-1, before_frames), (1, after_frames)]:
                ni = i + sign * delta
                if ni < 0 or ni >= n_frames: continue
                if ni in jump_frames: continue
                if seg_of(ni)[0] != seg_s: continue
                bucket.append(ni)

        if not before_frames or not after_frames:
            continue  # at segment edge

        # Pre-compute centroids for all neighbour frames (cheap)
        def get_centroids(frame_idx):
            nm = smooth_masks[frame_idx][1]
            nc, lb, st, _ = cv2.connectedComponentsWithStats(nm)
            return [(st[l, cv2.CC_STAT_LEFT] + st[l, cv2.CC_STAT_WIDTH]  // 2,
                     st[l, cv2.CC_STAT_TOP]  + st[l, cv2.CC_STAT_HEIGHT] // 2)
                    for l in range(1, nc)]

        before_centroids = [get_centroids(ni) for ni in before_frames]
        after_centroids  = [get_centroids(ni) for ni in after_frames]

        n_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(rbc_mask)
        rbc_mask_filtered = rbc_mask.copy()
        cr2 = FLICKER_CENTROID_R ** 2

        for l in range(1, n_c):
            area = int(stats_c[l, cv2.CC_STAT_AREA])
            if area >= FLICKER_MAX_AREA:
                continue  # leave substantial components alone

            cx = stats_c[l, cv2.CC_STAT_LEFT] + stats_c[l, cv2.CC_STAT_WIDTH]  // 2
            cy = stats_c[l, cv2.CC_STAT_TOP]  + stats_c[l, cv2.CC_STAT_HEIGHT] // 2

            hit_before = any(any((cx - ncx)**2 + (cy - ncy)**2 <= cr2 for ncx, ncy in cents)
                             for cents in before_centroids)
            hit_after  = any(any((cx - ncx)**2 + (cy - ncy)**2 <= cr2 for ncx, ncy in cents)
                             for cents in after_centroids)

            if not (hit_before and hit_after):
                rbc_mask_filtered[lbl_c == l] = 0

        smooth_masks[i] = (neutro_mask, rbc_mask_filtered)

    # ── Pass 3: render output video ──────────────────────────────────────────
    print("  Pass 3: rendering…")
    out_frames = []
    prev_neutro_mask = None
    prev_rbc_mask    = None
    for i in range(n_frames):
        frame = frames[i]
        is_jump = i in jump_frames
        if is_jump:
            prev_neutro_mask = None
            prev_rbc_mask    = None
            out = frame.copy()
            draw_legend(out, stage_jump=True)
        else:
            neutro_mask, rbc_mask = smooth_masks[i]

            # ── Cross-object boundary constraint ──────────────────────────
            # Objects don't suddenly jump to fill another object's space.
            # If neutrophil mask gained pixels that were RBC in prior frame, suppress.
            if prev_neutro_mask is not None and prev_rbc_mask is not None:
                new_neutro   = cv2.bitwise_and(neutro_mask, cv2.bitwise_not(prev_neutro_mask))
                invading_rbc = cv2.bitwise_and(new_neutro, prev_rbc_mask)
                invasion_px = int(invading_rbc.sum()) // 255
                new_neutro_px = int(new_neutro.sum()) // 255
                # Only suppress if invasion is large AND accounts for most of the new area
                # (i.e. truly jumping into RBC space, not just normal boundary expansion)
                if invasion_px > 400 and new_neutro_px > 0 and invasion_px / max(new_neutro_px, 1) > 0.5:
                    neutro_mask = cv2.bitwise_and(neutro_mask, cv2.bitwise_not(invading_rbc))

            prev_neutro_mask = neutro_mask.copy()
            prev_rbc_mask    = rbc_mask.copy()

            # Fill donut holes in RBC mask (applied after smoothing)
            rbc_mask = ndimage.binary_fill_holes(rbc_mask > 0).astype(np.uint8) * 255

            # Remove tiny RBC fragments below minimum size
            min_rbc_px = int(np.pi * (args.rbc_radius * 0.50) ** 2)
            n_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(rbc_mask)
            rbc_mask_clean = np.zeros_like(rbc_mask)
            for l in range(1, n_c):
                if stats_c[l, cv2.CC_STAT_AREA] >= min_rbc_px:
                    rbc_mask_clean[lbl_c == l] = 255
            rbc_mask = rbc_mask_clean

            out = draw_overlay(frame, rbc_mask, neutro_mask)
            draw_legend(out, tier=tiers[i])
        out_frames.append(out)

    print(f"\nWriting: {output_path}")
    iio.imwrite(str(output_path), np.stack(out_frames), plugin="pyav",
                codec="h264", fps=int(round(fps)), out_pixel_format="yuv420p")
    print(f"Done → {output_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
