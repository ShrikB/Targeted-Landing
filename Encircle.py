import os
import cv2
import numpy as np
import time

# ==== Tunables (edit up here) ================================================
Inputs = 0
SMALL_RADIUS_PIXELS = 40      # default inner/smaller circle radius (px), if no prev provided
OUTPUT_DIR = "outputs/circles_out"    # where to save binary masks with two circles
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_encircle(overlay, frame_id, prev_small_radius=None):
    """
    From a binary mask (white=foreground), find:
      - the largest inscribed circle (max EDT)
      - a smaller circle (either prev_small_radius or SMALL_RADIUS_PIXELS),
        clamped to never exceed the max circle.

    Saves a binary mask with exactly those two filled circles (white=255, else 0).

    Args:
        overlay (np.ndarray): Binary mask image. Accepts:
                              - single-channel uint8 (0/255)
                              - or 3-channel where non-zero means foreground
        frame_id (str|int):   Label for output filenames.
        prev_small_radius (float|int|None): If given, use this as the smaller
                              circle radius for continuity across frames.

    Returns:
        dict: {
            'center': (cx, cy),                 # pixel center of both circles
            'max_radius_px': float,             # largest inscribed radius (px)
            'small_radius_px': float,           # chosen smaller radius (px)
            'meets_minimum': bool,              # small_radius <= max_radius
            'white_area_pixels': int,           # count of foreground px in input mask
            'mask_path': str,                   # saved path for the 2-circle binary mask
        }
    """
    encircle_start = time.time()

    # --- 1) Ensure a single-channel binary mask (uint8 0/255) ----------------
    if overlay.ndim == 3:
        # Any non-zero across channels is foreground
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    else:
        gray = overlay.copy()

    # Binarize robustly (treat >0 as foreground)
    mask = (gray > 0).astype(np.uint8) * 255

    white_area_pixels = int(mask.sum() // 255)
    if white_area_pixels == 0:
        # Nothing to do: save an empty two-circle mask (all zeros)
        out_path = os.path.join(OUTPUT_DIR, f"circles_{frame_id}.png")
        empty = np.zeros_like(mask, dtype=np.uint8)
        cv2.imwrite(out_path, empty)
        return {
            'center': None,
            'max_radius_px': 0.0,
            'small_radius_px': 0.0,
            'meets_minimum': False,
            'white_area_pixels': 0,
            'mask_path': out_path,
        }

    # --- 2) Distance Transform (largest inscribed circle) ---------------------
    # OpenCV EDT expects non-zero = foreground; returns float32 distances in px
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)  # 3 is a bit faster than 5
    max_radius = float(dist.max())
    cy, cx = np.unravel_index(dist.argmax(), dist.shape)

    # --- 3) Decide smaller radius --------------------------------------------
    small_r = float(prev_small_radius) if prev_small_radius is not None else float(SMALL_RADIUS_PIXELS)
    # Clamp so the smaller circle never exceeds the max possible
    small_r = min(small_r, max_radius)
    meets_minimum = (small_r <= max_radius) and (small_r > 0)

    # --- 4) Build the 2-circle binary mask (filled) ---------------------------
    two_circles = np.zeros_like(mask, dtype=np.uint8)
    # Draw filled circles in white (255). If radii < 1, skip draw to avoid tiny kernels.
    if max_radius >= 1:
        cv2.circle(two_circles, (int(cx), int(cy)), int(max_radius), 255, thickness=-1)
    if small_r >= 1:
        cv2.circle(two_circles, (int(cx), int(cy)), int(small_r), 255, thickness=-1)

    # (Optional) If you want the smaller circle to be visually distinct in a preview,
    # you can also save a color visualization, but per your ask we’re saving a binary mask.
    out_path = os.path.join(OUTPUT_DIR, f"circles_{frame_id}.png")
    cv2.imwrite(out_path, two_circles)

    # --- 5) Return metadata ---------------------------------------------------
    return {
        'center': (int(cx), int(cy)),
        'max_radius_px': max_radius,
        'small_radius_px': small_r,
        'meets_minimum': meets_minimum,
        'white_area_pixels': white_area_pixels,
        'mask_path': out_path,
    }
