import cv2
import numpy as np
import os

class LandingSmoother2D:
    """
    Adaptive exponential smoothing for (x, y) using Gaussian-decay alpha based on jump distance.
    """
    def __init__(self, k_sensitivity=0.0005, min_alpha=0.05):
        self.prev_coord = None  # float np.array([x, y])
        self.k = float(k_sensitivity)
        self.min_alpha = float(min_alpha)

    def reset(self):
        self.prev_coord = None

    def update(self, current_raw_coord):
        current_raw = np.array(current_raw_coord, dtype=float)

        if self.prev_coord is None:
            self.prev_coord = current_raw
            return tuple(np.round(self.prev_coord).astype(int)), 0.0, 1.0

        distance = float(np.linalg.norm(current_raw - self.prev_coord))

        alpha = float(np.exp(-self.k * (distance ** 2)))
        alpha = max(self.min_alpha, alpha)

        smoothed = self.prev_coord + alpha * (current_raw - self.prev_coord)
        self.prev_coord = smoothed

        return tuple(np.round(smoothed).astype(int)), distance, alpha


class StickyCircleLandingZoneFinder:
    """
    Stateful landing zone selector that prefers staying near the previous landing zone
    unless a new candidate is sufficiently better.

    Uses per-blob maximum inscribed circle radius (distance transform peak) and applies:
        score = alpha * r_norm - beta * d_norm
    """
    def __init__(
        self,
        drone_size=15,
        alpha=1.0,
        beta=0.5,
        normalize_by="width",
        smoothing_enabled=True,
        smoothing_k=0.0005,
        smoothing_min_alpha=0.05,
    ):
        self.drone_size = drone_size
        self.min_radius = drone_size / 2.0
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_by = normalize_by  # "width" | "diag" | "none"
        self.previous_center = None
        self.target_locked = False

        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoother = LandingSmoother2D(k_sensitivity=smoothing_k, min_alpha=smoothing_min_alpha)

    def reset(self):
        self.previous_center = None
        self.target_locked = False
        self.smoother.reset()

    def _norm_factor(self, h, w):
        if self.normalize_by == "width":
            return float(w)
        if self.normalize_by == "diag":
            return float(np.hypot(w, h))
        return 1.0

    def _extract_candidates(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # White = safe
        white_mask = (img_rgb == [255, 255, 255]).all(axis=2).astype(np.uint8) * 255
        if np.sum(white_mask) == 0:
            return []

        num_labels, labels = cv2.connectedComponents(white_mask)

        candidates = []
        for lab in range(1, num_labels):
            component = (labels == lab).astype(np.uint8) * 255
            if np.sum(component) == 0:
                continue

            dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
            r = float(dist.max())
            if r <= 0:
                continue

            cy, cx = np.unravel_index(dist.argmax(), dist.shape)
            candidates.append({"center": (int(cx), int(cy)), "radius": r})

        return candidates

    def find(self, input_image_path, output_folder, save_debug=True):
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_image_path):
            print(f"Error: Input image does not exist: {input_image_path}")
            return None

        filename = os.path.basename(input_image_path)
        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not load {filename}")
            return None

        h, w = img.shape[:2]
        norm = self._norm_factor(h, w)

        candidates = self._extract_candidates(img)
        if not candidates:
            output_filename = f"{os.path.splitext(filename)[0]}_landing_zone.png"
            cv2.imwrite(os.path.join(output_folder, output_filename), img)
            self.target_locked = False
            # also reset smoother so next reacquire doesn't blend from stale point
            self.smoother.reset()
            return None

        candidates = [c for c in candidates if c["radius"] >= self.min_radius]
        if not candidates:
            output_filename = f"{os.path.splitext(filename)[0]}_landing_zone.png"
            cv2.imwrite(os.path.join(output_folder, output_filename), img)
            self.target_locked = False
            self.smoother.reset()
            return None

        # Select winner (sticky score)
        if not self.target_locked or self.previous_center is None:
            winner = max(candidates, key=lambda c: c["radius"])
        else:
            px, py = self.previous_center
            best_score = -1e18
            winner = None
            for c in candidates:
                cx, cy = c["center"]
                r = c["radius"]
                d = float(np.hypot(cx - px, cy - py))

                r_norm = r / norm
                d_norm = d / norm
                score = (self.alpha * r_norm) - (self.beta * d_norm)

                if score > best_score:
                    best_score = score
                    winner = c

        raw_center_x, raw_center_y = winner["center"]
        max_radius = float(winner["radius"])

        # Update "sticky" state using RAW winner center (selection space)
        self.previous_center = (raw_center_x, raw_center_y)
        self.target_locked = True

        # Apply adaptive smoothing to publishing/drawing coordinate
        if self.smoothing_enabled:
            (center_x, center_y), smooth_dist, smooth_alpha = self.smoother.update((raw_center_x, raw_center_y))
        else:
            center_x, center_y = raw_center_x, raw_center_y
            smooth_dist, smooth_alpha = 0.0, 1.0

        frame_center_x = w // 2
        frame_center_y = h // 2
        vector_x = frame_center_x - center_x
        vector_y = frame_center_y - center_y

        # Draw visualization using SMOOTHED center
        output_img = img.copy()
        circle_color = (0, 255, 0)
        cv2.circle(output_img, (center_x, center_y), int(max_radius), circle_color, 1)
        cv2.drawMarker(output_img, (center_x, center_y), circle_color, cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # (optional) draw raw center for debugging
        if save_debug and (raw_center_x, raw_center_y) != (center_x, center_y):
            cv2.drawMarker(output_img, (raw_center_x, raw_center_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

        frame_center_color = (255, 0, 0)
        cv2.drawMarker(output_img, (frame_center_x, frame_center_y), frame_center_color, cv2.MARKER_SQUARE, markerSize=8, thickness=2)
        cv2.arrowedLine(output_img, (center_x, center_y), (frame_center_x, frame_center_y), frame_center_color, thickness=2, tipLength=0.1)

        cv2.putText(output_img, f"R={max_radius:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, circle_color, 2)
        if save_debug:
            cv2.putText(output_img, f"Sticky: a={self.alpha}, b={self.beta}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if self.smoothing_enabled:
                cv2.putText(output_img, f"Smooth: k={self.smoother.k}, alpha={smooth_alpha:.3f}, d={smooth_dist:.1f}px", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        output_filename = f"{os.path.splitext(filename)[0]}_landing_zone.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, output_img)

        diameter = max_radius * 2.0
        print(f"{filename}, Diameter: {diameter:.1f}px, Center: {center_x}x{center_y}, Vector to frame center: ({vector_x}, {vector_y})")

        return {
            "filename": filename,
            "center": (center_x, center_y),          # smoothed
            "center_raw": (raw_center_x, raw_center_y),
            "frame_center": (frame_center_x, frame_center_y),
            "vector_to_frame_center": (vector_x, vector_y),
            "radius": float(max_radius),
            "diameter": float(diameter),
            "meets_minimum": True,
            "drone_size": self.drone_size,
            "min_radius_required": self.min_radius,
            "output_path": output_path,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_alpha": float(smooth_alpha),
            "smoothing_distance_px": float(smooth_dist),
        }


# Example usage
if __name__ == "__main__":
    input_image = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/masked_merged_single/semantic_frame_000246_modified.png"
    output_folder = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/landing_zones_single"

    finder = StickyCircleLandingZoneFinder(drone_size=15)
    result = finder.find(input_image, output_folder)

    if result:
        print(f"Landing zone processing successful!")
        print(f"   Center: ({result['center'][0]}, {result['center'][1]})")
        print(f"   Vector to frame center: {result['vector_to_frame_center']}")
        print(f"   Meets drone size requirement: {result['meets_minimum']}")
        print(f"   Output saved to: {result['output_path']}")
    else:
        print("Landing zone processing failed!")