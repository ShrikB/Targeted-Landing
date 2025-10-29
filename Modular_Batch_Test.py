from OneFormer_Inference_Video import process_video_with_oneformer
from Mask_Merge import process_semantic_masks
from Landing_Zone import find_landing_zones
import os
import time

# ========== CONFIGURATION ==========
model_path = "model/model7_cusdat"
video_input = "/home/shrekfedora/Projects/Targeted-Landing/inputs/gurt.mp4"
base_output_folder = "/home/shrekfedora/Projects/Targeted-Landing/outputs/pipeline_test"

# Define safe and unsafe classes
safe_classes = [
    [159, 66, 133],  # Purple sidewalk: #9F4285
    [38, 127, 102],  # Medium green road: #267F66
]

unsafe_classes = [
    [93, 220, 53],   # Green car: #5DDC35
]

# OneFormer processing parameters
window_size = 20
threshold = 0.10
frame_resolution = (512, 512)

# ========== FOLDER SETUP ==========
semantic_output = os.path.join(base_output_folder, "semantic_frames")
masked_output = os.path.join(base_output_folder, "masked_merged")
landing_zones_output = os.path.join(base_output_folder, "landing_zones")

os.makedirs(base_output_folder, exist_ok=True)

# ========== STAGE 1: ONEFORMER VIDEO PROCESSING ==========
try:
    frames_processed = process_video_with_oneformer(
        model_path=model_path,
        output_folder=semantic_output,
        video_input=video_input,
        window_size=window_size,
        threshold=threshold,
        frame_resolution=frame_resolution
    )
except Exception as e:
    print(f"❌ Stage 1 Failed: {e}")
    exit(1)

# ========== STAGE 2: MASK MERGING ==========
try:
    masks_processed = process_semantic_masks(
        input_folder=semantic_output,
        output_folder=masked_output,
        safe_classes=safe_classes,
        unsafe_classes=unsafe_classes
    )
except Exception as e:
    print(f"❌ Stage 2 Failed: {e}")
    exit(1)

# ========== STAGE 3: LANDING ZONE DETECTION ==========
try:
    landing_results = find_landing_zones(
        input_folder=masked_output,
        output_folder=landing_zones_output
    )
except Exception as e:
    print(f"❌ Stage 3 Failed: {e}")
    exit(1)

# ========== FIND BEST LANDING ZONE ==========
if landing_results:
    best_zone = max(landing_results, key=lambda x: x['radius'])
    print(f"🎯 BEST LANDING ZONE:")
    print(f"   Center: ({best_zone['center'][0]}, {best_zone['center'][1]})")
    print(f"   Vector to frame center: {best_zone['vector_to_frame_center']}")
    
    print(f"🚁 LANDING COORDINATES: ({best_zone['center'][0]}, {best_zone['center'][1]})")
else:
    print(f"❌ NO SAFE LANDING ZONES FOUND")