from OneFormer_Inference_Image import process_image_with_oneformer
from Mask_Merge_Singular import process_single_semantic_mask
from Landing_Zone_Singular import find_single_landing_zone
import cv2
import os
import time

# ========== CONFIGURATION ==========
model_path = "/home/avl-shrek/Documents/Projects/Targeted-Landing/model/model8_cusdat"
video_input = "/home/avl-shrek/Documents/Projects/Targeted-Landing/inputs/12579479_3840_2160_30fps.mp4"
base_output_folder = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/frame_pipeline"

# Define safe and unsafe classes
safe_classes = [
    [159, 66, 133]  # Purple sidewalk: #9F4285
   
]

potential_classes = [
    [38, 127, 102]  # Medium green road: #267F66
]

unsafe_classes = [
    [93, 220, 53]   # Green car: #5DDC35
]

# Processing parameters
frame_resolution = (1280, 720)

# ========== FOLDER SETUP ==========
frames_folder = os.path.join(base_output_folder, "extracted_frames")
semantic_folder = os.path.join(base_output_folder, "semantic_output")
masked_folder = os.path.join(base_output_folder, "masked_output")
landing_zones_folder = os.path.join(base_output_folder, "landing_zones")

# Create all directories
os.makedirs(base_output_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(semantic_folder, exist_ok=True)
os.makedirs(masked_folder, exist_ok=True)
os.makedirs(landing_zones_folder, exist_ok=True)

print(f"Processing video: {video_input}")
print(f"Output directory: {base_output_folder}")

# ========== STAGE 1: READ VIDEO FRAME BY FRAME ==========
cap = cv2.VideoCapture(video_input)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_input}")
    exit(1)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video properties: {total_frames} frames at {fps:.2f} FPS")
print(f"Processing all frames as they become available")

frame_count = 0
processed_count = 0
all_results = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\n--- Processing Frame {frame_count} ---")
        
        # Start timing for this frame
        frame_start_time = time.time()
        
        # ========== STAGE 2: SAVE FRAME ==========
        frame_filename = f"frame_{frame_count:06d}.png"
        frame_path = os.path.join(frames_folder, frame_filename)
        
        # Resize frame to target resolution
        resized_frame = cv2.resize(frame, frame_resolution)
        cv2.imwrite(frame_path, resized_frame)
        
        # ========== STAGE 3: SEMANTIC SEGMENTATION ==========
        try:
            success = process_image_with_oneformer(
                model_path=model_path,
                output_folder=semantic_folder,
                image_input=frame_path,
                frame_resolution=frame_resolution
            )
            
            if not success:
                print(f"Semantic segmentation failed for frame {frame_count}")
                frame_count += 1
                continue
            
            # Generate semantic output path
            semantic_filename = f"semantic_frame_{frame_count:06d}.png"
            semantic_path = os.path.join(semantic_folder, semantic_filename)
            
        except Exception as e:
            print(f"Semantic segmentation error for frame {frame_count}: {e}")
            frame_count += 1
            continue
        
        # ========== STAGE 4: MASK MERGING ==========
        try:
            mask_output_path = process_single_semantic_mask(
                input_image_path=semantic_path,
                output_folder=masked_folder,
                safe_classes=safe_classes,
                unsafe_classes=unsafe_classes,
                potential_classes=potential_classes
            )
            
            if not mask_output_path:
                print(f"Mask merging failed for frame {frame_count}")
                frame_count += 1
                continue
                
        except Exception as e:
            print(f"Mask merging error for frame {frame_count}: {e}")
            frame_count += 1
            continue
        
        # ========== STAGE 5: LANDING ZONE DETECTION ==========
        try:
            landing_result = find_single_landing_zone(
                input_image_path=mask_output_path,
                output_folder=landing_zones_folder
            )
            
            # End timing for this frame
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            
            if landing_result:
                all_results.append(landing_result)
                processed_count += 1
                
                # Print coordinates and timing
                print(f"Landing zone found:")
                print(f"   Center: ({landing_result['center'][0]}, {landing_result['center'][1]})")
                print(f"   Vector to frame center: {landing_result['vector_to_frame_center']}")
                print(f"   Processing time: {frame_processing_time:.3f}s")
            else:
                print(f"No landing zone found for frame {frame_count}")
                print(f"   Processing time: {frame_processing_time:.3f}s")
                
        except Exception as e:
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            print(f"Landing zone detection error for frame {frame_count}: {e}")
            print(f"   Processing time: {frame_processing_time:.3f}s")
        
        frame_count += 1

finally:
    cap.release()

# ========== FIND BEST LANDING ZONE ==========
if all_results:
    best_zone = max(all_results, key=lambda x: x['radius'])
    print(f"\nBEST LANDING ZONE:")
    print(f"   Center: ({best_zone['center'][0]}, {best_zone['center'][1]})")
    print(f"   Vector to frame center: {best_zone['vector_to_frame_center']}")

print(f"\nProcessing complete!")