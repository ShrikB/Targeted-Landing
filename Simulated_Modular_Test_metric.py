from OneFormer_Inference_Image import process_image_with_oneformer
from Mask_Merge_Singular import process_single_semantic_mask
from Landing_Zone_Singular import find_single_landing_zone
import cv2
import os
import time
import json
import numpy as np
from scipy.stats import mode

# ========== CONFIGURATION ==========
model_path = "/home/shrekfedora/Projects/Targeted-Landing/model/model_23_basedon19"
video_input = "/home/shrekfedora/Projects/Targeted-Landing/inputs/8564838-hd_1920_1080_30fps.mp4"
base_output_folder = "/home/shrekfedora/Projects/Targeted-Landing/outputs/polish_test11"

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

# CLAHE parameters
clahe_clip_limit = 3.0
clahe_tile_size = (16, 16)

# Temporal Stabilization parameters
temporal_stabilization_enabled = True
temporal_buffer_size = 8

# ========== FOLDER SETUP ==========
frames_folder = os.path.join(base_output_folder, "extracted_frames")
semantic_folder = os.path.join(base_output_folder, "semantic_output")
semantic_stabilized_folder = os.path.join(base_output_folder, "semantic_stabilized")
masked_folder = os.path.join(base_output_folder, "masked_output")
landing_zones_folder = os.path.join(base_output_folder, "landing_zones")

# Create all directories
os.makedirs(base_output_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(semantic_folder, exist_ok=True)
os.makedirs(semantic_stabilized_folder, exist_ok=True)
os.makedirs(masked_folder, exist_ok=True)
os.makedirs(landing_zones_folder, exist_ok=True)

print(f"Processing video: {video_input}")
print(f"Output directory: {base_output_folder}")
print(f"CLAHE enabled - Clip Limit: {clahe_clip_limit}, Tile Size: {clahe_tile_size}")
print(f"Temporal Stabilization enabled - Buffer Size: {temporal_buffer_size}")

# ========== TIMING TRACKING ==========
timing_data = {
    "metadata": {
        "video_input": video_input,
        "output_folder": base_output_folder,
        "frame_resolution": frame_resolution,
        "total_frames": 0,
        "processed_frames": 0,
        "fps": 0,
        "clahe_enabled": True,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_tile_size": clahe_tile_size,
        "temporal_stabilization_enabled": temporal_stabilization_enabled,
        "temporal_buffer_size": temporal_buffer_size
    },
    "per_frame_timings": [],
    "stage_statistics": {
        "frame_extraction": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0},
        "clahe_enhancement": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0},
        "semantic_segmentation": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0},
        "temporal_stabilization": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0},
        "mask_merging": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0},
        "landing_zone_detection": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0},
        "total_per_frame": {"total_time": 0, "count": 0, "avg_time": 0, "min_time": float('inf'), "max_time": 0}
    },
    "overall_timing": {
        "start_time": "",
        "end_time": "",
        "total_duration": 0
    }
}

def apply_rgb_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to RGB image.
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting (1.0 to 4.0 recommended)
        tile_size: Size of grid tiles for local histogram equalization
    
    Returns:
        CLAHE-enhanced BGR image
    """
    # Convert BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE to L (lightness) channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)
    
    # Merge the enhanced L channel with original a and b channels
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

class MaskStabilizer:
    """
    Stabilizes semantic masks using optical flow and temporal voting.
    Reduces flicker and noise in segmentation output.
    """
    def __init__(self, buffer_size=3):
        """
        Initialize mask stabilizer.
        
        Args:
            buffer_size: Number of frames to use for temporal voting
        """
        self.buffer_size = buffer_size
        self.mask_buffer = []
        self.prev_gray = None
        # UltraFast preset is optimized for real-time performance
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def stabilize(self, current_bgr_frame, current_semantic_mask):
        """
        Stabilize current semantic mask using optical flow and temporal voting.
        
        Args:
            current_bgr_frame: Current video frame in BGR format
            current_semantic_mask: Current semantic segmentation mask (RGB colors)
        
        Returns:
            Stabilized semantic mask with preserved RGB class colors
        """
        # Ensure input is 3-channel
        if len(current_semantic_mask.shape) == 2:
            current_semantic_mask = cv2.cvtColor(current_semantic_mask, cv2.COLOR_GRAY2BGR)
        if current_semantic_mask.shape[2] == 1:
            current_semantic_mask = cv2.cvtColor(current_semantic_mask, cv2.COLOR_GRAY2BGR)
        
        current_gray = cv2.cvtColor(current_bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # If first frame, initialize buffer
        if self.prev_gray is None:
            self.prev_gray = current_gray
            self.mask_buffer = [current_semantic_mask.copy() for _ in range(self.buffer_size)]
            return current_semantic_mask.copy()

        # 1. Calculate optical flow between frames
        flow = self.dis.calc(self.prev_gray, current_gray, None)
        h, w = current_gray.shape
        
        # Create remap maps for warping
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.float32(map_x + flow[..., 0])
        map_y = np.float32(map_y + flow[..., 1])

        # 2. Warp all masks in buffer to align with current frame
        warped_buffer = []
        for mask in self.mask_buffer:
            try:
                # INTER_NEAREST preserves exact RGB class colors
                warped = cv2.remap(
                    mask, 
                    map_x, 
                    map_y, 
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE
                )
                warped_buffer.append(warped)
            except Exception as e:
                print(f"   Warning: Warping failed, using original mask: {e}")
                warped_buffer.append(mask.copy())

        # 3. Add current raw mask and manage buffer size
        warped_buffer.append(current_semantic_mask.copy())
        if len(warped_buffer) > self.buffer_size:
            warped_buffer.pop(0)
            
        self.mask_buffer = warped_buffer
        self.prev_gray = current_gray

        # 4. Temporal Voting (Mode) - find most common color per pixel
        smoothed = np.zeros_like(self.mask_buffer[0], dtype=np.uint8)
        
        h, w, c = smoothed.shape
        
        for ch in range(c):
            channel_stack = np.stack([mask[:, :, ch] for mask in self.mask_buffer], axis=-1)
            smoothed_channel, _ = mode(channel_stack, axis=-1, keepdims=False)
            smoothed[:, :, ch] = smoothed_channel.astype(np.uint8)
        
        return smoothed


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def update_stage_stats(stage_name, duration):
    """Update statistics for a specific processing stage"""
    stats = timing_data["stage_statistics"][stage_name]
    stats["total_time"] += duration
    stats["count"] += 1
    stats["min_time"] = min(stats["min_time"], duration)
    stats["max_time"] = max(stats["max_time"], duration)
    stats["avg_time"] = stats["total_time"] / stats["count"]

# ========== INITIALIZE MODULES ==========
mask_stabilizer = MaskStabilizer(buffer_size=temporal_buffer_size) if temporal_stabilization_enabled else None

# ========== STAGE 1: READ VIDEO FRAME BY FRAME ==========
cap = cv2.VideoCapture(video_input)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_input}")
    exit(1)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

timing_data["metadata"]["total_frames"] = total_frames
timing_data["metadata"]["fps"] = fps

print(f"Video properties: {total_frames} frames at {fps:.2f} FPS")
print(f"Processing all frames as they become available")

# Record overall start time
overall_start_time = time.time()
timing_data["overall_timing"]["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(overall_start_time))

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
        
        # Initialize frame timing dictionary
        frame_timing = {
            "frame_number": frame_count,
            "stages": {},
            "total_time": 0,
            "success": False,
            "landing_zone_found": False
        }
        
        # ========== STAGE 2: SAVE FRAME ==========
        stage_start = time.time()
        
        frame_filename = f"frame_{frame_count:06d}.png"
        frame_path = os.path.join(frames_folder, frame_filename)
        
        # Resize frame to target resolution
        resized_frame = cv2.resize(frame, frame_resolution)
        cv2.imwrite(frame_path, resized_frame)
        
        stage_duration = time.time() - stage_start
        frame_timing["stages"]["frame_extraction"] = stage_duration
        update_stage_stats("frame_extraction", stage_duration)
        print(f"   Frame extraction: {stage_duration:.3f}s")
        
        # ========== STAGE 3: APPLY RGB CLAHE ENHANCEMENT ==========
        stage_start = time.time()
        
        try:
            # Apply CLAHE to the resized frame
            clahe_frame = apply_rgb_clahe(
                resized_frame,
                clip_limit=clahe_clip_limit,
                tile_size=clahe_tile_size
            )
            
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["clahe_enhancement"] = stage_duration
            update_stage_stats("clahe_enhancement", stage_duration)
            print(f"   CLAHE enhancement: {stage_duration:.3f}s")
            
            # Overwrite the original frame with CLAHE-enhanced version
            cv2.imwrite(frame_path, clahe_frame)
            clahe_frame_path = frame_path
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["clahe_enhancement"] = stage_duration
            frame_timing["error"] = f"CLAHE enhancement error: {str(e)}"
            print(f"   CLAHE enhancement error: {e}")
            clahe_frame_path = frame_path
        
        # ========== STAGE 4: SEMANTIC SEGMENTATION ==========
        stage_start = time.time()
        
        try:
            success = process_image_with_oneformer(
                model_path=model_path,
                output_folder=semantic_folder,
                image_input=clahe_frame_path,
                frame_resolution=frame_resolution
            )
            
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["semantic_segmentation"] = stage_duration
            update_stage_stats("semantic_segmentation", stage_duration)
            print(f"   Semantic segmentation: {stage_duration:.3f}s")
            
            if not success:
                print(f"Semantic segmentation failed for frame {frame_count}")
                frame_timing["stages"]["temporal_stabilization"] = 0
                frame_timing["stages"]["mask_merging"] = 0
                frame_timing["stages"]["landing_zone_detection"] = 0
                frame_timing["total_time"] = time.time() - frame_start_time
                timing_data["per_frame_timings"].append(frame_timing)
                frame_count += 1
                continue
            
            semantic_filename = f"semantic_frame_{frame_count:06d}.png"
            semantic_path = os.path.join(semantic_folder, semantic_filename)
            semantic_mask = cv2.imread(semantic_path)
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["semantic_segmentation"] = stage_duration
            frame_timing["stages"]["temporal_stabilization"] = 0
            frame_timing["stages"]["mask_merging"] = 0
            frame_timing["stages"]["landing_zone_detection"] = 0
            frame_timing["error"] = str(e)
            frame_timing["total_time"] = time.time() - frame_start_time
            timing_data["per_frame_timings"].append(frame_timing)
            print(f"Semantic segmentation error for frame {frame_count}: {e}")
            frame_count += 1
            continue
        
        # ========== STAGE 5: TEMPORAL STABILIZATION ==========
        if temporal_stabilization_enabled and mask_stabilizer is not None:
            stage_start = time.time()
            
            try:
                stabilized_mask = mask_stabilizer.stabilize(clahe_frame, semantic_mask)
                median_filtered = cv2.medianBlur(stabilized_mask, ksize=5)
                
                stage_duration = time.time() - stage_start
                frame_timing["stages"]["temporal_stabilization"] = stage_duration
                update_stage_stats("temporal_stabilization", stage_duration)
                print(f"   Temporal stabilization: {stage_duration:.3f}s")
                
                stabilized_filename = f"semantic_stabilized_{frame_count:06d}.png"
                stabilized_path = os.path.join(semantic_stabilized_folder, stabilized_filename)
                cv2.imwrite(stabilized_path, median_filtered)
                
                semantic_path = stabilized_path
                
            except Exception as e:
                stage_duration = time.time() - stage_start
                frame_timing["stages"]["temporal_stabilization"] = stage_duration
                frame_timing["error"] = f"Temporal stabilization error: {str(e)}"
                print(f"   Temporal stabilization error: {e}")
        else:
            frame_timing["stages"]["temporal_stabilization"] = 0
        
        # ========== STAGE 6: MASK MERGING ==========
        stage_start = time.time()
        
        try:
            mask_output_path = process_single_semantic_mask(
                input_image_path=semantic_path,
                output_folder=masked_folder,
                safe_classes=safe_classes,
                unsafe_classes=unsafe_classes,
                potential_classes=potential_classes
            )
            
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["mask_merging"] = stage_duration
            update_stage_stats("mask_merging", stage_duration)
            print(f"   Mask merging: {stage_duration:.3f}s")
            
            if not mask_output_path:
                print(f"Mask merging failed for frame {frame_count}")
                frame_timing["stages"]["landing_zone_detection"] = 0
                frame_timing["total_time"] = time.time() - frame_start_time
                timing_data["per_frame_timings"].append(frame_timing)
                frame_count += 1
                continue
                
        except Exception as e:
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["mask_merging"] = stage_duration
            frame_timing["stages"]["landing_zone_detection"] = 0
            frame_timing["error"] = str(e)
            frame_timing["total_time"] = time.time() - frame_start_time
            timing_data["per_frame_timings"].append(frame_timing)
            print(f"Mask merging error for frame {frame_count}: {e}")
            frame_count += 1
            continue
        
        # ========== STAGE 7: LANDING ZONE DETECTION ==========
        stage_start = time.time()
        
        try:
            landing_result = find_single_landing_zone(
                input_image_path=mask_output_path,
                output_folder=landing_zones_folder
            )
            
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["landing_zone_detection"] = stage_duration
            update_stage_stats("landing_zone_detection", stage_duration)
            print(f"   Landing zone detection: {stage_duration:.3f}s")
            
            # ========== STAGE 8: LANDING ZONE TRACKING & SMOOTHING ==========
            # DISABLED - Landing zone tracking removed for now
            frame_timing["stages"]["kalman_filtering"] = 0
            
            # End timing for this frame
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            frame_timing["total_time"] = frame_processing_time
            update_stage_stats("total_per_frame", frame_processing_time)
            
            if landing_result:
                all_results.append(landing_result)
                processed_count += 1
                frame_timing["success"] = True
                frame_timing["landing_zone_found"] = True
                frame_timing["landing_zone_center"] = landing_result['center']
                frame_timing["vector_to_frame_center"] = landing_result['vector_to_frame_center']
                frame_timing["landing_zone_radius"] = landing_result['radius']
                
                print(f"Landing zone found:")
                print(f"   Center: ({landing_result['center'][0]}, {landing_result['center'][1]})")
                print(f"   Vector to frame center: {landing_result['vector_to_frame_center']}")
                print(f"   Radius: {landing_result['radius']:.1f}px")
                print(f"   Total frame processing time: {frame_processing_time:.3f}s")
            else:
                frame_timing["success"] = True
                frame_timing["landing_zone_found"] = False
                print(f"No landing zone found for frame {frame_count}")
                print(f"   Total frame processing time: {frame_processing_time:.3f}s")
            
            timing_data["per_frame_timings"].append(frame_timing)
                
        except Exception as e:
            stage_duration = time.time() - stage_start
            frame_timing["stages"]["landing_zone_detection"] = stage_duration
            frame_timing["error"] = str(e)
            frame_timing["total_time"] = time.time() - frame_start_time
            timing_data["per_frame_timings"].append(frame_timing)
            print(f"Landing zone detection error for frame {frame_count}: {e}")
            print(f"   Total frame processing time: {frame_processing_time:.3f}s")
        
        frame_count += 1

finally:
    cap.release()
    
    # Record overall end time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    timing_data["overall_timing"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(overall_end_time))
    timing_data["overall_timing"]["total_duration"] = overall_duration
    timing_data["metadata"]["processed_frames"] = processed_count
    
    # Convert timing data to serializable format
    timing_data = convert_to_serializable(timing_data)
    
    # Save timing data to JSON
    timing_output_path = os.path.join(base_output_folder, "processing_timing.json")
    with open(timing_output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - TIMING SUMMARY")
    print("="*80)
    print(f"Total frames processed: {frame_count}")
    print(f"Successful landing zones: {processed_count}")
    print(f"Total processing time: {overall_duration:.2f}s")
    print(f"Average time per frame: {overall_duration/frame_count if frame_count > 0 else 0:.3f}s")
    print("\nStage-wise average timings:")
    for stage_name, stats in timing_data["stage_statistics"].items():
        if stats["count"] > 0:
            print(f"  {stage_name:.<30} {stats['avg_time']:.3f}s (min: {stats['min_time']:.3f}s, max: {stats['max_time']:.3f}s)")
    print(f"\nTiming data saved to: {timing_output_path}")
    print("="*80)