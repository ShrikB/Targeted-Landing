from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import numpy as np
import torch
import cv2
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from collections import defaultdict
from collections import deque
import matplotlib
import threading
matplotlib.use('Agg')

model_path = "model/model7_cusdat/"
output_fol = "outputs/simulated/semantic(model7)/"
circles_fol = "outputs/simulated/circles(model7)/"
if not os.path.exists(output_fol):
    os.makedirs(output_fol, exist_ok=True)
if not os.path.exists(circles_fol):
    os.makedirs(circles_fol, exist_ok=True)

processor = OneFormerProcessor.from_pretrained(model_path)
model = OneFormerForUniversalSegmentation.from_pretrained(model_path)
model.eval()
model.model.is_training = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

vid = "inputs/DJI_20250408174712_0038_D.MP4"

# Pre-generate a color palette (you can choose any palette you like)
np.random.seed(1234)
palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

window_size = 20
avg_buffer = deque(maxlen=window_size)
threshold = 0.2   # 20% relative change

# Encircle configuration
min_diameter_pixels = 30  # Half a meter = 30 pixels
min_radius_pixels = min_diameter_pixels / 2

# Shared variables for current frame access
current_frame = None
current_frame_id = 0
frame_lock = threading.Lock()
inference_busy = threading.Lock()
stop_threads = threading.Event()
video_active = threading.Event()
frame_count = 0
inference_count = 0

# Performance metrics
inference_times = []
encircle_times = []
first_inference_time = None
last_inference_time = None

def video_capture_thread():
    """Thread that continuously reads frames from video at normal playback speed"""
    global frame_count, current_frame, current_frame_id
    cap = cv2.VideoCapture(vid)
    
    # Get video properties for timing
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 fps if unknown
    frame_time = 1.0 / fps
    
    print(f"Video FPS: {fps}, Frame time: {frame_time:.3f}s")
    
    start_time = time.time()
    video_active.set()  # Signal that video is active
    
    while cap.isOpened() and not stop_threads.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        
        # Calculate when this frame should be displayed
        expected_time = start_time + (frame_count * frame_time)
        current_time = time.time()
        
        # Sleep if we're ahead of schedule
        if current_time < expected_time:
            time.sleep(expected_time - current_time)
        
        # Update current frame (thread-safe)
        with frame_lock:
            current_frame = frame.copy()
            current_frame_id = frame_count
        
        print(f"Video at frame {frame_count}")
        frame_count += 1
    
    cap.release()
    video_active.clear()  # Signal that video is no longer active
    print("Video capture thread finished")

def process_encircle(overlay, frame_id):
    """
    Process semantic overlay to find the largest safe landing circle
    
    Args:
        overlay: Semantic segmentation overlay image
        frame_id: Frame identifier
    
    Returns:
        dict: Results including center, radius, and whether it meets minimum size
    """
    encircle_start = time.time()
    
    # Copy for circle visualization
    circle_img = overlay.copy()

    # Build the yellow-road mask (assuming yellow represents safe landing area)
    b, g, r = cv2.split(overlay)
    road_mask = ((g > 200) & (r > 200) & (b < 200)).astype(np.uint8) * 255

    # Check if there's any road area detected
    if np.sum(road_mask) == 0:
        # Save image with no circle
        cv2.imwrite(f"{circles_fol}/circle_{frame_id}.png", circle_img)
        encircle_end = time.time()
        encircle_times.append(encircle_end - encircle_start)
        return {
            'center': None,
            'radius': 0,
            'meets_minimum': False,
            'landing_area_pixels': 0
        }

    # Distance transform to find largest inscribed circle
    dist = cv2.distanceTransform(road_mask, cv2.DIST_L2, 5)
    max_radius = dist.max()
    cy, cx = np.unravel_index(dist.argmax(), dist.shape)

    # Check if circle meets minimum size requirement
    meets_minimum = max_radius >= min_radius_pixels
    
    # Choose circle color based on size requirement
    circle_color = (0, 255, 0) if meets_minimum else (0, 0, 255)  # Green if good, red if too small
    
    # Draw circle and center marker
    cv2.circle(circle_img, (cx, cy), int(max_radius), circle_color, 2)
    cv2.drawMarker(circle_img, (cx, cy), circle_color, cv2.MARKER_CROSS,
                   markerSize=12, thickness=2)

    # Save the result
    cv2.imwrite(f"{circles_fol}/circle_{frame_id}.png", circle_img)
    
    encircle_end = time.time()
    encircle_times.append(encircle_end - encircle_start)
    
    return {
        'center': (cx, cy),
        'radius': float(max_radius),
        'diameter': float(max_radius * 2),
        'meets_minimum': meets_minimum,
        'landing_area_pixels': int(np.sum(road_mask) / 255)
    }

def inference_thread():
    """Thread that processes the current frame when inference is not busy"""
    global inference_count, first_inference_time, last_inference_time
    
    while video_active.is_set() or not stop_threads.is_set():
        # Only process if inference is not busy
        if inference_busy.acquire(blocking=False):
            try:
                # Get current frame snapshot
                with frame_lock:
                    if current_frame is not None:
                        frame_to_process = current_frame.copy()
                        frame_id_to_process = current_frame_id
                        has_frame = True
                    else:
                        has_frame = False
                
                if has_frame:
                    # Record timing for inference frame rate calculation
                    current_time = time.time()
                    if first_inference_time is None:
                        first_inference_time = current_time
                    last_inference_time = current_time
                    
                    process_frame(frame_id_to_process, frame_to_process)
                    inference_count += 1
                
            finally:
                inference_busy.release()
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.01)
    
    print("Inference thread finished")

def process_frame(frame_id, frame):
    """Process a single frame with inference and encircling"""
    start = time.time()
    
    imgr = cv2.resize(frame, (512, 512))
    inputs = processor(images=imgr, task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # Time inference only (without encircling)
    inference_start = time.time()
    with torch.cuda.amp.autocast(), torch.no_grad():
        outputs = model(**inputs)

    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[imgr.shape[:2]])[0].cpu().numpy().astype(np.uint8)
    
    # colorize & overlay
    color_mask = palette[seg]
    overlay = cv2.addWeighted(imgr, 0.0, color_mask, 1.0, 0)
    avg_col = color_mask.mean(axis=(0,1))

    # if we have any history, compute the rolling baseline
    if len(avg_buffer) > 0:
        baseline = np.stack(avg_buffer, axis=0).mean(axis=0)
        diff = np.linalg.norm(avg_col - baseline) / (np.linalg.norm(baseline) + 1e-6)
        if diff > threshold:
            print(f"[{frame_id}] dev {diff:.3f} > {threshold}, skipping save")
            avg_buffer.append(avg_col)
            return

    # otherwise accept this frame
    avg_buffer.append(avg_col)

    # write out semantic segmentation
    cv2.imwrite(f"{output_fol}/semantic_{frame_id}.png", overlay)
    
    inference_end = time.time()
    inference_only_time = inference_end - inference_start
    
    # Process encircling immediately after semantic segmentation
    circle_result = process_encircle(overlay, frame_id)
    
    end = time.time()
    total_time = end - start
    
    # Record timing metrics
    inference_times.append(inference_only_time)
    
    # Print results
    if circle_result['meets_minimum']:
        status = f"SAFE - R:{circle_result['radius']:.1f}px"
    elif circle_result['center'] is None:
        status = "NO LANDING AREA"
    else:
        status = f"TOO SMALL - R:{circle_result['radius']:.1f}px"
    
    encircle_time = encircle_times[-1] if encircle_times else 0
    print(f"Frame {frame_id} processed in {total_time:.3f}s (inf:{inference_only_time:.3f}s, enc:{encircle_time:.3f}s) - {status}")

def calculate_performance_metrics():
    """Calculate and display performance metrics"""
    if not inference_times or first_inference_time is None or last_inference_time is None:
        print("No inference data available for metrics")
        return
    
    # Calculate inference frame rate
    total_inference_time = last_inference_time - first_inference_time
    if total_inference_time > 0:
        inference_fps = (inference_count - 1) / total_inference_time  # -1 because we measure intervals
    else:
        inference_fps = 0
    
    # Calculate percentage of frames processed
    if frame_count > 0:
        frames_processed_percentage = (inference_count / frame_count) * 100
    else:
        frames_processed_percentage = 0
    
    # Calculate average times
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_encircle_time = np.mean(encircle_times) if encircle_times else 0
    avg_total_time = avg_inference_time + avg_encircle_time
    
    # Calculate performance impact of encircling
    if avg_inference_time > 0:
        encircle_overhead_percentage = (avg_encircle_time / avg_inference_time) * 100
    else:
        encircle_overhead_percentage = 0
    
    # Calculate theoretical max FPS without encircling
    if avg_inference_time > 0:
        max_fps_without_encircle = 1.0 / avg_inference_time
    else:
        max_fps_without_encircle = 0
    
    # Calculate actual FPS with encircling
    if avg_total_time > 0:
        actual_fps_with_encircle = 1.0 / avg_total_time
    else:
        actual_fps_with_encircle = 0
    
    print(f"\n=== PERFORMANCE METRICS ===")
    print(f"Video capture:")
    print(f"  Total frames captured: {frame_count}")
    print(f"  Total frames processed: {inference_count}")
    print(f"  Frames processed: {frames_processed_percentage:.1f}%")
    print(f"")
    print(f"Inference performance:")
    print(f"  Actual inference FPS: {inference_fps:.2f}")
    print(f"  Average inference time: {avg_inference_time:.3f}s")
    print(f"  Theoretical max FPS (inference only): {max_fps_without_encircle:.2f}")
    print(f"")
    print(f"Encircling performance:")
    print(f"  Average encircle time: {avg_encircle_time:.3f}s")
    print(f"  Encircle overhead: {encircle_overhead_percentage:.1f}% of inference time")
    print(f"")
    print(f"Combined performance:")
    print(f"  Average total time: {avg_total_time:.3f}s")
    print(f"  Actual FPS (with encircling): {actual_fps_with_encircle:.2f}")
    print(f"  Performance impact: {((max_fps_without_encircle - actual_fps_with_encircle) / max_fps_without_encircle * 100):.1f}% FPS reduction")

# Start threads
print("Starting camera simulation with integrated encircling...")
capture_thread = threading.Thread(target=video_capture_thread, daemon=True)
inference_thread_obj = threading.Thread(target=inference_thread, daemon=True)

capture_thread.start()
inference_thread_obj.start()

try:
    # Keep main thread alive and monitor progress
    while capture_thread.is_alive():
        time.sleep(1)
        with frame_lock:
            current_id = current_frame_id
        
        # Calculate real-time metrics
        if inference_count > 1 and first_inference_time is not None:
            elapsed_time = time.time() - first_inference_time
            current_fps = (inference_count - 1) / elapsed_time if elapsed_time > 0 else 0
            frame_percentage = (inference_count / max(frame_count, 1)) * 100
            print(f"Status: Video frame: {frame_count}, Current frame: {current_id}, "
                  f"Processed: {inference_count} ({frame_percentage:.1f}%), "
                  f"Inference FPS: {current_fps:.2f}")
        else:
            print(f"Status: Video frame: {frame_count}, Current frame: {current_id}, Processed: {inference_count}")
    
    # Allow some time for remaining inference to complete
    time.sleep(2)
    
except KeyboardInterrupt:
    print("\nInterrupted by user")

# Signal threads to stop
stop_threads.set()

# Wait for threads to finish
capture_thread.join(timeout=5)
inference_thread_obj.join(timeout=5)

print(f"\nSimulation completed. Total captured: {frame_count}, Total processed: {inference_count}")
print(f"Semantic images saved to: {output_fol}")
print(f"Circle images saved to: {circles_fol}")

# Calculate and display final performance metrics
calculate_performance_metrics()
