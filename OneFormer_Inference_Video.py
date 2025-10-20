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
matplotlib.use('Agg')

def process_video_with_oneformer(model_path, output_folder, video_input, window_size=20, threshold=0.2, frame_resolution=(512, 512)):
    """
    Process a video with OneFormer semantic segmentation model.
    
    Args:
        model_path (str): Path to the OneFormer model directory
        output_fol (str): Output folder for processed frames
        vid (str): Path to input video file
        window_size (int, optional): Size of rolling average window. Defaults to 20.
        threshold (float, optional): Threshold for frame change detection. Defaults to 0.2.
        frame_resolution (tuple, optional): Resolution to resize frames to. Defaults to (512, 512).
    
    Returns:
        int: Number of frames processed
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Load model and processor
    processor = OneFormerProcessor.from_pretrained(model_path)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_path)
    model.eval()
    model.model.is_training = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open video
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_input}")
    
    frame_count = 0

    # Pre-generate a color palette
    np.random.seed(1234)
    palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

    # Initialize rolling average buffer
    avg_buffer = deque(maxlen=window_size)

    print(f"Starting video processing...")
    print(f"Model: {model_path}")
    print(f"Output: {output_folder}")
    print(f"Video: {video_input}")
    print(f"Frame resolution: {frame_resolution}")
    print(f"Window size: {window_size}, Threshold: {threshold}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            
            # Resize frame
            imgr = cv2.resize(frame, frame_resolution)
            
            # Prepare inputs
            inputs = processor(images=imgr, task_inputs=["semantic"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # Run inference
            with torch.cuda.amp.autocast(), torch.no_grad():
                outputs = model(**inputs)

            # Post-process segmentation
            seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[imgr.shape[:2]])[0].cpu().numpy().astype(np.uint8)
            
            # Colorize & overlay
            color_mask = palette[seg]
            overlay = cv2.addWeighted(imgr, 0.0, color_mask, 1.0, 0)
            avg_col = color_mask.mean(axis=(0, 1))

            # Check for significant change using rolling average
            if len(avg_buffer) > 0:
                baseline = np.stack(avg_buffer, axis=0).mean(axis=0)
                diff = np.linalg.norm(avg_col - baseline) / (np.linalg.norm(baseline) + 1e-6)
                if diff > threshold:
                    print(f"[{frame_count}] dev {diff:.3f} > {threshold}, skipping")
                    # Still push it so average will "catch up" as camera moves
                    avg_buffer.append(avg_col)
                    frame_count += 1
                    continue

            # Accept this frame
            avg_buffer.append(avg_col)

            # Draw legend
            """for i, lid in enumerate(np.unique(seg)):
                y = 30 * i + 20
                c = palette[lid].tolist()
                cv2.rectangle(overlay, (5, y - 15), (25, y + 5), c, -1)
                cv2.putText(overlay,
                           f"{lid}:{model.config.id2label[lid]}",
                           (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
            """
            # Save frame
            cv2.imwrite(f"{output_folder}/semantic_{frame_count}.png", overlay)

            frame_count += 1
            end = time.time()
            print(f"Frame {frame_count} processed in {end - start:.3f} seconds")

    finally:
        cap.release()

    print(f"Video processing complete. Processed {frame_count} frames.")
    return frame_count

# Example usage (can be removed when importing)
if __name__ == "__main__":
    model_path = "model/model7_cusdat"
    output_folder = "outputs/gurt/"
    video_input = "inputs/gurt.mp4"

    frames_processed = process_video_with_oneformer(model_path, output_folder, video_input)
    print(f"Total frames processed: {frames_processed}")