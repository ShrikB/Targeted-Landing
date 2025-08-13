# oneformer_realsense_live.py
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import numpy as np
import torch
import cv2
import os
import time
import pyrealsense2 as rs                     # ← RealSense SDK
from collections import deque
import matplotlib
matplotlib.use("Agg")                         # headless back-end for cv2.imwrite

model_path  = "model/model3/"
output_fol  = "output(model3)/"
os.makedirs(output_fol, exist_ok=True)

processor = OneFormerProcessor.from_pretrained(model_path)
model     = OneFormerForUniversalSegmentation.from_pretrained(model_path)
model.eval()
model.model.is_training = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pipeline = rs.pipeline()
config   = rs.config()
# Pick a reasonable colour stream; change res/FPS if you like
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

np.random.seed(1234)
palette       = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
window_size   = 10
avg_buffer    = deque(maxlen=window_size)
threshold     = 0.20          # 20 % relative change
frame_count   = 0

print("⇢ Streaming from RealSense…  Press Ctrl-C to quit")
try:
    while True:
        # Grab a colour frame
        frames       = pipeline.wait_for_frames()
        color_frame  = frames.get_color_frame()
        if not color_frame:           # Occasionally happens; skip if so
            continue
        frame = np.asanyarray(color_frame.get_data())      # shape (480,640,3) BGR

        start = time.time()

        # Resize to model input resolution
        imgr   = cv2.resize(frame, (512, 512))

        # Prepare tensors
        inputs = processor(images=imgr,
                           task_inputs=["semantic"],
                           return_tensors="pt")
        inputs = {k: v.to(device)
                  for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.cuda.amp.autocast(), torch.no_grad():
            outputs = model(**inputs)

        seg = processor.post_process_semantic_segmentation(
                  outputs,
                  target_sizes=[imgr.shape[:2]])[0].cpu().numpy().astype(np.uint8)

        # Colourise & overlay
        color_mask = palette[seg]
        overlay    = cv2.addWeighted(imgr, 0.5, color_mask, 0.5, 0)
        avg_col    = color_mask.mean(axis=(0, 1))          # rolling-avg baseline

        # Rolling baseline check
        if len(avg_buffer):
            baseline = np.stack(avg_buffer).mean(axis=0)
            diff = np.linalg.norm(avg_col - baseline) / (np.linalg.norm(baseline) + 1e-6)
            if diff > threshold:
                print(f"[{frame_count}] Δ {diff:.3f} > {threshold}, skipping")
                avg_buffer.append(avg_col)
                frame_count += 1
                continue
        avg_buffer.append(avg_col)

        # Quick legend
        for i, lid in enumerate(np.unique(seg)):
            y = 30 * i + 20
            c = palette[lid].tolist()          # BGR
            cv2.rectangle(overlay, (5, y - 15), (25, y + 5), c, -1)
            cv2.putText(overlay,
                        f"{lid}:{model.config.id2label.get(lid, 'UNK')}",
                        (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Save frame (optional: comment out and use cv2.imshow if you prefer)
        cv2.imwrite(f"{output_fol}/semantic_{frame_count:06d}.png", overlay)

        frame_count += 1
        print(f"Frame {frame_count} processed in {time.time() - start:.3f}s")

except KeyboardInterrupt:
    print("\n⇠ Stopping")

finally:
    pipeline.stop()
