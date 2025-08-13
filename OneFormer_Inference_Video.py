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

model_path = "model/model4/"
output_fol = "outputs/gurt/"
if not os.path.exists(output_fol):
    os.makedirs(output_fol, exist_ok=True)

processor = OneFormerProcessor.from_pretrained(model_path)
model = OneFormerForUniversalSegmentation.from_pretrained(model_path)
model.eval()
model.model.is_training = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

vid = "inputs/DJI_20250408174712_0038_D.MP4"
cap = cv2.VideoCapture(vid)
frame_count = 0

# Pre-generate a color palette (you can choose any palette you like)
np.random.seed(1234)
palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

window_size = 20
avg_buffer  = deque(maxlen=window_size)
threshold   = 0.2   # 5% relative change

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    imgr = cv2.resize(frame, (512, 512)) #resolution changed from 512x512
    inputs = processor(images=imgr, task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.cuda.amp.autocast(), torch.no_grad():
        outputs = model(**inputs)


    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[imgr.shape[:2]])[0].cpu().numpy().astype(np.uint8)
    # colorize & overlay
    color_mask = palette[seg]
    overlay    = cv2.addWeighted(imgr, 0.0, color_mask, 1.0, 0)
    avg_col = color_mask.mean(axis=(0,1))  # e.g. [123.4, 98.2, 200.1]

    # if we have any history, compute the rolling baseline
    if len(avg_buffer) > 0:
        baseline = np.stack(avg_buffer, axis=0).mean(axis=0)
        diff = np.linalg.norm(avg_col - baseline) / (np.linalg.norm(baseline) + 1e-6)
        if diff > threshold:
            print(f"[{frame_count}] dev {diff:.3f} > {threshold}, skipping")
            # still push it so your average will “catch up” as camera moves
            avg_buffer.append(avg_col)
            frame_count += 1
            continue

    # otherwise accept this frame
    avg_buffer.append(avg_col)


    # draw legend (fast)
    for i, lid in enumerate(np.unique(seg)):
        y = 30*i + 20
        c = palette[lid].tolist()
        cv2.rectangle(overlay, (5,y-15), (25,y+5), c, -1)
        cv2.putText(overlay,
                    f"{lid}:{model.config.id2label[lid]}",
                    (30,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)

    # write out
    cv2.imwrite(f"{output_fol}/semantic_{frame_count}.png", overlay)

    frame_count += 1
    end = time.time()
    print(f"Frame {frame_count} processed in {end - start:.3f} seconds")

cap.release()
