from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Define paths
img_path = "inputs/IMG_1740.JPG"
model_path = "model/model8/"

# Load processor and model
processor = OneFormerProcessor.from_pretrained(model_path)
model = OneFormerForUniversalSegmentation.from_pretrained(model_path)

model.eval()
model.model.is_training = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open(img_path)
inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


for k,v in inputs.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)

# forward pass (no need for gradients at inference time)
with torch.no_grad():
  outputs = model(**inputs)

# postprocessing

semantic_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
semantic_segmentation = np.array(semantic_segmentation)
semantic_segmentation = semantic_segmentation.astype(np.uint8)
cv2.imwrite("outputs/semantic_segmentation.png", semantic_segmentation)
#semantic_segmentation.shape
