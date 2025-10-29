from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import numpy as np
import torch
import cv2
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from PIL import Image
import matplotlib
matplotlib.use('Agg')

# Export the function for importing
__all__ = ['process_image_with_oneformer']

def process_image_with_oneformer(model_path, output_folder, image_input, frame_resolution=(512, 512)):
    """
    Process an image with OneFormer semantic segmentation model.
    
    Args:
        model_path (str): Path to the OneFormer model directory
        output_folder (str): Output folder for processed image
        image_input (str): Path to input image file
        frame_resolution (tuple, optional): Resolution to resize image to. Defaults to (512, 512).
    
    Returns:
        bool: True if processing successful, False otherwise
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

    # Load and resize image
    try:
        image = Image.open(image_input)
        # Convert PIL to OpenCV format for resizing
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        imgr = cv2.resize(img_cv, frame_resolution)
        # Convert back to PIL for processor
        imgr_pil = Image.fromarray(cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"❌ Could not load image {image_input}: {e}")
        return False

    print(f"Processing image: {image_input}")
    print(f"Model: {model_path}")
    print(f"Output: {output_folder}")
    print(f"Image resolution: {frame_resolution}")

    start = time.time()

    try:
        # Prepare inputs
        inputs = processor(images=imgr_pil, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Run inference
        with torch.cuda.amp.autocast(), torch.no_grad():
            outputs = model(**inputs)

        # Post-process segmentation
        seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[imgr.shape[:2]])[0].cpu().numpy().astype(np.uint8)
        
        # Pre-generate a color palette (same as video version)
        np.random.seed(1234)
        palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
        
        # Colorize & overlay
        color_mask = palette[seg]
        overlay = cv2.addWeighted(imgr, 0.0, color_mask, 1.0, 0)

        # Draw legend (optional - uncommented version)
        """
        for i, lid in enumerate(np.unique(seg)):
            y = 30 * i + 20
            c = palette[lid].tolist()
            cv2.rectangle(overlay, (5, y - 15), (25, y + 5), c, -1)
            cv2.putText(overlay,
                       f"{lid}:{model.config.id2label[lid]}",
                       (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        """

        # Generate output filename
        image_name = os.path.splitext(os.path.basename(image_input))[0]
        output_path = os.path.join(output_folder, f"semantic_{image_name}.png")
        
        # Save image
        cv2.imwrite(output_path, overlay)

        end = time.time()
        print(f"Image processed in {end - start:.3f} seconds")
        print(f"Output saved to: {output_path}")
        
        return True

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False

# Example usage (can be removed when importing)
if __name__ == "__main__":
    model_path = "model/model7_cusdat"
    output_folder = "outputs/single_image/"
    image_input = "inputs/frame_2530.png"
    
    success = process_image_with_oneformer(model_path, output_folder, image_input)
    if success:
        print("✅ Image processing completed successfully")
    else:
        print("❌ Image processing failed")
