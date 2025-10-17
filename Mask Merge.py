import cv2
import numpy as np
import os
from pathlib import Path

# Define input and output paths
input_folder = "/home/shrekfedora/Projects/Targeted-Landing/mask outputs/test data folder/"
output_folder = os.path.join(input_folder, "masked_merged")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the exact target colors from hex values
# Purple for sidewalk: #9F4285 -> RGB(159, 66, 133)
# Medium green for road: #267F66 -> RGB(38, 127, 102)
# Green car: #5DDC35 -> RGB(93, 220, 53)
purple_color = np.array([159, 66, 133])
green_color = np.array([38, 127, 102])
car_color = np.array([93, 220, 53])

# Get all PNG files in the input folder
png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

print(f"Found {len(png_files)} PNG files to process")

# Process each semantic image
for filename in png_files:
    # Load the semantic segmentation image
    image_path = os.path.join(input_folder, filename)
    mask = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if mask is None:
        print(f"Warning: Could not load {filename}")
        continue
    
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Create masks for exact color matches
    purple_mask = np.all(mask_rgb == purple_color, axis=2)
    green_mask = np.all(mask_rgb == green_color, axis=2)
    car_mask = np.all(mask_rgb == car_color, axis=2)
    
    # Combine the sidewalk and road masks for white areas
    safe_landing_mask = np.logical_or(purple_mask, green_mask)
    
    # Create output image - start with all black
    output = np.zeros_like(mask_rgb)
    
    # Set the safe landing areas (sidewalk + road) to white
    output[safe_landing_mask] = [255, 255, 255]
    
    # Set the car areas to red
    output[car_mask] = [255, 0, 0]
    
    # Generate output filename
    name_without_ext = os.path.splitext(filename)[0]
    output_filename = f"{name_without_ext}_modified.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the result
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    
    print(f"Processed {filename} -> {output_filename}")

print(f"All files processed and saved to: {output_folder}")