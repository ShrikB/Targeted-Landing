import cv2
import numpy as np
import os
from pathlib import Path

# Define input and output paths
input_folder = "Targeted-Landing/outputs/test_batch/"
output_folder = os.path.join(input_folder, "masked_merged")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define safe and unsafe classes with their RGB colors
# Safe classes (will be colored white in output)
safe_classes = [
    [159, 66, 133],  # Purple for sidewalk: #9F4285
    [38, 127, 102],  # Medium green for road: #267F66
]

# Unsafe classes (will be colored red in output)
unsafe_classes = [
    [93, 220, 53],   # Green car: #5DDC35
]

# Convert to numpy arrays for efficient computation
safe_classes = [np.array(color) for color in safe_classes]
unsafe_classes = [np.array(color) for color in unsafe_classes]

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
    
    # Create safe mask by combining all safe class colors
    safe_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for safe_color in safe_classes:
        color_mask = np.all(mask_rgb == safe_color, axis=2)
        safe_mask = np.logical_or(safe_mask, color_mask)
    
    # Create unsafe mask by combining all unsafe class colors
    unsafe_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for unsafe_color in unsafe_classes:
        color_mask = np.all(mask_rgb == unsafe_color, axis=2)
        unsafe_mask = np.logical_or(unsafe_mask, color_mask)
    
    # Create output image - start with all black
    output = np.zeros_like(mask_rgb)
    
    # Set the safe areas to white
    output[safe_mask] = [255, 255, 255]
    
    # Set the unsafe areas to red
    output[unsafe_mask] = [255, 0, 0]
    
    # Combine both masks into one final mask (contains both safe and unsafe regions)
    combined_mask = np.logical_or(safe_mask, unsafe_mask)
    
    # The output image now contains:
    # - White pixels for safe areas (defined in safe_classes)
    # - Red pixels for unsafe areas (defined in unsafe_classes)
    # - Black pixels for all other areas
    
    # Generate output filename
    name_without_ext = os.path.splitext(filename)[0]
    output_filename = f"{name_without_ext}_modified.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the result
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    
    print(f"Processed {filename} -> {output_filename}")

print(f"All files processed and saved to: {output_folder}")