import cv2
import numpy as np
import os
from pathlib import Path

def process_semantic_masks(input_folder, output_folder, safe_classes, unsafe_classes):
    """
    Process semantic segmentation masks to create safe/unsafe landing area masks.
    
    Args:
        input_folder (str): Path to folder containing input PNG mask images
        output_folder (str): Path to folder where processed masks will be saved
        safe_classes (list): List of RGB color values for safe landing areas
        unsafe_classes (list): List of RGB color values for unsafe areas
    
    Returns:
        int: Number of images processed successfully
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert color lists to numpy arrays for efficient computation
    safe_classes = [np.array(color) for color in safe_classes]
    unsafe_classes = [np.array(color) for color in unsafe_classes]
    
    # Debug: Print the input folder path
    print(f"Looking for PNG files in: {input_folder}")
    print(f"Folder exists: {os.path.exists(input_folder)}")
    
    # Get all PNG files in the input folder
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        return 0
    
    try:
        all_files = os.listdir(input_folder)
        print(f"All files in folder: {all_files[:5]}...")  # Show first 5 files
        png_files = [f for f in all_files if f.lower().endswith('.png')]
    except Exception as e:
        print(f"Error reading folder: {e}")
        return 0

    print(f"Found {len(png_files)} PNG files to process")
    
    processed_count = 0
    
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
        processed_count += 1
    
    print(f"All {processed_count} files processed and saved to: {output_folder}")
    return processed_count


# Example usage
if __name__ == "__main__":
    # Define input and output paths
    input_folder = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/"
    output_folder = os.path.join(input_folder, "masked_merged2")
    
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
    
    # Process the masks
    processed_count = process_semantic_masks(input_folder, output_folder, safe_classes, unsafe_classes)