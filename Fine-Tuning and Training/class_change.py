import numpy as np
from PIL import Image
import os

def create_color_mapping():
    """
    Create mapping from finetuned model colors to UAVid colors
    
    New Model IDs (from finetuning) -> Modified UAVid IDs
    """
    # New model colors -> Modified UAVid colors mapping
    color_mapping = {
        # Background -> Objects
        (7, 107, 47): (159, 66, 133),
        
        # Trees -> Trees
        (226, 211, 49): (0, 128, 0),
        
        # Road -> Roads
        (38, 127, 102): (128, 64, 128),
        
        # Walkway -> Walkway
        (159, 66, 133): (159, 66, 133),
        
        # Car -> Car
        (93, 220, 53): (192, 0, 192),
        
        # Building -> Buildings
        (236, 204, 209): (128, 0, 0),
        
        # Objects -> Objects
        (116, 112, 14): (64, 64, 0),
        
        # Grass -> Grass
        (13, 66, 152): (0, 128, 0),
        
        # Person -> Objects
        (215, 249, 201): (64, 64, 0),
    }
    
    return color_mapping

def convert_finetuned_to_uavid(input_image_path, output_image_path):
    """
    Convert finetuned model inference colors to Modified UAVid colors
    """
    # Load image
    img = Image.open(input_image_path).convert('RGB')
    img_array = np.array(img)
    
    # Create output array
    output_array = np.zeros_like(img_array)
    
    # Get color mapping
    color_mapping = create_color_mapping()
    
    # Convert colors
    for model_color, uavid_color in color_mapping.items():
        # Find pixels matching model color
        mask = np.all(img_array == model_color, axis=2)
        # Set corresponding UAVid color
        output_array[mask] = uavid_color
    
    # Handle unmapped pixels (set to black or leave as is)
    unmapped_mask = np.all(output_array == [0, 0, 0], axis=2) & ~np.all(img_array == [0, 0, 0], axis=2)
    if np.any(unmapped_mask):
        print(f"Warning: Found {np.sum(unmapped_mask)} unmapped pixels")
        # Keep original colors for unmapped pixels or set to specific color
        output_array[unmapped_mask] = img_array[unmapped_mask]
    
    # Save converted image
    output_img = Image.fromarray(output_array.astype(np.uint8))
    output_img.save(output_image_path)
    
    return True

def batch_convert_folder(input_folder, output_folder):
    """
    Convert all images in a folder from finetuned model colors to Modified UAVid colors
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Converting {len(image_files)} images...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"uavid_{image_file}")
        
        try:
            convert_finetuned_to_uavid(input_path, output_path)
            successful += 1
            print(f"✓ Converted: {image_file}")
        except Exception as e:
            failed += 1
            print(f"✗ Failed to convert {image_file}: {e}")
    
    print("="*80)
    print(f"Conversion completed: {successful}/{len(image_files)} images successful, {failed} failed")

if __name__ == "__main__":
    # Example usage
    input_folder = "new comparisons/uavid_test_inference_model_22"  # Finetuned model inference results
    output_folder = "new comparisons/uavid convert mod22"         # Converted to Modified UAVid format

    batch_convert_folder(input_folder, output_folder)
    
