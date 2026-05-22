import numpy as np
from PIL import Image
import os

def apply_color_corrections(input_folder, output_folder):
    """
    Apply specific color corrections to mask images:
    - (0, 0, 0) -> (159, 66, 133) [Black to Walkway]
    - (64, 0, 128) -> (192, 0, 192) [Purple to Car]
    - (128, 128, 0) -> (0, 128, 0) [Yellow to Trees]
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Color corrections mapping
    corrections = {
        (0, 0, 0): (159, 66, 133),      # Black to Walkway
        (64, 0, 128): (192, 0, 192),    # Moving Car to Car
        (128, 128, 0): (0, 128, 0),      # Low Vegetation to Trees
        (64, 64, 0): (116, 112, 14)      # Humans to Objects
    }
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Applying color corrections to {len(image_files)} images...")
    successful = 0
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"corrected_{image_file}")
        
        try:
            # Load image
            img = Image.open(input_path).convert('RGB')
            img_array = np.array(img)
            
            # Apply corrections
            output_array = img_array.copy()
            total_changes = 0
            
            for old_color, new_color in corrections.items():
                # Find pixels matching old color
                mask = np.all(img_array == old_color, axis=2)
                # Set new color
                output_array[mask] = new_color
                
                # Count changes
                pixel_count = np.sum(mask)
                total_changes += pixel_count
                
                if pixel_count > 0:
                    print(f"  {image_file}: Changed {pixel_count} pixels from {old_color} to {new_color}")
            
            # Save corrected image
            output_img = Image.fromarray(output_array.astype(np.uint8))
            output_img.save(output_path)
            
            successful += 1
            if total_changes == 0:
                print(f"Processed: {image_file} (no changes needed)")
            else:
                print(f"Corrected: {image_file} (total: {total_changes} pixels)")
            
        except Exception as e:
            print(f"Failed to correct {image_file}: {e}")
    
    print(f"\nColor correction completed: {successful}/{len(image_files)} images")
    print(f"Results saved to: {output_folder}")

if __name__ == "__main__":
    # Get input folder from user or set default
    input_folder = input("Enter mask folder path (or press Enter for default): ").strip()
    
    if not input_folder:
        input_folder = "outputs/comparision files/convert"  # Default input folder
    
    output_folder = "outputs/comparision files/convert_Adjusted2"  # Output folder
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
    else:
        apply_color_corrections(input_folder, output_folder)
        
        # Print correction mappings for reference
        print("\nApplied Color Corrections:")