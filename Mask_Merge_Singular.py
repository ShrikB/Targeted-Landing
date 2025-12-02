import cv2
import numpy as np
import os
from pathlib import Path

def process_single_semantic_mask(input_image_path, output_folder, safe_classes, unsafe_classes, potential_classes):
    """
    Process a single semantic segmentation mask to create safe/unsafe/potential landing area mask.
    
    Args:
        input_image_path (str): Path to input PNG mask image
        output_folder (str): Path to folder where processed mask will be saved
        safe_classes (list): List of RGB color values for safe landing areas
        unsafe_classes (list): List of RGB color values for unsafe areas
        potential_classes (list): List of RGB color values for potential areas
    
    Returns:
        str or None: Path to output file if successful, None if failed
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert color lists to numpy arrays for efficient computation
    safe_classes = [np.array(color) for color in safe_classes]
    unsafe_classes = [np.array(color) for color in unsafe_classes]
    potential_classes = [np.array(color) for color in potential_classes]
    
    # Check if input file exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image does not exist: {input_image_path}")
        return None
    
    # Load the semantic segmentation image
    mask = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    
    if mask is None:
        print(f"Warning: Could not load {input_image_path}")
        return None
    
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
    
    # Create potential mask by combining all potential class colors
    potential_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for potential_color in potential_classes:
        color_mask = np.all(mask_rgb == potential_color, axis=2)
        potential_mask = np.logical_or(potential_mask, color_mask)
    
    # Create output image - start with all black
    output = np.zeros_like(mask_rgb)
    
    # Set the safe areas to white
    output[safe_mask] = [255, 255, 255]
    
    # Set the potential areas to grey
    output[potential_mask] = [128, 128, 128]
    
    # Process red objects (unsafe areas) and create buffer zones
    red_objects = []
    red_mask_uint8 = unsafe_mask.astype(np.uint8) * 255
    
    if np.sum(red_mask_uint8) > 0:
        # Find contours of red objects
        contours, _ = cv2.findContours(red_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate area to filter out very small noise
            area = cv2.contourArea(contour)
            if area < 10:  # Skip very small contours
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
            else:
                continue
            
            # Create mask for this specific object
            object_mask = np.zeros_like(red_mask_uint8)
            cv2.drawContours(object_mask, [contour], -1, 255, -1)
            
            # Use distance transform to find maximum distance from centroid to edge
            dist_transform_obj = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
            max_distance = dist_transform_obj.max()
            
            # Add 20 pixel buffer
            buffer_radius = max_distance + 20
            
            # Draw the buffer zone as red in the output (expanding the red area)
            cv2.circle(output, (centroid_x, centroid_y), int(buffer_radius), [255, 0, 0], -1)
            
            red_objects.append({
                'centroid': (centroid_x, centroid_y),
                'max_distance': max_distance,
                'buffer_radius': buffer_radius
            })
    
    # Set the original unsafe areas to red (to ensure they're visible)
    output[unsafe_mask] = [255, 0, 0]
    
    # Check if any red buffer zones touch grey areas
    grey_is_safe = True
    if np.sum(potential_mask) > 0:
        if len(red_objects) > 0:
            # Create a temporary mask with all red buffer zones
            red_buffer_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for red_obj in red_objects:
                centroid = red_obj['centroid']
                buffer_radius = red_obj['buffer_radius']
                cv2.circle(red_buffer_mask, centroid, int(buffer_radius), 255, -1)
            
            # Check if red buffer zones overlap with grey areas
            grey_mask_uint8 = potential_mask.astype(np.uint8) * 255
            overlap = cv2.bitwise_and(red_buffer_mask, grey_mask_uint8)
            
            if np.sum(overlap) > 0:
                grey_is_safe = False
                # Convert all grey areas to black
                output[potential_mask] = [0, 0, 0]
                print(f"  Red buffer zones detected in grey areas - converting all grey to black")
        else:
            # No red objects exist, so grey areas are safe - convert to white
            output[potential_mask] = [255, 255, 255]
            print(f"  No red objects detected - converting all grey to white (safe)")
    
    # The output image now contains:
    # - White pixels for safe areas (defined in safe_classes)
    # - Red pixels for unsafe areas with buffer zones (defined in unsafe_classes + 20px buffer)
    # - Grey pixels for potential areas (only if no red buffer zones touch them)
    # - Black pixels for all other areas and unsafe grey regions
    
    print(f"  Found {len(red_objects)} red objects with buffer zones")
    print(f"  Grey areas status: {'SAFE' if grey_is_safe else 'UNSAFE (converted to black)'}")
    
    # Generate output filename
    filename = os.path.basename(input_image_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_filename = f"{name_without_ext}_modified.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the result
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    
    print(f"Processed {filename} -> {output_filename}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Define input and output paths
    #input_image = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/single_image/semantic_frame_2529.png"
    input_image = "/home/avl-shrek/Documents/Projects/Targeted-Landing/inputs/semantic_frame_000246.png"
    output_folder = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/masked_merged_single"
    
    # Define safe and unsafe classes with their RGB colors
    # Safe classes (will be colored white in output)
    safe_classes = [
        [159, 66, 133]  # Purple for sidewalk: #9F4285

    ]
    
    # Unsafe classes (will be colored red in output)
    unsafe_classes = [
        [93, 220, 53]   # Green car: #5DDC35
    ]
    
    # Potential classes (will be colored grey in output)
    potential_classes = [
        [38, 127, 102]  # Medium green for road: #267F66
    ]
    
    # Process the single mask
    result_path = process_single_semantic_mask(input_image, output_folder, safe_classes, unsafe_classes, potential_classes)
    
    if result_path:
        print(f"Processing successful! Output saved to: {result_path}")
    else:
        print("Processing failed!")