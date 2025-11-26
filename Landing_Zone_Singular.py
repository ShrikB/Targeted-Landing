import cv2
import numpy as np
import os

def find_single_landing_zone(input_image_path, output_folder, drone_size=15):
    """
    Process a single PNG mask to find the largest circle that can fit in white (safe) regions.
    
    Args:
        input_image_path (str): Path to input PNG mask image with four colors:
                               - Black: background/obstacles
                               - White: safe landing regions  
                               - Red: unsafe regions
                               - Grey: potential regions (safe if no red objects present)
        output_folder (str): Path to folder where output image with circle will be saved
        drone_size (int): Diameter of the drone in pixels. Default is 15px.
    
    Returns:
        dict or None: Dictionary containing results for the processed image, None if failed
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate minimum radius from drone diameter
    min_radius = drone_size / 2
    
    # Check if input file exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image does not exist: {input_image_path}")
        return None
    
    filename = os.path.basename(input_image_path)
    
    # Load the image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Could not load {filename}")
        return None
    
    # Convert to RGB for easier color detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create mask for white regions (safe areas) - exact color match
    white_mask = (img_rgb == [255, 255, 255]).all(axis=2).astype(np.uint8) * 255
    
    # Create mask for red regions (unsafe objects) - exact color match
    red_mask = (img_rgb == [255, 0, 0]).all(axis=2).astype(np.uint8) * 255
    
    # Check if there are any white pixels
    if np.sum(white_mask) == 0:
        return None
    
    # Use Euclidean Distance Transform to find largest inscribed circle in white areas
    # Red buffer zones are already applied in the Mask_Merge step
    dist_transform = cv2.distanceTransform(white_mask, cv2.DIST_L2, 5)
    
    # Find the maximum distance (radius of largest circle)
    max_radius = dist_transform.max()
    
    # Check if the landing zone meets minimum drone size requirement
    if max_radius < min_radius:
        print(f"No safe landing zone available in {filename} - max radius {max_radius:.1f}px")
        # Save original image without any modifications
        output_filename = f"{os.path.splitext(filename)[0]}_landing_zone.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, img)
        return None
    
    # Create output image (copy of original for visualization)
    output_img = img.copy()
    
    # Find the center of the largest circle
    max_loc = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
    center_y, center_x = max_loc
    
    # Calculate frame center
    frame_height, frame_width = img.shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # Calculate vector from circle center to frame center
    vector_x = frame_center_x - center_x
    vector_y = frame_center_y - center_y
    
    # Draw the largest circle on the output image
    circle_color = (0, 255, 0)  # Green
    circle_thickness = 1
    
    cv2.circle(output_img, (center_x, center_y), int(max_radius), circle_color, circle_thickness)
    
    # Draw center point
    cv2.drawMarker(output_img, (center_x, center_y), circle_color, 
                  cv2.MARKER_CROSS, markerSize=10, thickness=2)
    
    # ===== REMOVABLE SECTION: Vector Visualization =====
    # Draw frame center
    frame_center_color = (255, 0, 0)  # Blue
    cv2.drawMarker(output_img, (frame_center_x, frame_center_y), frame_center_color, 
                  cv2.MARKER_SQUARE, markerSize=8, thickness=2)
    
    # Draw vector line from circle center to frame center
    cv2.arrowedLine(output_img, (center_x, center_y), (frame_center_x, frame_center_y), 
                   frame_center_color, thickness=2, tipLength=0.1)
    
    # Add vector text
    vector_text = f"Vector: ({vector_x}, {vector_y})"
    cv2.putText(output_img, vector_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, frame_center_color, 2)
    # ===== END REMOVABLE SECTION =====
    
    # Add text showing radius
    text = f"R={max_radius:.1f}px"
    cv2.putText(output_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, circle_color, 2)
    
    # Save the output image
    output_filename = f"{os.path.splitext(filename)[0]}_landing_zone.png"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, output_img)
    
    # Print only the essential information
    diameter = max_radius * 2
    print(f"{filename}, Diameter: {diameter:.1f}px, Center: {center_x}x{center_y}, Vector to frame center: ({vector_x}, {vector_y})")
    
    result = {
        'filename': filename,
        'center': (center_x, center_y),
        'frame_center': (frame_center_x, frame_center_y),
        'vector_to_frame_center': (vector_x, vector_y),
        'radius': float(max_radius),
        'diameter': float(diameter),
        'meets_minimum': max_radius >= min_radius,
        'drone_size': drone_size,
        'min_radius_required': min_radius,
        'output_path': output_path
    }
    
    return result


# Example usage
if __name__ == "__main__":
    input_image = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/masked_merged_single/semantic_frame_000246_modified.png"
    output_folder = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/landing_zones_single"

    result = find_single_landing_zone(input_image, output_folder, drone_size=15)  # 15px diameter drone

    if result:
        print(f"Landing zone processing successful!")
        print(f"   Center: ({result['center'][0]}, {result['center'][1]})")
        print(f"   Vector to frame center: {result['vector_to_frame_center']}")
        print(f"   Meets drone size requirement: {result['meets_minimum']}")
        print(f"   Output saved to: {result['output_path']}")
    else:
        print("Landing zone processing failed!")