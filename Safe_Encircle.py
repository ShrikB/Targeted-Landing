import os
import cv2
import numpy as np
import glob

# Configuration - Updated to use mask_merged output
input_folder = "/home/shrekfedora/Projects/Targeted-Landing/mask outputs/test data folder/masked_merged/"
output_folder = "/home/shrekfedora/Projects/Targeted-Landing/mask outputs/test data folder/encircled/"
min_diameter_pixels = 30  # Half a meter = 30 pixels
min_radius_pixels = min_diameter_pixels / 2

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_semantic_image(img_path, output_path):
    """
    Process a single semantic image to find the largest safe landing circle
    
    Args:
        img_path: Path to input semantic image
        output_path: Path to save output image with circle
    
    Returns:
        dict: Results including center, radius, and whether it meets minimum size
    """
    # Read the PNG -> NumPy array (BGR)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return None

    # Copy for output visualization
    out = img.copy()

    # Build the white mask (white areas represent safe landing area from mask merge)
    # Since mask merge outputs white for safe areas and black for unsafe
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    road_mask = (gray > 200).astype(np.uint8) * 255  # White areas

    # Check if there's any road area detected
    if np.sum(road_mask) == 0:
        print(f"No landing area detected in {os.path.basename(img_path)}")
        # Save image with no circle
        cv2.imwrite(output_path, out)
        return {
            'center': None,
            'radius': 0,
            'meets_minimum': False,
            'landing_area_pixels': 0
        }

    # Distance transform to find largest inscribed circle
    dist = cv2.distanceTransform(road_mask, cv2.DIST_L2, 5)
    max_radius = dist.max()
    cy, cx = np.unravel_index(dist.argmax(), dist.shape)

    # Check if circle meets minimum size requirement
    meets_minimum = max_radius >= min_radius_pixels
    
    # Choose circle color based on size requirement
    circle_color = (0, 255, 0) if meets_minimum else (0, 0, 255)  # Green if good, red if too small
    
    # Draw circle and center marker
    cv2.circle(out, (cx, cy), int(max_radius), circle_color, 2)
    cv2.drawMarker(out, (cx, cy), circle_color, cv2.MARKER_CROSS,
                   markerSize=12, thickness=2)
    
    # Add text showing radius and status
    status_text = f"R={max_radius:.1f}px"
    if meets_minimum:
        status_text += " SAFE"
    else:
        status_text += f" TOO SMALL (min={min_radius_pixels}px)"
    
    cv2.putText(out, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, circle_color, 2)

    # Save the result
    cv2.imwrite(output_path, out)
    
    return {
        'center': (cx, cy),
        'radius': float(max_radius),
        'diameter': float(max_radius * 2),
        'meets_minimum': meets_minimum,
        'landing_area_pixels': int(np.sum(road_mask) / 255)
    }

def main():
    # Find all modified PNG files in the masked_merged folder
    pattern = os.path.join(input_folder, "*_modified.png")
    img_files = glob.glob(pattern)
    
    if not img_files:
        print(f"No modified images found in {input_folder}")
        print(f"Looking for pattern: {pattern}")
        return
    
    # Sort files by name
    img_files.sort()
    
    print(f"Found {len(img_files)} modified images to process")
    print(f"Minimum safe diameter: {min_diameter_pixels} pixels (0.5 meters)")
    
    results = []
    safe_count = 0
    
    # Process each image
    for img_path in img_files:
        filename = os.path.basename(img_path)
        # Replace "_modified" with "_encircled" for output filename
        output_filename = filename.replace("_modified.png", "_encircled.png")
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Processing {filename}...", end=" ")
        
        result = process_semantic_image(img_path, output_path)
        
        if result is None:
            print("FAILED")
            continue
            
        result['filename'] = filename
        results.append(result)
        
        if result['meets_minimum']:
            safe_count += 1
            print(f"SAFE - Center: {result['center']}, Radius: {result['radius']:.1f}px")
        else:
            if result['center'] is None:
                print("NO LANDing AREA")
            else:
                print(f"TOO SMALL - Radius: {result['radius']:.1f}px < {min_radius_pixels}px")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total images processed: {len(results)}")
    print(f"Safe landing areas: {safe_count}")
    print(f"Unsafe/too small: {len(results) - safe_count}")
    print(f"Success rate: {safe_count/len(results)*100:.1f}%")
    
    # Find best landing spot
    safe_results = [r for r in results if r['meets_minimum']]
    if safe_results:
        best = max(safe_results, key=lambda x: x['radius'])
        print(f"\nBest landing spot: {best['filename']}")
        print(f"  Center: {best['center']}")
        print(f"  Radius: {best['radius']:.1f}px ({best['diameter']:.1f}px diameter)")
        print(f"  Estimated size: {best['diameter']/60:.2f} meters diameter")
    
    print(f"\nResults saved to: {output_folder}")

if __name__ == "__main__":
    main()