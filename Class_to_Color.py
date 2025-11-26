import cv2
import numpy as np
import os

# Path to semantic segmentation image (output from OneFormer)
image_path = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/frame_pipeline/semantic_output/semantic_frame_000066.png"

# Check if input file exists
if not os.path.exists(image_path):
    print(f"Error: Input image does not exist: {image_path}")
    exit()

# Load the semantic segmentation image
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img is None:
    print(f"Error: Could not load image {image_path}")
    exit()

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"\n{'='*70}")
print(f"Analyzing Semantic Segmentation Image")
print(f"{'='*70}")
print(f"Image: {os.path.basename(image_path)}")
print(f"Resolution: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
print(f"Total pixels: {img_rgb.shape[0] * img_rgb.shape[1]:,}")

# Find all unique colors in the image
pixels = img_rgb.reshape(-1, img_rgb.shape[-1])
unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

# Sort by pixel count (most common first)
sorted_indices = np.argsort(counts)[::-1]
unique_colors = unique_colors[sorted_indices]
counts = counts[sorted_indices]

print(f"\n{'='*70}")
print(f"Found {len(unique_colors)} unique colors (classes) in the image")
print(f"{'='*70}")

# Display each unique color
print(f"\n{'#':<4} {'RGB Color':<20} {'Hex':<10} {'Pixels':<12} {'Percentage':<10}")
print(f"{'-'*70}")

total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

for idx, (color, count) in enumerate(zip(unique_colors, counts), 1):
    rgb_tuple = tuple(color)
    
    # Convert to hex
    hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    
    # Calculate percentage
    percentage = (count / total_pixels) * 100
    
    # Print color information
    print(f"{idx:<4} {str(rgb_tuple):<20} {hex_color:<10} {count:<12,} {percentage:>6.2f}%")

print(f"{'-'*70}")
print(f"\nTotal unique colors: {len(unique_colors)}")

# Color array format for use in code
print(f"\n{'='*70}")
print("Color Array Format (for use in code):")
print(f"{'='*70}")
for idx, color in enumerate(unique_colors, 1):
    print(f"  [{color[0]}, {color[1]}, {color[2]}],  # Color {idx}")

print(f"{'='*70}\n")
