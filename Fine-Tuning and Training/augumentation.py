import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

# --- Input and Output Configuration ---
# Assumes images and masks have the same filenames
image_input_folder = "semantic_drone_dataset\Dataset2\Raw"
mask_input_folder = "semantic_drone_dataset\Dataset2\Mask"
image_output_folder = "semantic_drone_dataset\Dataset2\Augmented_Raw"
mask_output_folder = "semantic_drone_dataset\Dataset2\Augmented_Mask"
num_augmentations_per_image = 5 # How many augmented versions to create for each original

# --- Create output folders if they don't exist ---
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(mask_output_folder, exist_ok=True)

# --- Define Augmentation Function ---
def augment_image_and_mask(image, mask):
    """
    Applies the same random augmentations to both an image and its mask.
    """
    # 1. Random Resize and Crop
    # Get parameters for the crop once
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.8, 1.0), ratio=(0.75, 1.33)
    )
    # Apply the same crop to both
    image = TF.resized_crop(image, i, j, h, w, size=(1080, 1920), interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resized_crop(mask, i, j, h, w, size=(1080, 1920), interpolation=TF.InterpolationMode.NEAREST)

    # 2. Random Horizontal Flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # 3. Random Rotation
    if random.random() > 0.5:
        angle = transforms.RandomRotation.get_params(degrees=(-30, 30))
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

    # 4. Color Jitter (Applied ONLY to the image)
    if random.random() > 0.5:
        image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)

    return image, mask

# --- Main Processing Loop ---
image_filenames = [f for f in os.listdir(image_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_filenames)} images to augment.")

for filename in image_filenames:
    image_path = os.path.join(image_input_folder, filename)
    mask_path = os.path.join(mask_input_folder, filename) # Assumes same filename

    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {filename}, skipping.")
        continue

    with Image.open(image_path).convert("RGB") as img, Image.open(mask_path) as msk:
        for i in range(num_augmentations_per_image):
            augmented_img, augmented_msk = augment_image_and_mask(img, msk)

            # Define output filenames
            base_name, ext = os.path.splitext(filename)
            output_image_path = os.path.join(image_output_folder, f"{base_name}_aug_{i}{ext}")
            output_mask_path = os.path.join(mask_output_folder, f"{base_name}_aug_{i}{ext}")

            # Save the augmented files
            augmented_img.save(output_image_path)
            augmented_msk.save(output_mask_path)

print("Image and mask augmentation completed.")