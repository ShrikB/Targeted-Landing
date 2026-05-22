import cv2
import numpy as np
import os
from pathlib import Path
from Mask_Merge_Singular import process_single_semantic_mask

def batch_process_folders(input_folder, output_folder, safe_classes, unsafe_classes, potential_classes, folder_name="processed"):
    """
    Batch process all masks in a folder using the mask merge function.
    
    Args:
        input_folder (str): Path to folder containing input masks
        output_folder (str): Path to base output folder
        safe_classes (list): List of RGB color values for safe landing areas
        unsafe_classes (list): List of RGB color values for unsafe areas
        potential_classes (list): List of RGB color values for potential areas
        folder_name (str): Name for the output subfolder
    
    Returns:
        dict: Statistics about processing
    """
    
    # Create output directory
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Get all PNG files in input folder
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    
    if not input_files:
        print(f"Warning: No PNG files found in {input_folder}")
        return {"processed": 0, "failed": 0, "total": 0}
    
    print(f"\n{'='*80}")
    print(f"Processing {len(input_files)} images from: {input_folder}")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}\n")
    
    # Statistics
    stats = {
        "processed": 0,
        "failed": 0,
        "total": len(input_files)
    }
    
    # Process each file
    for idx, filename in enumerate(input_files, 1):
        input_image_path = os.path.join(input_folder, filename)
        
        print(f"[{idx}/{len(input_files)}] Processing: {filename}")
        
        try:
            result_path = process_single_semantic_mask(
                input_image_path=input_image_path,
                output_folder=output_path,
                safe_classes=safe_classes,
                unsafe_classes=unsafe_classes,
                potential_classes=potential_classes
            )
            
            if result_path:
                stats["processed"] += 1
            else:
                stats["failed"] += 1
                print(f"  ✗ Failed to process {filename}")
                
        except Exception as e:
            stats["failed"] += 1
            print(f"  ✗ Error processing {filename}: {e}")
    
    return stats

def main():
    # ========== CONFIGURATION ==========
    
    # Input folders
    gt_folder = "Labels_Adjusted"  # Ground truth masks
    pred_folder = "uavid_format"    # Prediction/inference masks
    
    # Output base folder
    output_base = "processed_masks"
    
    # Define safe, unsafe, and potential classes with their RGB colors
    # These should match the colors in your semantic segmentation output
    
    # Safe classes (will be colored white in output)
    safe_classes = [
        [159, 66, 133]  # Purple sidewalk: #9F4285
    ]
    
    # Unsafe classes (will be colored red in output with buffer zones)
    unsafe_classes = [
        [93, 220, 53]   # Green car: #5DDC35
    ]
    
    # Potential classes (will be colored grey in output, or white if safe)
    potential_classes = [
        [38, 127, 102]  # Medium green road: #267F66
    ]
    
    # ========== PROCESS GROUND TRUTH MASKS ==========
    print("\n" + "="*80)
    print("STEP 1: Processing Ground Truth Masks")
    print("="*80)
    
    gt_stats = batch_process_folders(
        input_folder=gt_folder,
        output_folder=output_base,
        safe_classes=safe_classes,
        unsafe_classes=unsafe_classes,
        potential_classes=potential_classes,
        folder_name="ground_truth_processed"
    )
    
    # ========== PROCESS PREDICTION MASKS ==========
    print("\n" + "="*80)
    print("STEP 2: Processing Prediction Masks")
    print("="*80)
    
    pred_stats = batch_process_folders(
        input_folder=pred_folder,
        output_folder=output_base,
        safe_classes=safe_classes,
        unsafe_classes=unsafe_classes,
        potential_classes=potential_classes,
        folder_name="predictions_processed"
    )
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\nGround Truth Masks ({gt_folder}):")
    print(f"  Total files:      {gt_stats['total']}")
    print(f"  Successfully processed: {gt_stats['processed']}")
    print(f"  Failed:           {gt_stats['failed']}")
    print(f"  Success rate:     {gt_stats['processed']/gt_stats['total']*100 if gt_stats['total'] > 0 else 0:.1f}%")
    
    print(f"\nPrediction Masks ({pred_folder}):")
    print(f"  Total files:      {pred_stats['total']}")
    print(f"  Successfully processed: {pred_stats['processed']}")
    print(f"  Failed:           {pred_stats['failed']}")
    print(f"  Success rate:     {pred_stats['processed']/pred_stats['total']*100 if pred_stats['total'] > 0 else 0:.1f}%")
    
    print(f"\nOutput Folders:")
    print(f"  Ground Truth: {os.path.join(output_base, 'ground_truth_processed')}")
    print(f"  Predictions:  {os.path.join(output_base, 'predictions_processed')}")
    
    print("\n" + "="*80)
    
    # ========== VERIFICATION ==========
    gt_output = os.path.join(output_base, "ground_truth_processed")
    pred_output = os.path.join(output_base, "predictions_processed")
    
    if os.path.exists(gt_output):
        gt_output_files = len([f for f in os.listdir(gt_output) if f.endswith('.png')])
        print(f"\nVerification - Ground Truth output files: {gt_output_files}")
    
    if os.path.exists(pred_output):
        pred_output_files = len([f for f in os.listdir(pred_output) if f.endswith('.png')])
        print(f"Verification - Predictions output files: {pred_output_files}")
    
    print("\n✓ Batch processing complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()