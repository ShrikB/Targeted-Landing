from OneFormer_Inference_Video import process_video_with_oneformer
from Mask_Merge import process_semantic_masks
from Landing_Zone import find_landing_zones
import os
import time

def run_complete_pipeline(model_path, video_input, base_output_folder, safe_classes, unsafe_classes, 
                         window_size=20, threshold=0.1, frame_resolution=(512, 512)):
    """
    Complete pipeline for video processing: OneFormer -> Mask Merge -> Landing Zone Detection
    
    Args:
        model_path (str): Path to OneFormer model
        video_input (str): Path to input video
        base_output_folder (str): Base folder for all outputs
        safe_classes (list): RGB colors for safe landing areas
        unsafe_classes (list): RGB colors for unsafe areas
        window_size (int): OneFormer rolling window size
        threshold (float): OneFormer change detection threshold
        frame_resolution (tuple): Video processing resolution
    
    Returns:
        dict: Results from each stage
    """
    print("=" * 60)
    print("STARTING COMPLETE LANDING ZONE DETECTION PIPELINE")
    print("=" * 60)
    
    # Define folder structure
    semantic_output = os.path.join(base_output_folder, "semantic_frames")
    masked_output = os.path.join(base_output_folder, "masked_merged")
    landing_zones_output = os.path.join(base_output_folder, "landing_zones")
    
    # Create base output folder
    os.makedirs(base_output_folder, exist_ok=True)
    
    results = {}
    
    # ========== STAGE 1: OneFormer Video Processing ==========
    print("\n🎥 STAGE 1: Running OneFormer semantic segmentation on video...")
    print(f"Input video: {video_input}")
    print(f"Output folder: {semantic_output}")
    
    stage1_start = time.time()
    try:
        frames_processed = process_video_with_oneformer(
            model_path=model_path,
            output_folder=semantic_output,
            video_input=video_input,
            window_size=window_size,
            threshold=threshold,
            frame_resolution=frame_resolution
        )
        stage1_time = time.time() - stage1_start
        results['stage1'] = {
            'success': True,
            'frames_processed': frames_processed,
            'processing_time': stage1_time,
            'output_folder': semantic_output
        }
        print(f"✅ Stage 1 Complete: {frames_processed} frames processed in {stage1_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Stage 1 Failed: {e}")
        results['stage1'] = {'success': False, 'error': str(e)}
        return results
    
    # ========== STAGE 2: Mask Merging ==========
    print(f"\n🎯 STAGE 2: Processing semantic masks for safe/unsafe areas...")
    print(f"Input folder: {semantic_output}")
    print(f"Output folder: {masked_output}")
    print(f"Safe classes: {safe_classes}")
    print(f"Unsafe classes: {unsafe_classes}")
    
    stage2_start = time.time()
    try:
        masks_processed = process_semantic_masks(
            input_folder=semantic_output,
            output_folder=masked_output,
            safe_classes=safe_classes,
            unsafe_classes=unsafe_classes
        )
        stage2_time = time.time() - stage2_start
        results['stage2'] = {
            'success': True,
            'masks_processed': masks_processed,
            'processing_time': stage2_time,
            'output_folder': masked_output
        }
        print(f"✅ Stage 2 Complete: {masks_processed} masks processed in {stage2_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Stage 2 Failed: {e}")
        results['stage2'] = {'success': False, 'error': str(e)}
        return results
    
    # ========== STAGE 3: Landing Zone Detection ==========
    print(f"\n🎯 STAGE 3: Finding optimal landing zones...")
    print(f"Input folder: {masked_output}")
    print(f"Output folder: {landing_zones_output}")
    
    stage3_start = time.time()
    try:
        landing_results = find_landing_zones(
            input_folder=masked_output,
            output_folder=landing_zones_output
        )
        stage3_time = time.time() - stage3_start
        results['stage3'] = {
            'success': True,
            'landing_zones_found': len(landing_results),
            'processing_time': stage3_time,
            'output_folder': landing_zones_output,
            'results': landing_results
        }
        print(f"✅ Stage 3 Complete: {len(landing_results)} landing zones analyzed in {stage3_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Stage 3 Failed: {e}")
        results['stage3'] = {'success': False, 'error': str(e)}
        return results
    
    # ========== PIPELINE SUMMARY ==========
    total_time = sum([results[f'stage{i}']['processing_time'] for i in range(1, 4) if results[f'stage{i}']['success']])
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Video frames processed: {results['stage1']['frames_processed']}")
    print(f"Masks created: {results['stage2']['masks_processed']}")
    print(f"Landing zones analyzed: {results['stage3']['landing_zones_found']}")
    
    # Find best landing zone
    if results['stage3']['success'] and results['stage3']['results']:
        best_zone = max(results['stage3']['results'], key=lambda x: x['radius'])
        print(f"\n🎯 BEST LANDING ZONE:")
        print(f"   File: {best_zone['filename']}")
        print(f"   Radius: {best_zone['radius']:.1f}px ({best_zone['diameter']:.1f}px diameter)")
        print(f"   Center: ({best_zone['center'][0]}, {best_zone['center'][1]})")
        print(f"   Vector to frame center: {best_zone['vector_to_frame_center']}")
        print(f"   Red objects nearby: {best_zone['red_objects_count']}")
        
        results['best_landing_zone'] = best_zone
    
    print(f"\n📁 OUTPUT FOLDERS:")
    print(f"   Semantic frames: {semantic_output}")
    print(f"   Processed masks: {masked_output}")
    print(f"   Landing zones: {landing_zones_output}")
    
    return results

if __name__ == "__main__":
    # Configuration
    model_path = "model/model7_cusdat"
    video_input = "/home/shrekfedora/Projects/Targeted-Landing/inputs/gurt.mp4"
    base_output_folder = "/home/shrekfedora/Projects/Targeted-Landing/outputs/pipeline_test"
    
    # Define safe and unsafe classes
    safe_classes = [
        [159, 66, 133],  # Purple sidewalk: #9F4285
        [38, 127, 102],  # Medium green road: #267F66
    ]
    
    unsafe_classes = [
        [93, 220, 53],   # Green car: #5DDC35
    ]
    
    # OneFormer processing parameters
    oneformer_params = {
        'window_size': 20,
        'threshold': 0.10,
        'frame_resolution': (512, 512)
    }
    
    # Run the complete pipeline
    results = run_complete_pipeline(
        model_path=model_path,
        video_input=video_input,
        base_output_folder=base_output_folder,
        safe_classes=safe_classes,
        unsafe_classes=unsafe_classes,
        **oneformer_params
    )
    
    # Optional: Print detailed results
    if results.get('best_landing_zone'):
        print(f"\n🚁 RECOMMENDED LANDING COORDINATES:")
        best = results['best_landing_zone']
        print(f"   Navigate to pixel coordinates: ({best['center'][0]}, {best['center'][1]})")
        print(f"   Safe radius: {best['radius']:.1f} pixels")
        estimated_meters = best['diameter'] / 60  # Assuming 60px = 1m
        print(f"   Estimated real-world diameter: {estimated_meters:.2f} meters")