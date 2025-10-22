from OneFormer_Inference_Video import process_video_with_oneformer
from Mask_Merge import process_semantic_masks
import os


if __name__ == "__main__":
    #model_path = "model/model7_cusdat"
    #output_folder = "outputs/my_results/"
    #video_path = "inputs/gurt.mp4"

    # Define your classes
    safe_classes = [[159, 66, 133], [38, 127, 102]]  # Purple sidewalk, green road
    unsafe_classes = [[93, 220, 53]]  # Green car
    input_folder = "/home/avl-shrek/Documents/Projects/Targeted-Landing/outputs/test_batch/"
    output_folder = os.path.join(input_folder, "masked_merged2")

    # Process masks
    count = process_semantic_masks(
        input_folder=input_folder,
        output_folder=output_folder,
    safe_classes=safe_classes,
    unsafe_classes=unsafe_classes
    )

# Process video with default settings - pass
#frames_processed = process_video_with_oneformer(model_path, output_folder, video_path)

# Process video with custom settings - pass
    """frames_processed = process_video_with_oneformer(
        model_path=model_path,
        output_folder=output_folder,
        video_input=video_path,
        window_size=20,
        threshold=0.10,
        frame_resolution=(512, 512)
    )
    print(f"Total frames processed: {frames_processed}")"""