from OneFormer_Inference_Video import process_video_with_oneformer

model_path = "model/model7_cusdat"
output_folder = "outputs/my_results/"
video_path = "inputs/gurt.mp4"

# Process video with default settings - pass
#frames_processed = process_video_with_oneformer(model_path, output_folder, video_path)

# Process video with custom settings - pass
frames_processed = process_video_with_oneformer(
    model_path=model_path,
    output_folder=output_folder,
    video_input=video_path,
    window_size=20,
    threshold=0.10,
    frame_resolution=(512, 512)
)
print(f"Total frames processed: {frames_processed}")