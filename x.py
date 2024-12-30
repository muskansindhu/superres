import argparse
import sieve
import shutil
import os
import subprocess
import cv2  
from gfpgan import GFPGANer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Restore musetalk video quality using GFPGAN/CodeFormer')
    parser.add_argument('--superres', choices=['GFPGAN', 'CodeFormer'], required=True, help="Super-resolution model to use.")
    parser.add_argument('-iv', required=True, help="Input video filepath")
    parser.add_argument('-ia', required=True, help="Input audio filepath")
    parser.add_argument('-o', required=True, help="Output video filepath")
    return parser.parse_args()


def process_musetalk_lipsync_video(input_video_path, input_audio_path, smooth=False, start_time=0, end_time=-1, downsample=True, override=-1):
    video = sieve.File(path=input_video_path)
    audio = sieve.File(path=input_audio_path)
 
    output_video_path = "results/musetalk_output.mp4"
    musetalk = sieve.function.get("sieve/musetalk")
    output = musetalk.run(video, audio, start_time, end_time, downsample, override, smooth)

    output_video_path_from_sieve = output.path
    
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    shutil.move(output_video_path_from_sieve, output_video_path)

    return output_video_path


def initialize_model(model_name):
    if model_name == "GFPGAN":
        return GFPGANer(model_path="gfpgan_pretrained_models/GFPGANCleanv1-NoCE-C2.pth")
    else:
        raise ValueError("Invalid model name.")


def detect_resolution_mismatch(input_frame, output_frame):
    input_height, input_width, _ = input_frame.shape
    output_height, output_width, _ = output_frame.shape
    height_ratio = input_height / output_height
    width_ratio = input_width / output_width
    return height_ratio, width_ratio


def apply_superresolution(frame, model):
    _, _, enhanced_frame = model.enhance(frame)  # Ignore cropped_faces and restored_faces
    return enhanced_frame


def enhance_video_frames(frame_dir, output_dir, input_frame, model):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)

        height_ratio, width_ratio = detect_resolution_mismatch(input_frame, frame)
        if height_ratio > 1.0 or width_ratio > 1.0:
            print(f"Enhancing frame {frame_file} to match resolution.")
            enhanced_frame = apply_superresolution(frame, model)
            output_path = os.path.join(output_dir, frame_file)
            cv2.imwrite(output_path, enhanced_frame)
        else:
            output_path = os.path.join(output_dir, frame_file)
            cv2.imwrite(output_path, frame)


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", video_path, f"{output_dir}/frame_%04d.png"], check=True)


def combine_frames_to_video(frame_dir, audio_path, output_path, fps):
    temp_video = "temp_video.mp4"
    subprocess.run([
        "ffmpeg", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_video
    ], check=True)
    subprocess.run([
        "ffmpeg", "-i", temp_video, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", output_path
    ], check=True)
    os.remove(temp_video)


def cleanup_directories(*dirs):
    for directory in dirs:
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
        os.rmdir(directory)


def main():
    args = parse_arguments()
    input_video = args.iv
    input_audio = args.ia
    output_video = args.o
    superres_model = args.superres
    fps = 30

    # Process the video with Musetalk
    output_path = process_musetalk_lipsync_video(input_video, input_audio)

    frame_dir = "frames"
    enhanced_frame_dir = "enhanced_frames"
    extract_frames(output_path, frame_dir)

    model = initialize_model(superres_model)
    input_video_capture = cv2.VideoCapture(input_video)

    ret_in, input_frame = input_video_capture.read()
    if not ret_in:
        print("Error: Could not read input video.")
        return

    enhance_video_frames(frame_dir, enhanced_frame_dir, input_frame, model)
    combine_frames_to_video(enhanced_frame_dir, input_audio, output_video, fps)

    cleanup_directories(frame_dir, enhanced_frame_dir)


if __name__ == "__main__":
    main()
