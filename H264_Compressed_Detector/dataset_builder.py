import os
import re
import numpy as np
from i_frame_extractor import extract_iframes_from_video

# --- CONFIG ---
INPUT_ROOT = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset"
OUTPUT_ROOT = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset__DCT"

# -------------------------------
# Helper to sort filenames numerically
# -------------------------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# =========================
# PROCESS SINGLE VIDEO
# =========================
def process_video(video_path, output_parent_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_parent_dir, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    print(f"⚡ Processing: {video_name}")

    # Call i_frame_extractor as-is — returns {frame_num: freq_map}
    results = extract_iframes_from_video(video_path)

    if not results:
        print(f"❌ No frames extracted for: {video_name}")
        return

    for f_num, freq_map in results.items():
        save_path = os.path.join(video_out_dir, f"{f_num - 1}.npy")
        np.save(save_path, freq_map)
        print(f"✅ Saved: {save_path}")

# =========================
# PROCESS DATASET RECURSIVELY
# =========================
def process_dataset():
    for main_folder in sorted(os.listdir(INPUT_ROOT), key=natural_sort_key):
        main_path = os.path.join(INPUT_ROOT, main_folder)
        if not os.path.isdir(main_path):
            continue

        main_out_dir = os.path.join(OUTPUT_ROOT, main_folder)
        os.makedirs(main_out_dir, exist_ok=True)

        for child in sorted(os.listdir(main_path), key=natural_sort_key):
            child_path = os.path.join(main_path, child)

            # If child is a folder, process its videos
            if os.path.isdir(child_path):
                child_out_dir = os.path.join(main_out_dir, child)
                os.makedirs(child_out_dir, exist_ok=True)
                for video_file in sorted(os.listdir(child_path), key=natural_sort_key):
                    if video_file.endswith(".mp4"):
                        video_path = os.path.join(child_path, video_file)
                        process_video(video_path, child_out_dir)

            # If child is a video file directly under main folder
            elif child.endswith(".mp4"):
                process_video(os.path.join(main_path, child), main_out_dir)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_dataset()
    