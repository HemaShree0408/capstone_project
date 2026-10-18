

import os
import re
from i_frame_extractor import process_video as process_i_frames
from p_frame_extractor import process_video as process_p_frames

# -----------------------------------------------------------------------
# CONFIG  - set these paths to match your dataset layout
# -----------------------------------------------------------------------
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_ROOT  = os.path.join(BASE_DIR, "Human Activity Recognition - Video Dataset")
OUTPUT_ROOT = os.path.join(BASE_DIR, "Human Activity Recognition - Video Dataset__DCT")
MVS_ROOT    = os.path.join(BASE_DIR, "Human Activity Recognition - Video Dataset__MVS")
ANN_ROOT    = os.path.join(BASE_DIR, "Human Activity Recognition - Video Dataset_annotations")


# -----------------------------------------------------------------------
# Helper: natural sort so v2.mp4 comes before v10.mp4
# -----------------------------------------------------------------------
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'([0-9]+)', s)]


# -----------------------------------------------------------------------
# Walk INPUT_ROOT, find every .mp4, process them one by one
# -----------------------------------------------------------------------
def process_dataset():
    print(f"[dataset_builder] Input  folder : {INPUT_ROOT}")
    print(f"[dataset_builder] DCT Output folder : {OUTPUT_ROOT}")
    print(f"[dataset_builder] MVS Output folder : {MVS_ROOT}")

    # Collect all mp4 paths in sorted order
    all_videos = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        dirs.sort(key=natural_sort_key)
        for fname in sorted(files, key=natural_sort_key):
            if fname.lower().endswith(".mp4"):
                all_videos.append(os.path.join(root, fname))

    total = len(all_videos)
    print(f"[dataset_builder] Found {total} video(s) to process.\n")

    for idx, video_path in enumerate(all_videos, start=1):
        # Mirror the folder structure from INPUT_ROOT into OUTPUT_ROOT
        rel_path   = os.path.relpath(video_path, INPUT_ROOT)   # e.g. ActionA\clip1.mp4
        rel_dir    = os.path.dirname(rel_path)                 # e.g. ActionA
        video_name = os.path.splitext(os.path.basename(rel_path))[0]  # e.g. clip1
        
        # A unique key for this video to store in video_metadata.json (e.g. ActionA\clip1)
        video_key  = os.path.splitext(rel_path)[0]

        # Each video gets its own sub-folder for its I-frame npy files
        npy_output_dir = os.path.join(OUTPUT_ROOT, rel_dir, video_name)
        os.makedirs(npy_output_dir, exist_ok=True)
        
        # Each video gets its own sub-folder for its P-frame MV npy files
        mv_output_dir = os.path.join(MVS_ROOT, rel_dir, video_name)
        os.makedirs(mv_output_dir, exist_ok=True)

        print(f"[dataset_builder] ({idx}/{total}) {rel_path}")

        # Call i_frame_extractor
        process_i_frames(
            video_path     = video_path,
            npy_output_dir = npy_output_dir,
            video_key      = video_key
        )
        
        # Call p_frame_extractor
        process_p_frames(
            video_path    = video_path,
            mv_output_dir = mv_output_dir,
            video_key     = video_key
        )

        print(f"[dataset_builder] Done with: {rel_path}\n")

    print("[dataset_builder] ALL VIDEOS PROCESSED.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    process_dataset()