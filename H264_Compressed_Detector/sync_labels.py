import os
import json
import shutil
import numpy as np

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANN_ROOT   = os.path.join(BASE_DIR, "Human Activity Recognition - Video Dataset_annotations")
NPY_ROOT   = os.path.join(BASE_DIR, "Human Activity Recognition - Video Dataset__DCT")

FINAL_ROOT = os.path.join(BASE_DIR, "FINAL_SSD_DATASET")
MAPS_OUT   = os.path.join(FINAL_ROOT, "maps")
LABELS_OUT = os.path.join(FINAL_ROOT, "labels")

VIDEO_METADATA_FILE = os.path.join(BASE_DIR, "video_metadata.json")

os.makedirs(MAPS_OUT,   exist_ok=True)
os.makedirs(LABELS_OUT, exist_ok=True)


# -----------------------------------------------------------------------
# Helper: convert a frame index (from the annotation file) to a PTS value
# -----------------------------------------------------------------------
def frame_idx_to_pts(frame_idx, fps, time_base):
    tb_num, tb_den = time_base.split("/")
    tb_num = int(tb_num)
    tb_den = int(tb_den)

    ticks_per_frame = tb_den / (fps * tb_num)
    pts = int(round(frame_idx * ticks_per_frame))
    return pts


# -----------------------------------------------------------------------
# Standalone mode: run after all videos have been processed by the 
#                  dataset builder / extractor.
# -----------------------------------------------------------------------
def sync_dataset():
    print("🔄 Starting Dataset Synchronization...")
    print(f"  NPY root: {NPY_ROOT}")
    print(f"  ANN root: {ANN_ROOT}")

    if not os.path.exists(VIDEO_METADATA_FILE):
        print(f"❌ ERROR: Metadata file not found: {VIDEO_METADATA_FILE}")
        print("   Did you run dataset_builder_coeffs.py first?")
        return

    # Load fps and time_base information
    with open(VIDEO_METADATA_FILE, "r") as f:
        video_metadata = json.load(f)

    total_synced = 0

    for root, dirs, files in os.walk(NPY_ROOT):
        npy_files = [f for f in files if f.endswith(".npy")]
        if not npy_files:
            continue

        # e.g. root = ".../ActionA/clip1"
        # rel_path = "ActionA\clip1"  <-- this exactly matches the video_key!
        video_key  = os.path.relpath(root, NPY_ROOT)
        video_name = os.path.basename(root)
        parent_dir = os.path.dirname(video_key)

        # Get fps and time_base for this specific video
        if video_key not in video_metadata:
            # Try with alternate slashes just in case
            alt_key = video_key.replace("\\", "/")
            if alt_key in video_metadata:
                video_key = alt_key
            else:
                alt_key = video_key.replace("/", "\\")
                if alt_key in video_metadata:
                    video_key = alt_key
                else:
                    print(f"  [sync_labels] WARNING: No metadata found for video '{video_key}'. Skipping.")
                    continue

        fps       = video_metadata[video_key]["fps"]
        time_base = video_metadata[video_key]["time_base"]
        frame_width = video_metadata[video_key].get("width", 1920)
        frame_height = video_metadata[video_key].get("height", 1080)

        ann_file = os.path.join(ANN_ROOT, parent_dir, f"{video_name}.txt")
        if not os.path.exists(ann_file):
            print(f"  [sync_labels] WARNING: Annotation file not found: {ann_file}")
            continue

        # Read annotation - we must group boxes by their converted PTS!
        # Notice: Multiple objects in the SAME frame will get the SAME PTS.
        # They will all be appended to the same list under that PTS key.
        pts_to_boxes = {}
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("frame"):
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                
                frame_idx = int(parts[0])
                cls       = int(parts[2]) + 1  # YOLO 0-based -> SSD 1-based
                x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                
                # Convert bounding box to center format
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1
                
                # Convert this annotation's frame index to PTS
                pts = frame_idx_to_pts(frame_idx, fps, time_base)
                
                if pts not in pts_to_boxes:
                    pts_to_boxes[pts] = []
                
                # Multiple lines (objects) with the same frame index get mapped
                # to the same PTS and appended here.
                pts_to_boxes[pts].append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Sync the I-frame .npy files with the annotations
        synced_for_this_video = 0
        for npy_name in npy_files:
            file_base = os.path.splitext(npy_name)[0]
            try:
                frame_num = int(file_base)  # This is the I-frame number from the bin
            except ValueError:
                continue
            
            # Convert I-frame number to PTS to look it up in annotation dict
            pts = frame_idx_to_pts(frame_num, fps, time_base)

            if pts in pts_to_boxes:
                safe_name = f"{video_name.replace(' ', '_')}_{frame_num}"
                
                # Copy npy map
                shutil.copy2(os.path.join(root, npy_name), os.path.join(MAPS_OUT, f"{safe_name}.npy"))
                
                # Write label text (all objects for this frame joined by newline)
                with open(os.path.join(LABELS_OUT, f"{safe_name}.txt"), "w") as lf:
                    lf.write("\n".join(pts_to_boxes[pts]))
                
                synced_for_this_video += 1
                total_synced += 1

        print(f"  [sync_labels] Synced {synced_for_this_video} I-frame(s) for video: {video_name}")

    print(f"\n✅ SYNC COMPLETE. Total samples in FINAL_SSD_DATASET: {total_synced}")


if __name__ == "__main__":
    sync_dataset()
