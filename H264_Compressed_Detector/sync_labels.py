import os
import numpy as np
import shutil
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
NPY_ROOT = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset__DCT"
ANN_ROOT = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset_annotations"

# Final destination for training
FINAL_ROOT = r"C:\Users\PESU-RF\capstone 211\capstone_project\FINAL_SSD_DATASET"
MAPS_OUT = os.path.join(FINAL_ROOT, "maps")
LABELS_OUT = os.path.join(FINAL_ROOT, "labels")

os.makedirs(MAPS_OUT, exist_ok=True)
os.makedirs(LABELS_OUT, exist_ok=True)

def sync_dataset():
    print("🔄 Starting Dataset Synchronization...")
    print(f"📂 Searching for .npy files in: {NPY_ROOT}")
    
    total_synced = 0
    
    # 1. Look for all .npy files in subfolders
    for root, dirs, files in os.walk(NPY_ROOT):
        npy_files = [f for f in files if f.endswith('.npy')]
        if not npy_files:
            continue
            
        rel_path = os.path.relpath(root, NPY_ROOT)
        video_name = os.path.basename(root)
        parent_dir = os.path.dirname(rel_path)
        
        ann_file = os.path.join(ANN_ROOT, parent_dir, f"{video_name}.txt")
        if not os.path.exists(ann_file):
            continue

        frame_data = {}
        with open(ann_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('frame'):
                    continue
                parts = line.strip().split(',')
                if len(parts) < 7: continue
                
                f_idx = int(parts[0])
                # SSD Standard: 0 = Background, 1 = Person. 
                # Since YOLO labels were 0 for person, we add 1.
                cls = int(parts[2]) + 1
                x1, y1, x2, y2 = map(float, parts[3:7])
                
                # Convert to center coords for data_loader
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                if f_idx not in frame_data:
                    frame_data[f_idx] = []
                frame_data[f_idx].append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 3. For each extracted frame .npy, grab the boxes
        for npy_name in npy_files:
            file_base = os.path.splitext(npy_name)[0]
            try:
                f_idx = int(file_base)
            except:
                continue
            
            if f_idx in frame_data:
                # Unique name for the final dataset: videoName_frameIdx
                safe_name = f"{video_name.replace(' ', '_')}_{f_idx}"
                
                # Copy .npy to maps
                shutil.copy2(os.path.join(root, npy_name), os.path.join(MAPS_OUT, f"{safe_name}.npy"))
                
                # Write boxes to labels
                with open(os.path.join(LABELS_OUT, f"{safe_name}.txt"), 'w') as f:
                    f.write("\n".join(frame_data[f_idx]))
                
                total_synced += 1

    print("\n" + "="*40)
    print(f"✅ SYNC COMPLETE!")
    print(f"📊 Total Samples in FINAL_SSD_DATASET: {total_synced}")
    print("="*40)

if __name__ == "__main__":
    sync_dataset()
