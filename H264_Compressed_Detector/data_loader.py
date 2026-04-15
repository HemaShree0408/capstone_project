import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class H264DatasetGenerator(Sequence):
    def __init__(self, dataset_dir, mode="train", batch_size=8, target_size=(300, 300), split_ratio=0.95):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.target_size = target_size

        self.grid_size = 7  # 7x7 grid

        self.maps_dir = os.path.join(dataset_dir, "maps")
        self.labels_dir = os.path.join(dataset_dir, "labels")

        all_samples = sorted([f.replace('.npy', '') for f in os.listdir(self.maps_dir) if f.endswith('.npy')])

        # Group by video name: 'Activity_(ID)_frame' -> 'Activity_(ID)'
        video_names = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in all_samples])))
        
        np.random.seed(42) # Seed to keep splits identical across train/val/test generators
        np.random.shuffle(video_names)

        num_videos = len(video_names)
        train_end = int(num_videos * 0.90)
        val_end = int(num_videos * 0.95)

        if mode == "train":
            selected_videos = set(video_names[:train_end])
        elif mode == "val":
            selected_videos = set(video_names[train_end:val_end])
        elif mode == "test":
            selected_videos = set(video_names[val_end:])
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

        # Select only frames that belong to the chosen videos
        self.sample_names = [f for f in all_samples if '_'.join(f.split('_')[:-1]) in selected_videos]

        print(f"✅ Loaded {len(self.sample_names)} frames from {len(selected_videos)} videos for {mode}")

    def __len__(self):
        return int(np.floor(len(self.sample_names) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.sample_names[idx * self.batch_size : (idx+1) * self.batch_size]

        X, Y = [], []

        for name in batch:
            # --- LOAD DCT ---
            img_path = os.path.join(self.maps_dir, f"{name}.npy")
            img = np.load(img_path)

            # Normalize
            img = (img - img.mean()) / (img.std() + 1e-8)

            # Resize
            img = cv2.resize(img, self.target_size)

            X.append(img)

            # --- CREATE GRID LABEL ---
            label_grid = np.zeros((self.grid_size, self.grid_size, 5))

            label_path = os.path.join(self.labels_dir, f"{name}.txt")

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue

                        parts = list(map(float, line.split()))
                        _, cx, cy, w, h = parts  # YOLO format

                        # Determine grid cell
                        grid_x = int(cx * self.grid_size)
                        grid_y = int(cy * self.grid_size)

                        grid_x = min(grid_x, self.grid_size - 1)
                        grid_y = min(grid_y, self.grid_size - 1)

                        # Relative position inside cell
                        cell_x = cx * self.grid_size - grid_x
                        cell_y = cy * self.grid_size - grid_y

                        # Assign to grid
                        label_grid[grid_y, grid_x, 0] = 1  # confidence
                        label_grid[grid_y, grid_x, 1:] = [cell_x, cell_y, w, h]

            Y.append(label_grid)

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)