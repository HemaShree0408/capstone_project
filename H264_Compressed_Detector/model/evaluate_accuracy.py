import sys
import os
import numpy as np
import tensorflow as tf
import cv2

# Add the current directory to sys.path to support local imports when run from elsewhere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import SSD300_H264
except ImportError:
    from .model import SSD300_H264

# --- CONFIG ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(BASE_DIR, "FINAL_SSD_DATASET")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
GRID_SIZE = 7

# --- IOU FUNCTION ---
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# --- CONVERT GRID TO BOXES ---
def decode_predictions(pred):
    boxes = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            conf = pred[i, j, 0]

            if conf < CONF_THRESHOLD:
                continue

            cx, cy, w, h = pred[i, j, 1:]

            # convert cell → image coords
            abs_cx = (j + cx) / GRID_SIZE
            abs_cy = (i + cy) / GRID_SIZE

            xmin = (abs_cx - w/2) * 300
            ymin = (abs_cy - h/2) * 300
            xmax = (abs_cx + w/2) * 300
            ymax = (abs_cy + h/2) * 300

            boxes.append([xmin, ymin, xmax, ymax])

    return boxes

# --- LOAD GT BOXES ---
def load_gt(label_path):
    boxes = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                _, cx, cy, w, h = map(float, line.split())

                cx *= 300
                cy *= 300
                w *= 300
                h *= 300

                xmin = cx - w/2
                ymin = cy - h/2
                xmax = cx + w/2
                ymax = cy + h/2

                boxes.append([xmin, ymin, xmax, ymax])

    return boxes

# --- MAIN EVALUATION ---
def evaluate_accuracy():
    print("🛠️ Loading Model...")

    model = SSD300_H264(n_classes=1, image_shape=(300, 300, 3))
    model.load_weights(WEIGHTS_PATH)

    maps_dir = os.path.join(DATASET_PATH, "maps")
    labels_dir = os.path.join(DATASET_PATH, "labels")

    files = [f.replace(".npy", "") for f in os.listdir(maps_dir) if f.endswith(".npy")]

    np.random.seed(42)
    np.random.shuffle(files)

    split = int(len(files) * 0.95)
    test_files = files[split:]

    print(f"✅ Evaluating on {len(test_files)} samples")

    TP, FP, FN = 0, 0, 0

    for idx, name in enumerate(test_files):
        # --- INPUT ---
        img = np.load(os.path.join(maps_dir, name + ".npy"))
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = cv2.resize(img, (300, 300))
        img = np.expand_dims(img, axis=0)

        # --- PREDICT ---
        pred = model.predict(img, verbose=0)[0]

        pred_boxes = decode_predictions(pred)
        gt_boxes = load_gt(os.path.join(labels_dir, name + ".txt"))

        matched = []

        for p in pred_boxes:
            best_iou = 0
            best_idx = -1

            for i, g in enumerate(gt_boxes):
                if i in matched:
                    continue

                val = iou(p, g)
                if val > best_iou:
                    best_iou = val
                    best_idx = i

            if best_iou >= IOU_THRESHOLD:
                TP += 1
                matched.append(best_idx)
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(test_files)}")

    # --- METRICS ---
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n" + "="*40)
    print("🎯 FINAL RESULTS")
    print("="*40)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_accuracy()