import os
import cv2
import numpy as np
import tensorflow as tf
from model import SSD300_H264
from data_loader import H264DatasetGenerator
import time

# --- CONFIG ---
DATASET_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\FINAL_SSD_DATASET"
MODEL_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\H264_Compressed_Detector\best_model.h5"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
GRID_SIZE = 7
IMG_SIZE = 300
BATCH_SIZE = 8

def compute_iou(box1, box2):
    """
    Computes IoU between two bounding boxes [xmin, ymin, xmax, ymax]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou

def decode_predictions(pred, conf_thresh=CONF_THRESHOLD):
    boxes = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            conf = pred[i, j, 0]
            if conf < conf_thresh:
                continue

            cx, cy, w, h = pred[i, j, 1:]
            abs_cx = (j + cx) / GRID_SIZE
            abs_cy = (i + cy) / GRID_SIZE

            xmin = max(0, (abs_cx - w/2))
            ymin = max(0, (abs_cy - h/2))
            xmax = min(1.0, (abs_cx + w/2))
            ymax = min(1.0, (abs_cy + h/2))

            boxes.append((xmin, ymin, xmax, ymax, conf))
    return boxes

def extract_gt_boxes_from_grid(grid):
    """ Extract raw GT boxes from the 7x7 grid. """
    boxes = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j, 0] > 0.5:  # Has object
                cx, cy, w, h = grid[i, j, 1:]
                abs_cx = (j + cx) / GRID_SIZE
                abs_cy = (i + cy) / GRID_SIZE

                xmin = max(0, (abs_cx - w/2))
                ymin = max(0, (abs_cy - h/2))
                xmax = min(1.0, (abs_cx + w/2))
                ymax = min(1.0, (abs_cy + h/2))

                boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def evaluate():
    print("=" * 50)
    print("🚀 STARTING EVALUATION ON STRICT TEST SPLIT")
    print("=" * 50)

    # 1. Provide Generator specifically in 'test' mode
    test_gen = H264DatasetGenerator(DATASET_PATH, mode="test", batch_size=BATCH_SIZE)
    if len(test_gen) == 0:
        print("❌ Not enough data in test split. Check dataset magnitude.")
        return

    # 2. Load the trained model
    print("🛠️ Loading Model...")
    model = SSD300_H264(n_classes=1, image_shape=(IMG_SIZE, IMG_SIZE, 3))
    try:
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load model weights at {MODEL_PATH}. Exception: {e}")
        return

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("📈 Evaluating...")
    start_time = time.time()
    
    # Iterate through test dataset batch by batch
    for batch_idx in range(len(test_gen)):
        X_batch, Y_batch = test_gen[batch_idx]
        preds = model.predict(X_batch, verbose=0)
        
        for i in range(len(X_batch)):
            pred_boxes = decode_predictions(preds[i])
            gt_boxes = extract_gt_boxes_from_grid(Y_batch[i])

            matched_gt = set()
            for p_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, g_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt: 
                        continue
                    iou = compute_iou(p_box[:4], g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= IOU_THRESHOLD:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            false_negatives += (len(gt_boxes) - len(matched_gt))

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_gen):
            print(f"Processed {batch_idx + 1}/{len(test_gen)} batches...")

    end_time = time.time()

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n" + "=" * 50)
    print(f"🎯 FINAL EVALUATION RESULTS (Took {end_time - start_time:.2f}s)")
    print("=" * 50)
    print(f"Total True Positives : {true_positives}")
    print(f"Total False Positives: {false_positives}")
    print(f"Total False Negatives: {false_negatives}")
    print("-" * 50)
    print(f"Precision : {precision * 100:.2f} %")
    print(f"Recall    : {recall * 100:.2f} %")
    print(f"F1-Score  : {f1 * 100:.2f} %")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()