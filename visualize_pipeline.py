import sys
import os
import json
import numpy as np
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "H264_Compressed_Detector"))
sys.path.append(os.path.join(BASE_DIR, "H264_Compressed_Detector", "model"))

from i_frame_extractor import process_video as process_i_frames
from p_frame_extractor import process_video as process_p_frames
from bbox_propagator import propagate_bbox
from model.model import SSD300_H264
import tensorflow as tf

def decode_predictions_with_scores(pred, conf_thresh=0.5, grid_size=7):
    boxes = []
    scores = []
    for i in range(grid_size):
        for j in range(grid_size):
            conf = pred[i, j, 0]
            if conf < conf_thresh:
                continue

            cx, cy, w, h = pred[i, j, 1:]
            abs_cx = (j + cx) / grid_size
            abs_cy = (i + cy) / grid_size

            xmin = (abs_cx - w/2) * 300
            ymin = (abs_cy - h/2) * 300
            xmax = (abs_cx + w/2) * 300
            ymax = (abs_cy + h/2) * 300

            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(conf)
            
    return boxes, scores

def apply_nms(boxes, scores, iou_threshold=0.4):
    if not boxes: return []
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=50, iou_threshold=iou_threshold
    )
    return [boxes[i] for i in indices.numpy()]

def resize_feature_map(feat, size=(300, 300)):
    # Resize a 3-channel float32 numpy array using PIL to replace cv2.resize
    channels = []
    for c in range(3):
        img_c = Image.fromarray(feat[:, :, c], mode='F')
        img_resized = img_c.resize(size, Image.BILINEAR)
        channels.append(np.array(img_resized))
    return np.stack(channels, axis=-1)

def main():
    video_path = os.path.join(BASE_DIR, "data", "videos", "Walking.mp4")
    i_frame_dir = os.path.join(BASE_DIR, "results", "visualizations", "_tmp_I_frames")
    mv_dir = os.path.join(BASE_DIR, "results", "visualizations", "_tmp_MVS")
    video_key = "Walking"

    # 1. Extraction
    if os.path.exists(i_frame_dir): shutil.rmtree(i_frame_dir)
    if os.path.exists(mv_dir): shutil.rmtree(mv_dir)
    os.makedirs(i_frame_dir)
    os.makedirs(mv_dir)

    print("Extracting I-Frames...")
    process_i_frames(video_path, i_frame_dir, video_key)
    print("Extracting MVs...")
    process_p_frames(video_path, mv_dir, video_key)

    # 2. Load Model
    print("Loading Model...")
    model = SSD300_H264(n_classes=1, image_shape=(300, 300, 3))
    model.load_weights(os.path.join(BASE_DIR, "models", "best_model.h5"))

    # 3. Load GT
    gt_boxes = {}
    txt_path = os.path.join(BASE_DIR, "data", "annotations", "Walking.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('frame'): continue
                parts = line.split(',')
                if len(parts) < 7: continue
                f_idx = int(parts[0])
                x1_n, y1_n, x2_n, y2_n = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if f_idx not in gt_boxes:
                    gt_boxes[f_idx] = []
                gt_boxes[f_idx].append((x1_n, y1_n, x2_n, y2_n))

    # -------------------------------------------------
    #  <<<  NEW SECTION – replace the old av‑based I/O  >>>
    # -------------------------------------------------

    # ...

    # ----- after loading GT (still in main()) -----
    # Open the source video with imageio (no PyAV needed)
    reader = imageio.get_reader(video_path, 'ffmpeg')
    meta = reader.get_meta_data()
    width = meta['size'][0]
    height = meta['size'][1]
    fps = meta.get('fps', 30)  # fallback to 30 if fps missing

    # Prepare the output writer (imageio will use ffmpeg under the hood)
    out_path = os.path.join(BASE_DIR, "results", "visualizations", "Walking_Visualization.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)

    current_bboxes = []
    frame_idx = 0

    print("Rendering video...")
    for img_array in reader:                     # img_array is an RGB numpy array
        pts = frame_idx  # we no longer have a PTS; just use the sequential index

        # -------------------- I‑frame handling --------------------
        i_npy_path = os.path.join(i_frame_dir, f"{pts}.npy")
        if os.path.exists(i_npy_path):
            # It's an I‑frame → run the model
            feat = np.load(i_npy_path)
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            feat_resized = resize_feature_map(feat, size=(300, 300))
            feat_input = np.expand_dims(feat_resized, axis=0)

            pred = model.predict(feat_input, verbose=0)[0]
            boxes_300, scores = decode_predictions_with_scores(pred, conf_thresh=0.5)
            boxes_300_nms = apply_nms(boxes_300, scores)

            # Scale back to full‑resolution coordinates
            current_bboxes = []
            for (xmin, ymin, xmax, ymax) in boxes_300_nms:
                x1 = xmin * (width / 300.0)
                y1 = ymin * (height / 300.0)
                x2 = xmax * (width / 300.0)
                y2 = ymax * (height / 300.0)
                current_bboxes.append((x1, y1, x2, y2))

        else:
            # -------------------- P‑frame handling --------------------
            mv_path = os.path.join(mv_dir, f"{pts}.npy")
            if os.path.exists(mv_path):
                mv_array = np.load(mv_path, allow_pickle=True)
                new_bboxes = [propagate_bbox(bbox, mv_array) for bbox in current_bboxes]
                current_bboxes = new_bboxes

        # -------------------- Drawing --------------------
        pil_img = Image.fromarray(img_array)               # img_array is already RGB
        draw = ImageDraw.Draw(pil_img)

        # Default font (falls back if arial not available)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Predicted boxes (green)
        for (x1, y1, x2, y2) in current_bboxes:
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.text((x1, max(0, y1 - 25)), "Pred", fill=(0, 255, 0), font=font)

        # Ground‑truth boxes (red)
        if frame_idx in gt_boxes:
            for (nx1, ny1, nx2, ny2) in gt_boxes[frame_idx]:
                gx1, gy1, gx2, gy2 = nx1 * width, ny1 * height, nx2 * width, ny2 * height
                draw.rectangle([gx1, gy1, gx2, gy2], outline=(255, 0, 0), width=3)
                draw.text((gx1, max(0, gy1 - 25)), "GT", fill=(255, 0, 0), font=font)

        # Write the frame to the output video
        writer.append_data(np.array(pil_img))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Rendered {frame_idx} frames...")

    reader.close()
    writer.close()
    print(f"Done! Visualization saved to {out_path}")
    # -------------------------------------------------


if __name__ == "__main__":
    main()
