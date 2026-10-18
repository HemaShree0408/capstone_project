"""
prepare_action_dataset.py
=========================
End-to-End Pipeline for Action Recognition Preprocessing:
1. Iterates over all videos in the dataset.
2. Extracts I-frame DCT feature maps and P-frame Motion Vectors
   (one single-threaded custom-FFmpeg pass, vectorized parsing: fast_compressed_ops.py).
3. Runs the compressed-domain SSD300_H264 detection model on I-frames to get bounding boxes.
4. Propagates bounding boxes across P-frames (and holds/interpolates across B-frames/gaps).
5. Exports complete frame-by-frame bounding boxes to CSV (normalized + pixel coords).
6. Crops motion vectors within and slightly around the bounding box (with configurable margin).
7. Saves cropped motion vectors as per-frame .npy files ready for Action Recognition training/inference.
"""

import os
import sys
import re
import shutil
import argparse
import subprocess
import csv
import numpy as np
from PIL import Image
from pathlib import Path

# Ensure local modules are accessible
import importlib.util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = os.path.join(BASE_DIR, "H264_Compressed_Detector")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DETECTOR_DIR)

# Import project extractors and propagator
from i_frame_extractor import process_video as process_i_frames, get_fps_and_timebase
from p_frame_extractor import process_video as process_p_frames
from bbox_propagator import propagate_bbox, block_overlaps_bbox

# Safely load the SSD300_H264 architecture from H264_Compressed_Detector/model/model.py
_ssd_spec = importlib.util.spec_from_file_location(
    "ssd_detector_module",
    os.path.join(DETECTOR_DIR, "model", "model.py")
)
_ssd_module = importlib.util.module_from_spec(_ssd_spec)
_ssd_spec.loader.exec_module(_ssd_module)
SSD300_H264 = _ssd_module.SSD300_H264
import tensorflow as tf


def natural_sort_key(s):
    """Natural sorting key so v2.mp4 comes before v10.mp4"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'([0-9]+)', str(s))]


def decode_predictions_with_scores(pred, conf_thresh=0.5, grid_size=7):
    """
    Decodes SSD300_H264 predictions (7, 7, 5) into bounding boxes and confidence scores.
    Each cell: [confidence, cx, cy, w, h] relative to 300x300.
    """
    boxes = []
    scores = []
    for i in range(grid_size):
        for j in range(grid_size):
            conf = float(pred[i, j, 0])
            if conf < conf_thresh:
                continue

            cx, cy, w, h = pred[i, j, 1:]
            abs_cx = (j + cx) / grid_size
            abs_cy = (i + cy) / grid_size

            xmin = (abs_cx - w / 2) * 300.0
            ymin = (abs_cy - h / 2) * 300.0
            xmax = (abs_cx + w / 2) * 300.0
            ymax = (abs_cy + h / 2) * 300.0

            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(conf)

    return boxes, scores


def apply_nms(boxes, scores, iou_threshold=0.4, max_output_size=50):
    """Applies non-maximum suppression (NMS) to predicted boxes"""
    if not boxes:
        return []
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=max_output_size, iou_threshold=iou_threshold
    )
    return [boxes[i] for i in indices.numpy()]


def resize_feature_map(feat, size=(300, 300)):
    """Resizes a 3-channel DCT feature map to 300x300 for SSD input using PIL"""
    channels = []
    for c in range(3):
        img_c = Image.fromarray(feat[:, :, c].astype(np.float32), mode='F')
        img_resized = img_c.resize(size, Image.BILINEAR)
        channels.append(np.array(img_resized))
    return np.stack(channels, axis=-1)


def expand_bbox_with_margin(bbox, pad_ratio=0.15, frame_w=1920, frame_h=1080):
    """
    Expands bounding box by pad_ratio (e.g., 15%) on all sides to capture
    motion vectors slightly around the subject, clipped to frame boundaries.
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    
    exp_x1 = max(0.0, x1 - pad_w)
    exp_y1 = max(0.0, y1 - pad_h)
    exp_x2 = min(float(frame_w), x2 + pad_w)
    exp_y2 = min(float(frame_h), y2 + pad_h)
    
    return (exp_x1, exp_y1, exp_x2, exp_y2)


def get_video_frame_count(video_path):
    """Retrieves total frame count using ffprobe"""
    ffmpeg_dir = os.path.join(BASE_DIR, "FFmpeg", "FFmpeg")
    ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe.exe") if os.path.exists(os.path.join(ffmpeg_dir, "ffprobe.exe")) else "ffprobe"

    cmd = [
        ffprobe_exe, "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        count_str = res.stdout.strip()
        if count_str.isdigit():
            return int(count_str)
    except Exception:
        pass
    return None


class ActionDatasetPreprocessor:
    """End-to-end dataset preprocessor for action recognition"""

    def __init__(self, model_weights=None, conf_thresh=0.5, iou_thresh=0.4, padding_ratio=0.15):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.padding_ratio = padding_ratio
        
        # Load I-frame detection model
        if model_weights is None:
            model_weights = os.path.join(BASE_DIR, "models", "best_model.h5")
            
        print(f"Loading I-frame Detection Model weights from: {model_weights}")
        self.detector = SSD300_H264(n_classes=1, image_shape=(300, 300, 3))
        self.detector.load_weights(model_weights)
        print("✓ Detection model loaded successfully!")

    def detect_on_i_frame(self, i_frame_npy_path, width, height):
        """Runs SSD inference on a single I-frame DCT feature map"""
        feat = np.load(i_frame_npy_path)
        # Standardize features
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        feat_resized = resize_feature_map(feat, size=(300, 300))
        feat_input = np.expand_dims(feat_resized, axis=0)

        pred = self.detector.predict(feat_input, verbose=0)[0]
        boxes_300, scores = decode_predictions_with_scores(pred, conf_thresh=self.conf_thresh)
        boxes_300_nms = apply_nms(boxes_300, scores, iou_threshold=self.iou_thresh)

        # Rescale from 300x300 back to full video resolution (pixels)
        pixel_boxes = []
        for (xmin, ymin, xmax, ymax) in boxes_300_nms:
            x1 = max(0.0, xmin * (width / 300.0))
            y1 = max(0.0, ymin * (height / 300.0))
            x2 = min(float(width), xmax * (width / 300.0))
            y2 = min(float(height), ymax * (height / 300.0))
            pixel_boxes.append((x1, y1, x2, y2))

        return pixel_boxes

    def process_single_video(self, video_path, output_bbox_dir, output_mv_dir, keep_temp=False,
                             video_name=None):
        """
        Processes a single video with the vectorized front end (fast_compressed_ops.py):
        - ONE single-threaded custom-FFmpeg pass -> I-frame coefficients + P-frame MVs
        - Detects on I-frames (SSD300_H264)
        - Propagates to P-frames & holds on B-frames
        - Writes CSV and cropped MV .npy files

        Same algorithm and same outputs as process_single_video_original, but ~24x faster
        and without the multithreaded MV-corruption issue of the original extractor.
        Returns a dict with the video's metadata on success, False otherwise.
        """
        import fast_compressed_ops as fco

        if video_name is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'='*70}")
        print(f"Processing Video: {video_name} ({video_path})")
        print(f"{'='*70}")

        video_mv_out_dir = os.path.join(output_mv_dir, video_name)
        os.makedirs(video_mv_out_dir, exist_ok=True)
        os.makedirs(output_bbox_dir, exist_ok=True)

        # 1. Video metadata + frame types
        fps, time_base, width, height = get_fps_and_timebase(video_path)
        print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")
        abs_i, abs_p = fco.get_frame_type_indices(video_path)
        # (the custom ffprobe also writes STRICT_FINAL_*.bin, so every ffprobe call happens here,
        #  before extraction and cleanup)
        total_frames = get_video_frame_count(video_path)

        # 2. Single-pass extraction -> in-memory feature maps and MVs
        print("  [1/4] Extracting I-frame DCT coefficients + P-frame motion vectors (single pass)...")
        try:
            i_bin, p_bin = fco.run_single_pass_extraction(video_path)
            feature_maps = fco.build_iframe_feature_maps(i_bin, abs_i)
            p_mvs = fco.parse_p_bin_fast(p_bin, abs_p)
        finally:
            fco.cleanup_bins()
        print(f"  [2/4] {len(feature_maps)} I-frames, {len(p_mvs)} P-frames")

        if keep_temp:
            temp_dir = os.path.join(BASE_DIR, f"temp_proc_{video_name}")
            for sub, data in (("I_frames", feature_maps), ("MVS", p_mvs)):
                os.makedirs(os.path.join(temp_dir, sub), exist_ok=True)
                for f_idx, arr in data.items():
                    np.save(os.path.join(temp_dir, sub, f"{f_idx}.npy"), arr)

        all_extracted_indices = set(feature_maps) | set(p_mvs)
        if not all_extracted_indices:
            print(f"  [WARNING] No frames extracted for {video_name}. Skipping.")
            return False
        if total_frames is None or total_frames < max(all_extracted_indices) + 1:
            total_frames = max(all_extracted_indices) + 1

        # 3. I-frame detection + temporal propagation
        print(f"  [3/4] Running I-frame Detection & Temporal BBox Propagation ({total_frames} frames)...")
        key_boxes = {
            f_idx: fco.detect_on_feature_map(self.detector, feat, width, height,
                                             conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh)
            for f_idx, feat in feature_maps.items()
        }
        bboxes_by_frame = fco.propagate_timeline_fast(total_frames, key_boxes, p_mvs, width, height)

        # 4. Save bounding boxes to CSV
        csv_path = os.path.join(output_bbox_dir, f"{video_name}_bboxes.csv")
        csv_records = []
        for f_idx, box_list in sorted(bboxes_by_frame.items()):
            for track_id, (x1, y1, x2, y2) in enumerate(box_list):
                csv_records.append({
                    "frame": f_idx, "track_id": track_id, "class": "person",
                    "x1_norm": round(x1 / width, 6), "y1_norm": round(y1 / height, 6),
                    "x2_norm": round(x2 / width, 6), "y2_norm": round(y2 / height, 6),
                    "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                })
        fieldnames = ["frame", "track_id", "class", "x1_norm", "y1_norm", "x2_norm", "y2_norm", "x1", "y1", "x2", "y2"]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_records)
        print(f"  [OK] Saved bounding box CSV ({len(csv_records)} rows): {csv_path}")

        # 5. Crop MVs inside and slightly around the boxes (vectorized overlap test,
        #    same rule as bbox_propagator.block_overlaps_bbox)
        print(f"  [4/4] Cropping Motion Vectors with {int(self.padding_ratio*100)}% margin around BBoxes...")
        cropped_count = 0
        for f_idx, full_mv_array in p_mvs.items():
            if f_idx not in bboxes_by_frame:
                continue
            bx1 = full_mv_array["src_x"].astype(np.int64)
            by1 = full_mv_array["src_y"].astype(np.int64)
            bx2 = bx1 + full_mv_array["w"]
            by2 = by1 + full_mv_array["h"]
            keep = np.zeros(len(full_mv_array), dtype=bool)
            for box in bboxes_by_frame[f_idx]:
                x1, y1, x2, y2 = expand_bbox_with_margin(box, pad_ratio=self.padding_ratio,
                                                         frame_w=width, frame_h=height)
                keep |= (bx1 < x2) & (bx2 > x1) & (by1 < y2) & (by2 > y1)
            np.save(os.path.join(video_mv_out_dir, f"{f_idx}.npy"), full_mv_array[keep])
            cropped_count += 1

        print(f"  [OK] Saved {cropped_count} cropped MV .npy files to: {video_mv_out_dir}")
        return {"width": width, "height": height, "fps": round(fps, 3),
                "num_frames": total_frames, "num_mv_files": cropped_count}

    def process_single_video_original(self, video_path, output_bbox_dir, output_mv_dir, keep_temp=False,
                                      video_name=None):
        """
        ORIGINAL implementation (kept for reference / comparison):
        - Extracts I-frames and MVs (two FFmpeg passes, per-macroblock Python parsing)
        - Detects on I-frames
        - Propagates to P-frames & B-frames
        - Writes CSV and cropped MV .npy files

        Note: the original p_frame_extractor runs FFmpeg multithreaded, which can corrupt
        the MV dump (see fast_compressed_ops.run_single_pass_extraction).
        Returns a dict with the video's metadata on success, False otherwise.
        """
        if video_name is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'='*70}")
        print(f"Processing Video: {video_name} ({video_path})")
        print(f"{'='*70}")

        # Temporary working directories for full extraction
        temp_dir = os.path.join(BASE_DIR, f"temp_proc_{video_name}")
        temp_i_dir = os.path.join(temp_dir, "I_frames")
        temp_mv_dir = os.path.join(temp_dir, "MVS")
        os.makedirs(temp_i_dir, exist_ok=True)
        os.makedirs(temp_mv_dir, exist_ok=True)

        # Output folder for this video's cropped motion vectors
        video_mv_out_dir = os.path.join(output_mv_dir, video_name)
        os.makedirs(video_mv_out_dir, exist_ok=True)
        os.makedirs(output_bbox_dir, exist_ok=True)

        try:
            # 1. Get video metadata (resolution & fps)
            fps, time_base, width, height = get_fps_and_timebase(video_path)
            print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")

            # 2. Extract I-frame DCT feature maps
            print("  [1/4] Extracting I-frames...")
            process_i_frames(video_path, temp_i_dir, video_name)

            # 3. Extract full-frame P-frame Motion Vectors
            print("  [2/4] Extracting Motion Vectors...")
            process_p_frames(video_path, temp_mv_dir, video_name)

            # Gather all extracted frame indices
            i_frame_files = {
                int(os.path.splitext(f)[0]): os.path.join(temp_i_dir, f)
                for f in os.listdir(temp_i_dir) if f.endswith(".npy")
            }
            p_frame_files = {
                int(os.path.splitext(f)[0]): os.path.join(temp_mv_dir, f)
                for f in os.listdir(temp_mv_dir) if f.endswith(".npy")
            }

            all_extracted_indices = set(i_frame_files.keys()).union(set(p_frame_files.keys()))
            if not all_extracted_indices:
                print(f"  [WARNING] No frames extracted for {video_name}. Skipping.")
                return False

            total_frames = get_video_frame_count(video_path)
            max_idx = max(all_extracted_indices)
            if total_frames is None or total_frames < max_idx + 1:
                total_frames = max_idx + 1

            print(f"  [3/4] Running I-frame Detection & Temporal BBox Propagation ({total_frames} frames)...")

            # Track bounding boxes frame-by-frame
            # bboxes_by_frame: { frame_idx: list of (x1, y1, x2, y2) }
            bboxes_by_frame = {}
            current_bboxes = []

            for f_idx in range(total_frames):
                # Case A: Current frame is an I-frame -> Run object detection
                if f_idx in i_frame_files:
                    detected_boxes = self.detect_on_i_frame(
                        i_frame_files[f_idx], width=width, height=height
                    )
                    if detected_boxes:
                        current_bboxes = detected_boxes
                    # If I-frame has no new detections, we preserve previous boxes if any

                # Case B: Current frame is a P-frame -> Propagate with motion vectors
                elif f_idx in p_frame_files and current_bboxes:
                    mv_array = np.load(p_frame_files[f_idx], allow_pickle=True)
                    propagated = []
                    for bbox in current_bboxes:
                        new_box = propagate_bbox(bbox, mv_array)
                        # Clamp to video boundaries
                        clamped_box = (
                            max(0.0, min(float(width), new_box[0])),
                            max(0.0, min(float(height), new_box[1])),
                            max(0.0, min(float(width), new_box[2])),
                            max(0.0, min(float(height), new_box[3])),
                        )
                        propagated.append(clamped_box)
                    current_bboxes = propagated

                # Case C: B-frame or frame without MV -> Keep current_bboxes (hold position)
                else:
                    pass

                # Store result for this frame if boxes are present
                if current_bboxes:
                    bboxes_by_frame[f_idx] = list(current_bboxes)

            # 4. Save bounding boxes to CSV
            csv_path = os.path.join(output_bbox_dir, f"{video_name}_bboxes.csv")
            csv_records = []

            for f_idx, box_list in sorted(bboxes_by_frame.items()):
                for track_id, (x1, y1, x2, y2) in enumerate(box_list):
                    csv_records.append({
                        "frame": f_idx,
                        "track_id": track_id,
                        "class": "person",
                        "x1_norm": round(x1 / width, 6),
                        "y1_norm": round(y1 / height, 6),
                        "x2_norm": round(x2 / width, 6),
                        "y2_norm": round(y2 / height, 6),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    })

            fieldnames = ["frame", "track_id", "class", "x1_norm", "y1_norm", "x2_norm", "y2_norm", "x1", "y1", "x2", "y2"]
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_records)
            print(f"  ✓ Saved bounding box CSV ({len(csv_records)} rows): {csv_path}")

            # 5. Crop Motion Vectors inside and slightly around the bounding boxes
            print(f"  [4/4] Cropping Motion Vectors with {int(self.padding_ratio*100)}% margin around BBoxes...")
            cropped_count = 0

            for f_idx, mv_path in p_frame_files.items():
                if f_idx not in bboxes_by_frame:
                    continue

                full_mv_array = np.load(mv_path, allow_pickle=True)
                boxes_in_frame = bboxes_by_frame[f_idx]

                # Expand all bounding boxes in this frame with padding margin
                expanded_boxes = [
                    expand_bbox_with_margin(
                        box, pad_ratio=self.padding_ratio, frame_w=width, frame_h=height
                    )
                    for box in boxes_in_frame
                ]

                # Collect all macroblocks that overlap with any of the expanded bounding boxes
                overlapping_indices = []
                for i, mv in enumerate(full_mv_array):
                    src_x = int(mv['src_x'])
                    src_y = int(mv['src_y'])
                    blk_w = int(mv['w'])
                    blk_h = int(mv['h'])

                    if any(block_overlaps_bbox(src_x, src_y, blk_w, blk_h, exp_box) for exp_box in expanded_boxes):
                        overlapping_indices.append(i)

                cropped_mv_array = full_mv_array[overlapping_indices]
                out_npy_path = os.path.join(video_mv_out_dir, f"{f_idx}.npy")
                np.save(out_npy_path, cropped_mv_array)
                cropped_count += 1

            print(f"  ✓ Saved {cropped_count} cropped MV .npy files to: {video_mv_out_dir}")
            return {"width": width, "height": height, "fps": round(fps, 3),
                    "num_frames": total_frames, "num_mv_files": cropped_count}

        finally:
            # Clean up temporary full-frame directories unless requested to keep
            if not keep_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


def collect_video_files(input_path):
    """Gathers all mp4 videos from a file path or directory recursively"""
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".mp4"):
            return [input_path]
        return []

    videos = []
    for root, _, files in os.walk(input_path):
        for fname in sorted(files, key=natural_sort_key):
            if fname.lower().endswith(".mp4"):
                # Avoid processing already generated visualization videos
                if "_visualization" in fname.lower():
                    continue
                videos.append(os.path.join(root, fname))
    return videos


def video_name_and_label(video_path, video_root):
    """
    Uses the first sub-folder below video_root as the action label
    (videos/Walking/v1.mp4 -> label 'Walking', name 'Walking_v1') so that
    identically named files in different class folders do not overwrite each other.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if not os.path.isdir(video_root):
        return stem, ""
    rel_parent = os.path.relpath(os.path.dirname(os.path.abspath(video_path)), os.path.abspath(video_root))
    if rel_parent in (".", "") or rel_parent.startswith(".."):
        return stem, ""
    label = Path(rel_parent).parts[0]
    if stem.lower().startswith(label.lower()):
        return stem, label
    return f"{label}_{stem}", label


def update_manifest(manifest_path, records):
    """Merges records {video_name: row} into video_manifest.csv (used by the trainer)"""
    fieldnames = ["video_name", "label", "width", "height", "fps", "num_frames", "num_mv_files", "source_path"]
    rows = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows = {r["video_name"]: r for r in csv.DictReader(f)}
    rows.update(records)
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(rows, key=natural_sort_key):
            writer.writerow(rows[name])


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Preprocessor: I-frame SSD Detection + MV Propagation + Cropped MVs"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=BASE_DIR,
        help="Path to folder containing videos or a single video file"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default=os.path.join(BASE_DIR, "models", "best_model.h5"),
        help="Path to trained SSD300_H264 weights (.h5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(BASE_DIR, "Action_Recognition", "data", "raw"),
        help="Base directory for Action Recognition inputs (creates bboxes/ and motion_vectors/)"
    )
    parser.add_argument(
        "--padding_ratio",
        type=float,
        default=0.15,
        help="Margin around bounding box to retain motion vectors (0.15 = 15%% padding)"
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for I-frame SSD detections"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep intermediate uncropped I-frame and MV files"
    )

    args = parser.parse_args()

    # Setup output paths expected by Action_Recognition/config/config.yaml
    output_bbox_dir = os.path.join(args.output_dir, "bboxes")
    output_mv_dir = os.path.join(args.output_dir, "motion_vectors")
    os.makedirs(output_bbox_dir, exist_ok=True)
    os.makedirs(output_mv_dir, exist_ok=True)

    video_files = collect_video_files(args.video_dir)
    if not video_files:
        print(f"No .mp4 video files found in: {args.video_dir}")
        return

    print(f"\n=======================================================")
    print(f"Action Dataset Preprocessor")
    print(f"Total Videos Found: {len(video_files)}")
    print(f"Output BBoxes Dir : {output_bbox_dir}")
    print(f"Output MVs Dir    : {output_mv_dir}")
    print(f"Padding Ratio     : {args.padding_ratio * 100:.1f}%")
    print(f"=======================================================\n")

    preprocessor = ActionDatasetPreprocessor(
        model_weights=args.model_weights,
        conf_thresh=args.conf_thresh,
        padding_ratio=args.padding_ratio
    )

    manifest_path = os.path.join(args.output_dir, "video_manifest.csv")
    successful = 0
    for idx, video_path in enumerate(video_files, start=1):
        print(f"\n[{idx}/{len(video_files)}] Processing {os.path.basename(video_path)}")
        video_name, label = video_name_and_label(video_path, args.video_dir)
        info = preprocessor.process_single_video(
            video_path=video_path,
            output_bbox_dir=output_bbox_dir,
            output_mv_dir=output_mv_dir,
            keep_temp=args.keep_temp,
            video_name=video_name
        )
        if info:
            successful += 1
            # Written after every video so an interrupted run keeps its progress
            update_manifest(manifest_path, {
                video_name: {"video_name": video_name, "label": label,
                             "source_path": os.path.abspath(video_path), **info}
            })

    print(f"\n=======================================================")
    print(f"PREPROCESSING COMPLETED!")
    print(f"Successfully processed: {successful}/{len(video_files)} videos")
    print(f"BBoxes saved in       : {output_bbox_dir}")
    print(f"Cropped MVs saved in  : {output_mv_dir}")
    print(f"Video manifest        : {manifest_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
