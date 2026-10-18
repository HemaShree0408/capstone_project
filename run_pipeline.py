"""
run_pipeline.py
===============
End-to-end compressed-domain action recognition: give it a video, get back an
annotated RGB video with person bounding boxes and the recognised action.

    python run_pipeline.py --video data/videos/Walking.mp4

Pipeline (all existing modules, run in-line):
  1. Custom FFmpeg -> I-frame DCT feature maps + P-frame motion vectors
     (i_frame_extractor.py / p_frame_extractor.py)
  2. SSD300_H264 detection on every I-frame                (models/best_model.h5)
  3. Box propagation to P-frames with motion vectors, B-frames hold the box
     (bbox_propagator.py)
  4. Motion vectors cropped around the boxes               (prepare_action_dataset.py)
  5. TemporalCNN action recognition on 10-frame MV clips   (Action_Recognition/)
  6. RGB video rendered with boxes + per-clip and overall action labels

Outputs (in --output_dir/<video_name>/):
  <video_name>_action.mp4    annotated video
  <video_name>_result.json   overall prediction, top-k, per-clip predictions
  bboxes/<video_name>_bboxes.csv
"""

import os
import sys
import json
import time
import shutil
import argparse

# TensorFlow (detector) and PyTorch (action model) share the process:
# stop TF from reserving the whole GPU memory up front.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# The existing modules print symbols like "✓"; a Windows cp1252 console would crash on them
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import cv2
import imageio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_DIR = os.path.join(BASE_DIR, "Action_Recognition")
sys.path.insert(0, ACTION_DIR)

# Action recognition (PyTorch) - imported before TensorFlow
import torch
from src.inference import ActionRecognizer
from src.data_loader import load_bbox_csv

# Extraction + SSD detection + propagation + MV cropping (TensorFlow)
from prepare_action_dataset import ActionDatasetPreprocessor


# ============================================================
# ACTION TIMELINE
# ============================================================

def build_frame_actions(clips, num_frames_total, max_extend=15):
    """
    Turns per-clip predictions into a per-frame action label.
    A frame covered by several overlapping clips takes the most confident one;
    frames between / just outside clips take the nearest clip (up to max_extend frames).

    Returns: list (len num_frames_total) of (class_name, confidence) or None
    """
    frame_actions = [None] * num_frames_total
    for clip in clips:
        for f in range(clip['start_frame'], min(clip['end_frame'] + 1, num_frames_total)):
            current = frame_actions[f]
            if current is None or clip['confidence'] > current[1]:
                frame_actions[f] = (clip['class_name'], clip['confidence'])

    if not clips:
        return frame_actions

    # Fill uncovered frames from the nearest clip
    centers = np.array([(c['start_frame'] + c['end_frame']) / 2 for c in clips])
    for f in range(num_frames_total):
        if frame_actions[f] is not None:
            continue
        nearest = int(np.argmin(np.abs(centers - f)))
        clip = clips[nearest]
        dist = max(clip['start_frame'] - f, f - clip['end_frame'], 0)
        if dist <= max_extend:
            frame_actions[f] = (clip['class_name'], clip['confidence'])
    return frame_actions


# ============================================================
# RENDERING
# ============================================================

def put_label(img, text, x, y, color, scale, thickness):
    """Draws text on a filled background box, kept inside the image"""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    h, w = img.shape[:2]
    x = int(max(0, min(x, w - tw - 8)))
    y = int(max(th + baseline + 6, min(y, h - 2)))
    cv2.rectangle(img, (x, y - th - baseline - 6), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness, cv2.LINE_AA)


def render_video(video_path, out_path, boxes_by_frame, frame_actions, overall,
                 coded_width, coded_height, box_mode="largest"):
    """
    Decodes the RGB video and draws:
      - every tracked person box (green)
      - the action label on the person the action model looked at
        (largest box, or all boxes when the model used box_mode='union')
      - a top banner with the overall video-level action
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30) or 30
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)

    frame_idx = 0
    try:
        for rgb in reader:
            img = np.ascontiguousarray(rgb)
            h, w = img.shape[:2]
            # Boxes are in coded-stream pixels; rescale if the decoded frame differs
            sx, sy = w / float(coded_width), h / float(coded_height)
            scale = max(0.5, min(w, h) / 900.0)
            thick = max(2, int(round(min(w, h) / 360)))

            boxes = boxes_by_frame.get(frame_idx, [])
            action = frame_actions[frame_idx] if frame_idx < len(frame_actions) else None
            if boxes:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                main_idx = int(np.argmax(areas))
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    p1 = (int(x1 * sx), int(y1 * sy))
                    p2 = (int(x2 * sx), int(y2 * sy))
                    cv2.rectangle(img, p1, p2, (0, 255, 0), thick)
                    is_actor = box_mode == "union" or i == main_idx
                    if is_actor and action is not None:
                        text = f"{action[0]} {action[1]:.0%}"
                    else:
                        text = "person"
                    put_label(img, text, p1[0], p1[1] - 4, (0, 255, 0), scale, max(1, thick - 1))

            banner = f"Action: {overall['name']} ({overall['confidence']:.0%})" if overall \
                else "Action: unknown"
            put_label(img, banner, 10, int(40 * scale) + 10, (255, 200, 0), scale * 1.2, thick)
            put_label(img, f"frame {frame_idx}", 10, h - 10, (200, 200, 200), scale * 0.7,
                      max(1, thick - 1))

            writer.append_data(img)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Rendered {frame_idx} frames...")
    finally:
        reader.close()
        writer.close()
    return frame_idx


# ============================================================
# PIPELINE
# ============================================================

def run(video_path, output_dir, detector_weights, action_model, conf_thresh=0.5,
        device="cuda", clip_stride=None, keep_intermediate=False):
    t_start = time.time()
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(action_model):
        raise FileNotFoundError(f"Action model not found: {action_model} (train it first)")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_dir, video_name)
    bbox_dir = os.path.join(out_dir, "bboxes")
    mv_root = os.path.join(out_dir, "motion_vectors")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#' * 70}\n# Compressed-domain action recognition: {video_name}\n{'#' * 70}")

    # ---- Load action model first: its config defines the MV crop margin ----
    print("\n[Stage A] Loading action recognition model...")
    recognizer = ActionRecognizer(model_path=action_model, device=device, clip_stride=clip_stride)
    pad_ratio = recognizer.pad_ratio
    print(f"  Classes: {recognizer.class_names}")

    # ---- Stages 1-4: extraction, I-frame detection, propagation, MV crop ----
    print("\n[Stage B] Extraction -> I-frame detection -> propagation -> MV cropping")
    preprocessor = ActionDatasetPreprocessor(
        model_weights=detector_weights,
        conf_thresh=conf_thresh,
        padding_ratio=pad_ratio
    )
    info = preprocessor.process_single_video(
        video_path=video_path,
        output_bbox_dir=bbox_dir,
        output_mv_dir=mv_root,
        keep_temp=False,
        video_name=video_name
    )
    if not info:
        raise RuntimeError("Compressed-domain extraction failed (see log above)")

    width, height = info["width"], info["height"]
    bbox_csv = os.path.join(bbox_dir, f"{video_name}_bboxes.csv")
    mv_dir = os.path.join(mv_root, video_name)
    boxes_by_frame = load_bbox_csv(bbox_csv)
    print(f"  Person boxes on {len(boxes_by_frame)} / {info['num_frames']} frames")

    # ---- Stage 5: action recognition ----
    print("\n[Stage C] Running action recognition...")
    result = None
    try:
        result = recognizer.predict(mv_dir, bbox_csv, width=width, height=height)
        print(f"  Overall action: {result['predicted_class_name']} "
              f"({result['confidence']:.2%}) over {len(result['clips'])} clips")
        for p in result["top_5"]:
            print(f"    {p['rank']}. {p['class_name']}: {p['probability']:.4f}")
    except ValueError as e:
        print(f"  [WARNING] Action recognition skipped: {e}")

    clips = result["clips"] if result else []
    frame_actions = build_frame_actions(clips, info["num_frames"])
    overall = {"name": result["predicted_class_name"], "confidence": result["confidence"]} \
        if result else None

    # ---- Stage 6: render annotated RGB video ----
    print("\n[Stage D] Rendering annotated RGB video...")
    out_video = os.path.join(out_dir, f"{video_name}_action.mp4")
    n_rendered = render_video(video_path, out_video, boxes_by_frame, frame_actions, overall,
                              width, height, box_mode=recognizer.box_mode)

    # ---- Save results ----
    summary = {
        "video": video_path,
        "resolution": [width, height],
        "fps": info["fps"],
        "frames": n_rendered,
        "frames_with_person": len(boxes_by_frame),
        "action": overall,
        "top_5": result["top_5"] if result else [],
        "clips": clips,
        "output_video": out_video,
        "bboxes_csv": bbox_csv,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    result_json = os.path.join(out_dir, f"{video_name}_result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not keep_intermediate:
        shutil.rmtree(mv_root, ignore_errors=True)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    if overall:
        print(f"  Action        : {overall['name']} ({overall['confidence']:.2%})")
    print(f"  Output video  : {out_video}")
    print(f"  Results JSON  : {result_json}")
    print(f"  Total time    : {summary['runtime_sec']} s")
    print(f"{'=' * 70}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Video in -> annotated video out: compressed-domain detection, "
                    "tracking and action recognition"
    )
    parser.add_argument("--video", required=True, help="Input .mp4 (H.264) video")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "results", "pipeline_output"),
                        help="Where results are written (a sub-folder per video)")
    parser.add_argument("--detector_weights", default=os.path.join(BASE_DIR, "models", "best_model.h5"),
                        help="SSD300_H264 I-frame detector weights")
    parser.add_argument("--action_model",
                        default=os.path.join(ACTION_DIR, "models", "best_model.pth"),
                        help="Trained TemporalCNN checkpoint")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="I-frame detection confidence threshold")
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu' for the action model")
    parser.add_argument("--clip_stride", type=int, default=None,
                        help="Frames between action clips (smaller = finer timeline)")
    parser.add_argument("--keep_intermediate", action="store_true",
                        help="Keep the cropped motion vector .npy files")
    args = parser.parse_args()

    run(
        video_path=args.video,
        output_dir=args.output_dir,
        detector_weights=args.detector_weights,
        action_model=args.action_model,
        conf_thresh=args.conf_thresh,
        device=args.device,
        clip_stride=args.clip_stride,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":
    main()
