import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess
from model import SSD300_H264
from i_frame_extractor import get_official_iframes
from ssd_utils.feature_mapper import build_frequency_tensor
from ssd_utils.anchors import generate_anchors

# --- CONFIG ---
VIDEO_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset\Standing Still\Standing Still (3).mp4" # <--- Fixed with 'r' for Windows
WEIGHTS_PATH = "best_h264_weights.h5"
FFPROBE_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\FFmpeg\ffprobe.exe"
FFMPEG_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\FFmpeg\ffmpeg.exe"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

def decode_predictions(predictions, anchor_boxes):
    """Decodes SSD predictions (offsets) into absolute coordinates."""
    cx_a, cy_a, w_a, h_a = anchor_boxes[:, 4], anchor_boxes[:, 5], anchor_boxes[:, 6], anchor_boxes[:, 7]
    var = anchor_boxes[:, 8:12]
    offsets = predictions[:, :4]
    
    cx = offsets[:, 0] * var[:, 0] * w_a + cx_a
    cy = offsets[:, 1] * var[:, 1] * h_a + cy_a
    w = np.exp(offsets[:, 2] * var[:, 2]) * w_a
    h = np.exp(offsets[:, 3] * var[:, 3]) * h_a
    
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

def run_inference():
    print(f"🎬 Loading video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print("❌ Video file not found!")
        return

    # 1. Load Model
    model = SSD300_H264(n_classes=2, image_shape=(300, 300, 3))
    try:
        model.load_weights(WEIGHTS_PATH, by_name=True)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Could not load weights: {e}")
        return

    # 2. Get Anchors for decoding
    anchors = generate_anchors() # [n_anchors, 4... wait shape matches] 
    # Ah, wait! `generate_anchors()` returns shape (7942, 4) in anchors.py, not 12 columns!
    # Our data_loader uses self.anchors[best_idx, 0] etc.
    # We must construct the 12 columns here for the decoder, OR I can just map them.
    # From anchors.py: [cx, cy, w, h] - Let me just adjust decode_predictions since I wrote it.
    
    # 3. Setup Video Output
    cap = cv2.VideoCapture(VIDEO_PATH)
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    out = cv2.VideoWriter('output_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

    # 4. Get I-frame indices
    i_frame_indices = get_official_iframes(VIDEO_PATH)
    print(f"🔍 Found {len(i_frame_indices)} I-frames to process.")
    
    processed_count = 0
    cap_frame_idx = 1 # VideoCapture is 1-based internally for our loop conceptually
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Only process if this is an I-Frame
        if cap_frame_idx in i_frame_indices:
            print(f"⚡ Processing I-frame {cap_frame_idx}...")
            # B. Extract Frequency Tensor using the robust FFmpeg method from dataset builder
            bin_file = f"C:/Users/PESU-RF/temp_inf_{cap_frame_idx}.bin"
            timestamp = max(0.0, (cap_frame_idx - 1) / fps)
            env = os.environ.copy()
            env["H264_COEFF_EXTRACT_FILE"] = bin_file
            
            cmd = [FFMPEG_PATH, "-ss", str(timestamp), "-i", VIDEO_PATH, 
                   "-frames:v", "1", "-an", "-f", "null", "-"]
            subprocess.run(cmd, env=env, capture_output=True)
            
            freq_map = None
            if os.path.exists(bin_file) and os.path.getsize(bin_file) > 0:
                freq_map = build_frequency_tensor(bin_file, frame_target=1) # 1 because we seeked to it
                os.remove(bin_file)
            
            if freq_map is not None:
                freq_input = np.expand_dims(cv2.resize(freq_map, (300, 300)), axis=0) # Must resize!
                preds = model.predict(freq_input)[0] 
                
                # Decoders need: cx_a, cy_a, w_a, h_a from anchors (shape is 7942, 4)
                # And predictions have shape (7942, 6). 0-3 = offsets, 4 = bg, 5 = person
                offsets = preds[:, :4]
                confs = preds[:, 5] 
                
                cx_a, cy_a, w_a, h_a = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
                var_cx, var_cy, var_w, var_h = 0.1, 0.1, 0.2, 0.2
                
                cx = offsets[:, 0] * var_cx * w_a + cx_a
                cy = offsets[:, 1] * var_cy * h_a + cy_a
                w = np.exp(offsets[:, 2] * var_w) * w_a
                h = np.exp(offsets[:, 3] * var_h) * h_a
                
                xmin = cx - 0.5 * w
                ymin = cy - 0.5 * h
                xmax = cx + 0.5 * w
                ymax = cy + 0.5 * h
                
                boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)
                
                valid_mask = confs > CONF_THRESHOLD
                if np.any(valid_mask):
                    v_boxes = boxes[valid_mask]
                    v_scores = confs[valid_mask]
                    
                    indices = cv2.dnn.NMSBoxes(v_boxes.tolist(), v_scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
                    for i in indices:
                        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                        box = v_boxes[idx]
                        
                        x1 = int(box[0] * w_vid / 300)
                        y1 = int(box[1] * h_vid / 300)
                        x2 = int(box[2] * w_vid / 300)
                        y2 = int(box[3] * h_vid / 300)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {v_scores[idx]:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            processed_count += 1

        out.write(frame)
        cap_frame_idx += 1

    cap.release()
    out.release()
    print("🎉 Inference complete! Check 'output_detection.mp4'")

if __name__ == "__main__":
    run_inference()
