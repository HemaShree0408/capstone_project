import os
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
input_root = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset"
output_root = input_root + "_annotations"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Load YOLO model
model = YOLO("yolov8n.pt") 

# -----------------------------
# PROCESS VIDEO
# -----------------------------
def process_video(video_path, video_name, relative_dir):
    cap = cv2.VideoCapture(video_path)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 🔥 Create matching subfolder in output
    output_dir = os.path.join(output_root, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Use exact input video name for the output, just replace extension with .txt
    annotation_file = os.path.join(output_dir, f"{video_name}.txt")
    frame_count = 0

    with open(annotation_file, "w") as f:
        f.write(f"# resolution: {W},{H}\n")
        f.write("frame,track_id,class,x1_norm,y1_norm,x2_norm,y2_norm\n")

        print(f"🎬 Processing: {video_name} ({W}x{H})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        if cls == 0 and conf > 0.5:
                            x1, y1, x2, y2 = map(float, box.xyxy[0])
                            
                            track_id = int(box.id[0]) if box.id is not None else -1

                            x1_n = x1 / W
                            y1_n = y1 / H
                            x2_n = x2 / W
                            y2_n = y2 / H

                            f.write(f"{frame_count},{track_id},{cls},{x1_n:.4f},{y1_n:.4f},{x2_n:.4f},{y2_n:.4f}\n")

            frame_count += 1

    cap.release()
    print(f"✅ Saved annotations: {annotation_file}")

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    for root, dirs, files in os.walk(input_root):
        # Skip the annotation folder itself to prevent reruns
        dirs[:] = [d for d in dirs if os.path.join(root, d) != output_root]

        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                v_path = os.path.join(root, file)

                # Keep the exact input filename (without extension) for output
                v_name = os.path.splitext(file)[0]

                # 🔥 Get relative subfolder path
                relative_dir = os.path.relpath(root, input_root)

                process_video(v_path, v_name, relative_dir)

    print("\n🎉 ALL ANNOTATIONS COMPLETED! Ready for H.264 Feature Mapping.")