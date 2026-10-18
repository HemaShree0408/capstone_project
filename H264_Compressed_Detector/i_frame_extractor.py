import os
import sys
import json
import struct
import subprocess

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FFMPEG_PATH  = os.path.join(BASE_DIR, "FFmpeg", "FFmpeg", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(BASE_DIR, "FFmpeg", "FFmpeg", "ffprobe.exe")

if not os.path.exists(FFMPEG_PATH):
    FFMPEG_PATH = "ffmpeg"
if not os.path.exists(FFPROBE_PATH):
    FFPROBE_PATH = "ffprobe"

# The custom FFmpeg build writes coefficients here.
# Note: The custom build automatically appends "_I.bin" to whatever we pass
# in the H264_COEFF_EXTRACT_FILE env variable.
BIN_BASE = os.path.join(BASE_DIR, "STRICT_FINAL")
BIN_PATH = BIN_BASE + "_I.bin"

# JSON file where we store fps + time_base for every video processed
# (sync_labels.py will read this file later)
VIDEO_METADATA_FILE = os.path.join(BASE_DIR, "video_metadata.json")

# feature_map.py lives one level up (in Capstone_Final)
sys.path.insert(0, BASE_DIR)
from feature_map import build_and_save_npy


# -----------------------------------------------------------------------
# STEP 1: Run ffprobe to get fps and time_base for this video
# -----------------------------------------------------------------------
def get_fps_and_timebase(video_path):
    cmd = [
        FFPROBE_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,time_base,width,height",
        "-of", "default=noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    fps       = 30.0
    time_base = "1/15360"
    width     = 1920
    height    = 1080

    for line in result.stdout.splitlines():
        if line.startswith("avg_frame_rate="):
            val = line.split("=")[1].strip()
            if "/" in val:
                num, den = val.split("/")
                if int(den) != 0:
                    fps = int(num) / int(den)
        elif line.startswith("time_base="):
            time_base = line.split("=")[1].strip()
        elif line.startswith("width="):
            width = int(line.split("=")[1].strip())
        elif line.startswith("height="):
            height = int(line.split("=")[1].strip())

    print(f"  [i_frame_extractor] fps={fps}  time_base={time_base}  res={width}x{height}")
    return fps, time_base, width, height


# -----------------------------------------------------------------------
# STEP 1b: Save fps + time_base for this video into the JSON metadata file
#          If the file already exists we just add/update the entry for this video.
#          The key is the relative path of the video (no extension).
# -----------------------------------------------------------------------
def save_video_metadata(video_key, fps, time_base, width, height):
    # Load existing data if file already exists
    if os.path.exists(VIDEO_METADATA_FILE):
        with open(VIDEO_METADATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Add / overwrite entry for this video
    data[video_key] = {
        "fps": fps,
        "time_base": time_base,
        "width": width,
        "height": height
    }

    # Save back
    with open(VIDEO_METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"  [i_frame_extractor] Saved metadata for '{video_key}' -> {VIDEO_METADATA_FILE}")


# -----------------------------------------------------------------------
# STEP 2: Run the custom FFmpeg build on the whole video
#         Writes ALL I-frame coefficients into STRICT_FINAL_I.bin
# -----------------------------------------------------------------------
def run_ffmpeg_on_video(video_path):
    if os.path.exists(BIN_PATH):
        os.remove(BIN_PATH)

    env = os.environ.copy()
    env["H264_COEFF_EXTRACT_FILE"] = BIN_BASE

    cmd = [
        FFMPEG_PATH,
        "-i", video_path,
        "-f", "null", "-"
    ]

    print(f"  [i_frame_extractor] Running FFmpeg on: {video_path}")
    subprocess.run(cmd, env=env)

    if os.path.exists(BIN_PATH) and os.path.getsize(BIN_PATH) > 0:
        size_mb = os.path.getsize(BIN_PATH) / (1024 * 1024)
        print(f"  [i_frame_extractor] STRICT_FINAL_I.bin created  ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  [i_frame_extractor] ERROR: STRICT_FINAL_I.bin was NOT created.")
        return False


# -----------------------------------------------------------------------
# Helper: scan the bin and return all unique frame numbers in order
# -----------------------------------------------------------------------
def get_unique_frame_numbers(bin_path):
    HDR_FORMAT    = '<qiiii'
    HDR_SIZE      = struct.calcsize(HDR_FORMAT)
    COEFF_BYTES   = 16 * 48 * 2 * 2
    LUMA_DC_BYTES = 3 * 16 * 2 * 2
    REC_SIZE      = HDR_SIZE + COEFF_BYTES + LUMA_DC_BYTES

    file_size = os.path.getsize(bin_path)
    num_mbs   = file_size // REC_SIZE

    seen     = []
    seen_set = set()
    with open(bin_path, 'rb') as f:
        for _ in range(num_mbs):
            chunk = f.read(REC_SIZE)
            frame_num = struct.unpack(HDR_FORMAT, chunk[:HDR_SIZE])[0]
            if frame_num not in seen_set:
                seen_set.add(frame_num)
                seen.append(frame_num)

    return seen


# -----------------------------------------------------------------------
# Helper: use ffprobe to get the ABSOLUTE video frame number (0-based)
#         for each I-frame in decode order.
# -----------------------------------------------------------------------
def get_i_frame_abs_indices(video_path):
    cmd = [
        FFPROBE_PATH,
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0',
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    i_indices = []
    for abs_idx, line in enumerate(result.stdout.splitlines()):
        if line.strip() == 'I':
            i_indices.append(abs_idx)
    print(f"  [i_frame_extractor] ffprobe: I-frames at absolute positions {i_indices}")
    return i_indices


# -----------------------------------------------------------------------
# STEP 3: Read the bin, build one .npy per I-frame, save to output_dir
#         npy files are named by ABSOLUTE video frame number, not 0,1,2
# -----------------------------------------------------------------------
def convert_bin_to_npy_files(output_dir, abs_i_indices):
    os.makedirs(output_dir, exist_ok=True)

    frame_numbers = get_unique_frame_numbers(BIN_PATH)
    print(f"  [i_frame_extractor] Found {len(frame_numbers)} I-frame(s) in bin.")

    saved_frames = []
    for seq_idx, frame_num in enumerate(frame_numbers):
        # Use the absolute video frame number as the filename
        if seq_idx < len(abs_i_indices):
            abs_frame = abs_i_indices[seq_idx]
        else:
            abs_frame = seq_idx   # fallback
        npy_path = os.path.join(output_dir, f"{abs_frame}.npy")
        ok = build_and_save_npy(BIN_PATH, frame_num, npy_path)
        if ok:
            print(f"  [i_frame_extractor] Saved npy: {npy_path}")
            saved_frames.append(abs_frame)

    return saved_frames


# -----------------------------------------------------------------------
# STEP 4: Delete the bin immediately after all npy files are saved
# -----------------------------------------------------------------------
def clear_bin():
    if os.path.exists(BIN_PATH):
        os.remove(BIN_PATH)
        print(f"  [i_frame_extractor] STRICT_FINAL_I.bin deleted.")


# -----------------------------------------------------------------------
# MAIN FUNCTION called by dataset_builder_coeffs.py for each video
#
#   video_path    : full path to the .mp4 file
#   npy_output_dir: folder where npy files for this video will be saved
#   video_key     : a unique string to identify this video in the metadata
#                   (usually the relative path from INPUT_ROOT without extension)
# -----------------------------------------------------------------------
def process_video(video_path, npy_output_dir, video_key):
    print(f"\n{'='*60}")
    print(f"[i_frame_extractor] Processing: {video_key}")
    print(f"{'='*60}")

    # Step 1 - get fps and time_base from ffprobe
    fps, time_base, width, height = get_fps_and_timebase(video_path)

    # Step 1b - save fps + time_base to the metadata JSON file
    save_video_metadata(video_key, fps, time_base, width, height)

    # Step 1c - get absolute frame numbers for each I-frame (for .npy naming)
    abs_i_indices = get_i_frame_abs_indices(video_path)

    # Step 2 - run FFmpeg to create the bin
    ok = run_ffmpeg_on_video(video_path)
    if not ok:
        print(f"  [i_frame_extractor] Skipping npy conversion - bin not created.")
        return

    # Step 3 - convert bin -> one npy per I-frame (named by absolute frame number)
    saved_frames = convert_bin_to_npy_files(npy_output_dir, abs_i_indices)
    print(f"  [i_frame_extractor] {len(saved_frames)} npy file(s) saved at absolute frame positions: {saved_frames}")

    # Step 4 - delete bin immediately so next video starts clean
    clear_bin()


# -----------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python i_frame_extractor.py <video_path> <npy_output_dir> <video_key>")
        sys.exit(1)

    process_video(sys.argv[1], sys.argv[2], sys.argv[3])