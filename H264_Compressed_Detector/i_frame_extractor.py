import os
import numpy as np
import struct
import subprocess
from ssd_utils.feature_mapper import build_frequency_tensor

# --- CONFIG ---
COEFFS_PATH = "C:/Users/PESU-RF/capstone 211/capstone_project/FFmpeg/STRICT_FINAL.bin"
VIDEO_PATH = "C:/Users/PESU-RF/capstone 211/capstone_project/Human Activity Recognition - Video Dataset/Meet and Split/Meet and Split (100).mp4"
FFPROBE_PATH = "C:/Users/PESU-RF/capstone 211/capstone_project/FFmpeg/ffprobe.exe"
OUTPUT_DIR = "./test_extracted_npy"

def get_official_iframes(video_path):
    """Universal Judge: Asks ffprobe for the TRUE I-frame list for ANY video."""
    print(f"⚖️ Judge: Consulting ffprobe for TRUE I-frames for: {video_path}...")
    cmd = [
        FFPROBE_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    official_iframes = [i + 1 for i, t in enumerate(lines) if t.strip() == 'I']
    print(f"📖 Official Judge says these are the Real I-Frames: {official_iframes}")
    return sorted(official_iframes)

def run_ffmpeg_bin(video_path, bin_path):
    """Generates STRICT_FINAL.bin for this video."""
    print(f"⚡ Generating STRICT_FINAL.bin for: {video_path}")
    env = os.environ.copy()
    env["H264_COEFF_EXTRACT_FILE"] = bin_path
    cmd = [
        bin_path.replace("STRICT_FINAL.bin","ffmpeg.exe"), # placeholder for actual FFmpeg call
        "-i", video_path,
        "-f", "null", "-"
    ]
    subprocess.run(cmd, env=env, check=True)
    return os.path.exists(bin_path) and os.path.getsize(bin_path) > 0

def extract_all_iframes():
    # Ensure bin file exists and has data
    if not run_ffmpeg_bin(VIDEO_PATH, COEFFS_PATH):
        print(f"❌ Failed to generate .bin for {VIDEO_PATH}")
        return

    file_size = os.path.getsize(COEFFS_PATH)
    print(f"📈 Binary File Size: {file_size} bytes ({file_size//3284} macroblocks captured)")

    real_iframes = get_official_iframes(VIDEO_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for f_idx, f_num in enumerate(real_iframes):
        try:
            freq_map = build_frequency_tensor(COEFFS_PATH, frame_target=f_idx + 1)
            if freq_map is not None:
                np.save(os.path.join(OUTPUT_DIR, f"frame_{f_num}.npy"), freq_map)
                print(f"✅ EXTRACTED & MATCHED: Saved as 'frame_{f_num}.npy' 🚀🏆")
        except Exception as e:
            print(f"❌ Error extracting frame {f_num}: {e}")

    print(f"\n🏁 Finished! Your Capstone dataset for '{os.path.basename(VIDEO_PATH)}' is ready!")


def extract_iframes_from_video(video_path):
    """
    Importable function for dataset_builder.
    Runs FFmpeg once per I-frame (seeking to each one) with a unique bin file.
    This prevents the double-decoder truncation problem.
    Returns a dict: {frame_num (1-based): freq_map numpy array}
    """
    ffmpeg_path = FFPROBE_PATH.replace("ffprobe.exe", "ffmpeg.exe")

    # Get FPS to calculate seek timestamps
    fps_result = subprocess.run(
        [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    try:
        num, den = fps_result.stdout.strip().split('/')
        fps = int(num) / int(den)
    except Exception:
        fps = 30.0  # Safe fallback

    real_iframes = get_official_iframes(video_path)
    results = {}

    for f_num in real_iframes:
        # Unique, short, space-free bin path per frame
        bin_path = f"C:/Users/PESU-RF/frame_{f_num}.bin"
        timestamp = max(0.0, (f_num - 1) / fps)

        env = os.environ.copy()
        env["H264_COEFF_EXTRACT_FILE"] = bin_path

        # Seek to this I-frame and process only 1 frame
        # Both decoder instances will write the same frame → no data loss from truncation
        cmd = [ffmpeg_path, "-ss", str(timestamp), "-i", video_path,
               "-frames:v", "1", "-an", "-f", "null", "-"]
        subprocess.run(cmd, env=env, capture_output=True)

        if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
            print(f"❌ No bin data for frame {f_num}")
            continue

        # After seek, the frame counter in the bin restarts from 1
        freq_map = build_frequency_tensor(bin_path, frame_target=1)
        if freq_map is not None:
            results[f_num] = freq_map
        else:
            print(f"❌ No tensor data for frame {f_num}")

        # Clean up bin file
        if os.path.exists(bin_path):
            os.remove(bin_path)

    return results


if __name__ == "__main__":
    extract_all_iframes()