"""
p_frame_extractor.py
====================

Runs the custom FFmpeg binary (which now writes STRICT_FINAL_P.bin via the
modified h264_coeff_extract.c), then splits that binary file into per-frame
NumPy .npy files that visualize_pipeline.py consumes.

P-frame binary record layout (little-endian, 80 bytes each):
  int64_t  frame_idx       sequential P-frame index (0, 1, 2, …)
  int32_t  mb_x            macroblock column
  int32_t  mb_y            macroblock row
  int16_t  mv[16][2]       16 × (dx, dy) quarter-pixel units  (list-0)

Each .npy output file is a structured NumPy array with the fields that
bbox_propagator.py already expects:
  source   (u2)   – always 0 (forward reference, list-0)
  w        (u2)   – macroblock width  in pixels (always 16)
  h        (u2)   – macroblock height in pixels (always 16)
  src_x    (u2)   – source x in pixels (mb_x * 16)
  src_y    (u2)   – source y in pixels (mb_y * 16)
  dst_x    (u2)   – destination x = src_x + dx/4
  dst_y    (u2)   – destination y = src_y + dy/4
  flags    (u2)   – 0 (forward)
  motion_x (i2)   – dominant dx (median of 16 sub-blocks, quarter-pixel)
  motion_y (i2)   – dominant dy (median of 16 sub-blocks, quarter-pixel)
  motion_scale (u2) – always 4 (quarter-pixel scale factor)

One .npy file is created per P-frame:  {mv_output_dir}/{frame_idx}.npy
"""

import os
import sys
import subprocess
import struct
import numpy as np

# ---------------------------------------------------------------------------
# NumPy structured dtype that bbox_propagator.py reads
# ---------------------------------------------------------------------------
MV_DTYPE = np.dtype([
    ('source',       np.uint16),
    ('w',            np.uint16),
    ('h',            np.uint16),
    ('src_x',        np.uint16),
    ('src_y',        np.uint16),
    ('dst_x',        np.uint16),
    ('dst_y',        np.uint16),
    ('flags',        np.uint16),
    ('motion_x',     np.int16),
    ('motion_y',     np.int16),
    ('motion_scale', np.uint16),
])

# Binary record size written by the custom FFmpeg decoder
_RECORD_BYTES = 8 + 4 + 4 + 16 * 2 * 2   # = 80 bytes
_RECORD_STRUCT = struct.Struct('<q i i 32h')  # frame_idx, mb_x, mb_y, 32 int16 (16×dx,dy interleaved)


def _dominant_mv(mv_flat):
    """Return (dx, dy) as the element-wise median over 16 sub-block vectors.
    
    Motion vectors in h->cur_pic.motion_val are int16_t quarter-pixel values.
    However, uninitialized slots (intra MBs inside a P-slice, skipped MBs)
    can contain garbage.  We filter values outside a sane range before taking
    the median.
    """
    dx_vals = np.array(mv_flat[0::2], dtype=np.int16)   # even indices → dx
    dy_vals = np.array(mv_flat[1::2], dtype=np.int16)   # odd  indices → dy

    # Quarter-pixel units: ±4096 = ±1024 pixels, generous upper bound
    MAX_QP = 4096
    valid = (np.abs(dx_vals) <= MAX_QP) & (np.abs(dy_vals) <= MAX_QP)
    if valid.any():
        dx = int(np.median(dx_vals[valid]))
        dy = int(np.median(dy_vals[valid]))
    else:
        # All slots look like garbage → treat as zero motion
        dx, dy = 0, 0

    # Clamp to int16 range just in case
    dx = max(-32768, min(32767, dx))
    dy = max(-32768, min(32767, dy))
    return dx, dy


def _parse_p_bin(bin_path):
    """
    Parse STRICT_FINAL_P.bin produced by the custom FFmpeg binary.

    IMPORTANT: The 'frame_idx' field written by the C code may not be a clean
    sequential 0,1,2,... counter (it can be PTS-based or otherwise corrupt).
    We therefore IGNORE it and instead detect frame boundaries ourselves:
    whenever (mb_x == 0 AND mb_y == 0) we start a new frame group.

    Returns a dict:  {sequential_frame_idx: [record, ...]}
    where sequential_frame_idx is 0, 1, 2, ... in decode order.
    """
    frames = {}
    if not os.path.isfile(bin_path):
        return frames

    with open(bin_path, 'rb') as f:
        raw = f.read()

    n_records = len(raw) // _RECORD_BYTES
    if n_records == 0:
        return frames

    seq_idx = -1   # will become 0 on the first (mb_x=0, mb_y=0) record

    for i in range(n_records):
        chunk = raw[i * _RECORD_BYTES: (i + 1) * _RECORD_BYTES]
        if len(chunk) < _RECORD_BYTES:
            break
        fields = _RECORD_STRUCT.unpack(chunk)
        # fields[0] = frame_idx (IGNORED - may be garbage)
        mb_x    = fields[1]
        mb_y    = fields[2]
        mv_flat = fields[3:]   # 32 int16 values: dx0,dy0, dx1,dy1, ...

        # New frame starts when the first macroblock (top-left) appears
        if mb_x == 0 and mb_y == 0:
            seq_idx += 1

        if seq_idx < 0:
            # Shouldn't happen, but skip any records before the first (0,0)
            continue

        if seq_idx not in frames:
            frames[seq_idx] = []
        frames[seq_idx].append((mb_x, mb_y, mv_flat))

    return frames


def _write_frame_npy(mb_records, out_path):
    """
    Convert a list of (mb_x, mb_y, mv_flat) records for one P-frame into
    the structured NumPy array and save it as an .npy file.

    We build the array field-by-field (not via list-of-tuples) to avoid
    OverflowError from NumPy rejecting large Python ints for uint16 fields.
    """
    n = len(mb_records)
    if n == 0:
        np.save(out_path, np.array([], dtype=MV_DTYPE))
        return

    arr = np.zeros(n, dtype=MV_DTYPE)
    arr['source']       = 0
    arr['w']            = 16
    arr['h']            = 16
    arr['flags']        = 0
    arr['motion_scale'] = 4

    for i, (mb_x, mb_y, mv_flat) in enumerate(mb_records):
        dx, dy = _dominant_mv(mv_flat)  # clamped to int16 range

        src_x = np.clip(mb_x * 16, 0, 65535).astype(np.uint16) if isinstance(mb_x * 16, np.generic) else int(max(0, min(65535, mb_x * 16)))
        src_y = np.clip(mb_y * 16, 0, 65535).astype(np.uint16) if isinstance(mb_y * 16, np.generic) else int(max(0, min(65535, mb_y * 16)))
        dst_x = int(max(0, min(65535, int(src_x) + dx // 4)))
        dst_y = int(max(0, min(65535, int(src_y) + dy // 4)))

        arr[i]['src_x']    = src_x
        arr[i]['src_y']    = src_y
        arr[i]['dst_x']    = dst_x
        arr[i]['dst_y']    = dst_y
        arr[i]['motion_x'] = dx
        arr[i]['motion_y'] = dy

    np.save(out_path, arr)


def _get_p_frame_abs_indices(video_path, project_dir):
    """
    Use ffprobe to list every frame in decode order with its type (I/P/B).
    Returns a list of absolute video frame numbers (0-based) for every P-frame,
    in decode order.  So result[0] is the absolute index of the 1st P-frame,
    result[1] is the 2nd P-frame, etc.

    This is used to rename the sequential P-frame .npy files to their true
    absolute position in the video so that visualize_pipeline.py can find them
    by frame_idx.
    """
    ffprobe_exe = os.path.join(project_dir, 'FFmpeg', 'FFmpeg', 'ffprobe.exe')
    if not os.path.isfile(ffprobe_exe):
        ffprobe_exe = 'ffprobe'  # fall back to system

    cmd = [
        ffprobe_exe,
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0',
        video_path,
    ]
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True,
                            text=True, encoding='utf-8', errors='replace')

    p_indices = []
    for abs_idx, line in enumerate(result.stdout.splitlines()):
        frame_type = line.strip()
        if frame_type == 'P':
            p_indices.append(abs_idx)

    print(f"  [p_frame_extractor] ffprobe found {len(p_indices)} P-frames in video.")
    return p_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_video(video_path: str, mv_output_dir: str, video_key: str):
    """
    1. Run the custom FFmpeg binary on `video_path` to produce
       STRICT_FINAL_P.bin (written in the current working directory).
    2. Parse the binary and split it into per-frame .npy files under
       `mv_output_dir`.

    B-frames produce no records in STRICT_FINAL_P.bin (the C code skips
    them), so only P-frames get a .npy file.
    """
    os.makedirs(mv_output_dir, exist_ok=True)

    # Locate the custom FFmpeg binary (relative to this script's directory)
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    ffmpeg_exe  = os.path.join(project_dir, 'FFmpeg', 'FFmpeg', 'ffmpeg.exe')

    if not os.path.isfile(ffmpeg_exe):
        print(f"  [p_frame_extractor] ERROR: ffmpeg.exe not found at {ffmpeg_exe}")
        sys.exit(1)

    # The custom binary writes STRICT_FINAL_P.bin in the CWD.
    # We run ffmpeg from the project root so the file lands there.
    bin_path = os.path.join(project_dir, 'STRICT_FINAL_P.bin')

    print(f"\n{'='*60}")
    print(f"[p_frame_extractor] Processing: {video_key} for Motion Vectors")
    print(f"{'='*60}")
    print(f"  [p_frame_extractor] Running custom FFmpeg to extract MVs …")

    cmd = [
        ffmpeg_exe,
        '-hide_banner',
        '-loglevel', 'info',
        '-y',
        '-i', video_path,
        '-an',          # drop audio
        '-f', 'null',   # discard video output
        'NUL',
    ]

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )

    if result.returncode != 0:
        print(f"  [p_frame_extractor] FFmpeg stderr:\n{result.stderr}")
        print(f"  [p_frame_extractor] ERROR: FFmpeg exited with code {result.returncode}")
        sys.exit(1)

    if not os.path.isfile(bin_path):
        print(f"  [p_frame_extractor] ERROR: Expected {bin_path} but it was not created.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Build a mapping: sequential P-frame index -> absolute video frame number
    # using ffprobe so .npy files are named by absolute frame position
    # -----------------------------------------------------------------------
    abs_p_indices = _get_p_frame_abs_indices(video_path, project_dir)

    # -----------------------------------------------------------------------
    # Parse the binary and write per-frame .npy files
    # -----------------------------------------------------------------------
    print(f"  [p_frame_extractor] Parsing {bin_path} ...")
    frames = _parse_p_bin(bin_path)   # {0: [...], 1: [...], ...} - sequential P-frame idx

    if not frames:
        print("  [p_frame_extractor] WARNING: No P-frame records found in binary.")
    else:
        n_written = 0
        for seq_idx in sorted(frames.keys()):
            # Map sequential P-frame index to absolute video frame number
            if seq_idx < len(abs_p_indices):
                abs_frame = abs_p_indices[seq_idx]
            else:
                # ffprobe count mismatch - fall back to sequential
                abs_frame = seq_idx
            out_path = os.path.join(mv_output_dir, f"{abs_frame}.npy")
            _write_frame_npy(frames[seq_idx], out_path)
            n_written += 1
        print(f"  [p_frame_extractor] Wrote {n_written} .npy files to {mv_output_dir}/")

    # Clean up the binary file
    if os.path.exists(bin_path):
        os.remove(bin_path)
        print(f"  [p_frame_extractor] STRICT_FINAL_P.bin deleted.")

    print(f"  [p_frame_extractor] Done.")


# ---------------------------------------------------------------------------
# Command-line entry point (for testing standalone)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python p_frame_extractor.py <video_path> <mv_output_dir> <video_key>")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2], sys.argv[3])
