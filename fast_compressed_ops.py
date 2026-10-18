"""
fast_compressed_ops.py
======================
Vectorized (NumPy) re-implementation of the compressed-domain front end.
Produces the SAME outputs as the original modules, without per-macroblock Python loops:

  original                                         vectorized here
  -----------------------------------------------  ------------------------------------
  2 FFmpeg passes (I-frame run + P-frame run)       1 FFmpeg pass writing both .bin files
  2 ffprobe -show_frames calls                      1 ffprobe call (I and P indices)
  feature_map.build_and_save_npy (2 file scans      build_iframe_feature_maps: one read,
     per I-frame, loop over every 4x4 block)           array ops per I-frame
  p_frame_extractor._parse_p_bin + _write_frame_npy parse_p_bin_fast: np.fromfile + cumsum
     (struct.unpack + median per macroblock)           frame split + row-wise median
  bbox_propagator.propagate_bbox (loop over blocks) propagate_bbox_fast: boolean masks
  disk round-trip of every .npy                     everything kept in memory (optional save)

The original files are not modified. verify_against_original() checks equivalence.
"""

import os
import subprocess

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(BASE_DIR, "FFmpeg", "FFmpeg", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(BASE_DIR, "FFmpeg", "FFmpeg", "ffprobe.exe")
if not os.path.exists(FFPROBE_PATH):
    FFPROBE_PATH = "ffprobe"

BIN_BASE = os.path.join(BASE_DIR, "STRICT_FINAL")
I_BIN_PATH = BIN_BASE + "_I.bin"
P_BIN_PATH = BIN_BASE + "_P.bin"

# I-frame coefficient record (feature_map.py): '<qiiii' header + 16*48*2 int16 coeffs + 3*16*2 int16 luma DC
I_RECORD = np.dtype([
    ("frame", "<i8"), ("mb_x", "<i4"), ("mb_y", "<i4"), ("mb_type", "<i4"), ("cbp", "<i4"),
    ("coeffs", "<i2", (16 * 48 * 2,)), ("luma_dc", "<i2", (3 * 16 * 2,)),
])
assert I_RECORD.itemsize == 3288

# P-frame MV record (p_frame_extractor.py): '<q i i 32h'
P_RECORD = np.dtype([("frame", "<i8"), ("mb_x", "<i4"), ("mb_y", "<i4"), ("mv", "<i2", (32,))])
assert P_RECORD.itemsize == 80

# Output dtype identical to p_frame_extractor.MV_DTYPE
MV_DTYPE = np.dtype([
    ("source", np.uint16), ("w", np.uint16), ("h", np.uint16),
    ("src_x", np.uint16), ("src_y", np.uint16), ("dst_x", np.uint16), ("dst_y", np.uint16),
    ("flags", np.uint16), ("motion_x", np.int16), ("motion_y", np.int16), ("motion_scale", np.uint16),
])

# DCT frequency bands and H.264 4x4 block Z-scan positions (feature_map.py)
LOW_IDX = [0, 1, 4]
MID_IDX = [2, 3, 5, 6, 8]
HIGH_IDX = [7, 9, 10, 11, 12, 13, 14, 15]
_BLK = np.arange(16)
BLOCK_COL = (_BLK % 2) + ((_BLK // 4) % 2) * 2
BLOCK_ROW = ((_BLK // 2) % 2) + ((_BLK // 8) % 2) * 2


# ============================================================
# BITSTREAM EXTRACTION
# ============================================================

def run_single_pass_extraction(video_path):
    """
    One run of the custom FFmpeg writes BOTH STRICT_FINAL_I.bin (coefficients,
    enabled by H264_COEFF_EXTRACT_FILE) and STRICT_FINAL_P.bin (motion vectors, cwd).

    -threads 1 is required: with FFmpeg's default frame-threaded decoding, several
    decoder threads write STRICT_FINAL_P.bin concurrently and the records interleave
    (non-deterministic file, ~1.5M bogus "frames" instead of one per P-frame).
    Single-threaded decoding is deterministic and, here, also faster (no write contention).

    The custom ffprobe build decodes frames too and therefore also (re)writes these .bin
    files: run every ffprobe call BEFORE this extraction, never between extraction and parsing.
    """
    for p in (I_BIN_PATH, P_BIN_PATH):
        if os.path.exists(p):
            os.remove(p)
    env = os.environ.copy()
    env["H264_COEFF_EXTRACT_FILE"] = BIN_BASE
    res = subprocess.run(
        [FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-threads", "1", "-y",
         "-i", video_path, "-an", "-f", "null", "NUL"],
        cwd=BASE_DIR, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if res.returncode != 0 or not os.path.exists(I_BIN_PATH) or not os.path.exists(P_BIN_PATH):
        raise RuntimeError(f"Custom FFmpeg extraction failed:\n{res.stderr[-2000:]}")
    return I_BIN_PATH, P_BIN_PATH


def get_frame_type_indices(video_path):
    """One ffprobe call -> (absolute I-frame indices, absolute P-frame indices), display order"""
    cmd = [FFPROBE_PATH, "-v", "quiet", "-select_streams", "v:0", "-show_frames",
           "-show_entries", "frame=pict_type", "-of", "csv=p=0", video_path]
    out = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    types = [line.strip() for line in out.splitlines()]
    return ([i for i, t in enumerate(types) if t == "I"],
            [i for i, t in enumerate(types) if t == "P"])


def cleanup_bins():
    for p in (I_BIN_PATH, P_BIN_PATH):
        if os.path.exists(p):
            os.remove(p)


# ============================================================
# I-FRAME DCT FEATURE MAPS  (== feature_map.build_and_save_npy)
# ============================================================

def build_iframe_feature_maps(i_bin_path, abs_i_indices):
    """
    Returns {absolute_frame_idx: (H/4, W/4, 3) float32 feature map}, identical to the
    .npy files written by i_frame_extractor.convert_bin_to_npy_files.
    """
    rec = np.fromfile(i_bin_path, dtype=I_RECORD)
    if rec.size == 0:
        return {}
    frame_nums = rec["frame"]
    # unique frame numbers in order of first appearance (== get_unique_frame_numbers)
    _, first_pos = np.unique(frame_nums, return_index=True)
    ordered = frame_nums[np.sort(first_pos)]

    maps = {}
    for seq_idx, frame_num in enumerate(ordered):
        r = rec[frame_nums == frame_num]
        mb_x = r["mb_x"].astype(np.int64)
        mb_y = r["mb_y"].astype(np.int64)
        grid_rows = (int(mb_y.max()) + 1) * 4
        grid_cols = (int(mb_x.max()) + 1) * 4
        feature = np.zeros((grid_rows, grid_cols, 3), dtype=np.float32)

        luma = r["coeffs"][:, :256].astype(np.float32).reshape(-1, 16, 16)  # (MB, block, coeff)
        sq = np.square(luma)
        rows = mb_y[:, None] * 4 + BLOCK_ROW[None, :]
        cols = mb_x[:, None] * 4 + BLOCK_COL[None, :]
        feature[rows, cols, 0] = np.log1p(sq[:, :, LOW_IDX].sum(axis=2))
        feature[rows, cols, 1] = np.log1p(sq[:, :, MID_IDX].sum(axis=2))
        feature[rows, cols, 2] = np.log1p(sq[:, :, HIGH_IDX].sum(axis=2))

        abs_frame = abs_i_indices[seq_idx] if seq_idx < len(abs_i_indices) else seq_idx
        maps[abs_frame] = feature
    return maps


# ============================================================
# P-FRAME MOTION VECTORS  (== p_frame_extractor._parse_p_bin + _write_frame_npy)
# ============================================================

def _rowwise_median_valid(values, valid):
    """
    Median of the valid entries of each row (float64), NaN for rows without valid
    entries. Same result as np.median(row[valid_row]).
    """
    v = values.astype(np.float64)
    v[~valid] = np.nan
    v.sort(axis=1)  # NaNs sorted to the end
    k = valid.sum(axis=1)
    lo = np.clip((k - 1) // 2, 0, v.shape[1] - 1)
    hi = np.clip(k // 2, 0, v.shape[1] - 1)
    med = (np.take_along_axis(v, lo[:, None], 1)[:, 0] + np.take_along_axis(v, hi[:, None], 1)[:, 0]) / 2
    med[k == 0] = np.nan
    return med


def parse_p_bin_fast(p_bin_path, abs_p_indices):
    """
    Returns {absolute_frame_idx: structured MV array (MV_DTYPE)} identical to the
    .npy files written by p_frame_extractor.process_video.
    """
    rec = np.fromfile(p_bin_path, dtype=P_RECORD)
    if rec.size == 0:
        return {}
    mb_x = rec["mb_x"].astype(np.int64)
    mb_y = rec["mb_y"].astype(np.int64)

    # A new frame starts at macroblock (0, 0); records before the first one are dropped
    seq = np.cumsum((mb_x == 0) & (mb_y == 0)) - 1
    keep = seq >= 0
    rec, mb_x, mb_y, seq = rec[keep], mb_x[keep], mb_y[keep], seq[keep]

    # Dominant MV = median over 16 sub-blocks of the "sane" (|v| <= 4096 qpel) vectors
    mv = rec["mv"]
    dx_vals, dy_vals = mv[:, 0::2], mv[:, 1::2]
    valid = (np.abs(dx_vals) <= 4096) & (np.abs(dy_vals) <= 4096)
    med_x = _rowwise_median_valid(dx_vals, valid)
    med_y = _rowwise_median_valid(dy_vals, valid)
    dx = np.nan_to_num(np.trunc(med_x), nan=0.0).astype(np.int64)  # int(np.median(...))
    dy = np.nan_to_num(np.trunc(med_y), nan=0.0).astype(np.int64)
    dx = np.clip(dx, -32768, 32767)
    dy = np.clip(dy, -32768, 32767)

    src_x = np.clip(mb_x * 16, 0, 65535)
    src_y = np.clip(mb_y * 16, 0, 65535)

    out = np.zeros(len(rec), dtype=MV_DTYPE)
    out["w"] = 16
    out["h"] = 16
    out["motion_scale"] = 4
    out["src_x"] = src_x
    out["src_y"] = src_y
    out["dst_x"] = np.clip(src_x + np.floor_divide(dx, 4), 0, 65535)
    out["dst_y"] = np.clip(src_y + np.floor_divide(dy, 4), 0, 65535)
    out["motion_x"] = dx
    out["motion_y"] = dy

    # Split into frames (records of a frame are contiguous)
    bounds = np.flatnonzero(np.diff(seq)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(seq)]))
    frames = {}
    for s, e in zip(starts, ends):
        seq_idx = int(seq[s])
        abs_frame = abs_p_indices[seq_idx] if seq_idx < len(abs_p_indices) else seq_idx
        frames[abs_frame] = out[s:e]
    return frames


# ============================================================
# BOX PROPAGATION  (== bbox_propagator.propagate_bbox)
# ============================================================

class FrameMV:
    """Pre-split columns of one P-frame's MV array for fast repeated box queries"""
    __slots__ = ("x1", "y1", "x2", "y2", "dx", "dy")

    def __init__(self, mv_array):
        scale = mv_array["motion_scale"].astype(np.float64)
        ok = scale != 0
        a = mv_array[ok]
        self.x1 = a["src_x"].astype(np.float64)
        self.y1 = a["src_y"].astype(np.float64)
        self.x2 = self.x1 + a["w"]
        self.y2 = self.y1 + a["h"]
        self.dx = a["motion_x"] / scale[ok]
        self.dy = a["motion_y"] / scale[ok]


def propagate_bbox_fast(prev_bbox, fmv):
    """Shift a box by the median motion of the moving blocks overlapping it"""
    x1, y1, x2, y2 = prev_bbox
    if fmv is None or fmv.dx.size == 0:
        return prev_bbox
    overlap = (fmv.x1 < x2) & (fmv.x2 > x1) & (fmv.y1 < y2) & (fmv.y2 > y1)
    moving = overlap & ((fmv.dx != 0) | (fmv.dy != 0))
    if not moving.any():
        return prev_bbox
    mdx = float(np.median(fmv.dx[moving]))
    mdy = float(np.median(fmv.dy[moving]))
    # H.264 MVs point backwards in time -> subtract to move the box forward
    return (x1 - mdx, y1 - mdy, x2 - mdx, y2 - mdy)


def propagate_timeline_fast(total_frames, key_boxes, p_mvs, width, height):
    """
    Same timeline logic as ActionDatasetPreprocessor.process_single_video:
    key frame -> detections replace boxes (if any); P-frame -> MV propagation + clamp;
    other frames -> hold.
    """
    boxes_by_frame = {}
    current = []
    for f_idx in range(total_frames):
        if f_idx in key_boxes:
            if key_boxes[f_idx]:
                current = list(key_boxes[f_idx])
        elif f_idx in p_mvs and current:
            fmv = p_mvs[f_idx] if isinstance(p_mvs[f_idx], FrameMV) else FrameMV(p_mvs[f_idx])
            current = [
                (max(0.0, min(float(width), b[0])), max(0.0, min(float(height), b[1])),
                 max(0.0, min(float(width), b[2])), max(0.0, min(float(height), b[3])))
                for b in (propagate_bbox_fast(bb, fmv) for bb in current)
            ]
        if current:
            boxes_by_frame[f_idx] = list(current)
    return boxes_by_frame


# ============================================================
# I-FRAME DETECTION  (== ActionDatasetPreprocessor.detect_on_i_frame, from memory)
# ============================================================

def detect_on_feature_map(detector, feat, width, height, conf_thresh=0.5, iou_thresh=0.4):
    """Same steps as detect_on_i_frame, but takes the feature map from memory"""
    from prepare_action_dataset import resize_feature_map, decode_predictions_with_scores, apply_nms
    feat = (feat - feat.mean()) / (feat.std() + 1e-8)
    x = np.expand_dims(resize_feature_map(feat, size=(300, 300)), axis=0)
    pred = np.asarray(detector(x, training=False))[0]
    boxes_300, scores = decode_predictions_with_scores(pred, conf_thresh=conf_thresh)
    boxes_300 = apply_nms(boxes_300, scores, iou_threshold=iou_thresh)
    return [
        (max(0.0, xmin * width / 300.0), max(0.0, ymin * height / 300.0),
         min(float(width), xmax * width / 300.0), min(float(height), ymax * height / 300.0))
        for (xmin, ymin, xmax, ymax) in boxes_300
    ]


# ============================================================
# EQUIVALENCE CHECK
# ============================================================

def verify_against_original(video_path, work_dir):
    """
    Runs the ORIGINAL parsers (feature_map / p_frame_extractor / bbox_propagator) and the
    vectorized ones on the SAME extracted .bin files and compares feature maps, MV arrays
    and propagated boxes.
    """
    import shutil
    import sys
    sys.path.insert(0, BASE_DIR)
    sys.path.insert(0, os.path.join(BASE_DIR, "H264_Compressed_Detector"))
    import i_frame_extractor as ife
    import p_frame_extractor as pfe
    from bbox_propagator import propagate_bbox

    shutil.rmtree(work_dir, ignore_errors=True)
    i_dir, p_dir = os.path.join(work_dir, "I"), os.path.join(work_dir, "P")
    os.makedirs(i_dir)
    os.makedirs(p_dir)

    abs_i, abs_p = get_frame_type_indices(video_path)
    assert abs_i == ife.get_i_frame_abs_indices(video_path)
    assert abs_p == pfe._get_p_frame_abs_indices(video_path, BASE_DIR)
    run_single_pass_extraction(video_path)

    print("Running ORIGINAL parsers ...")
    ife.convert_bin_to_npy_files(i_dir, abs_i)            # reads STRICT_FINAL_I.bin
    for seq_idx, recs in pfe._parse_p_bin(P_BIN_PATH).items():
        abs_frame = abs_p[seq_idx] if seq_idx < len(abs_p) else seq_idx
        pfe._write_frame_npy(recs, os.path.join(p_dir, f"{abs_frame}.npy"))

    print("Running VECTORIZED parsers ...")
    maps = build_iframe_feature_maps(I_BIN_PATH, abs_i)
    mvs = parse_p_bin_fast(P_BIN_PATH, abs_p)
    cleanup_bins()

    report = {"iframes_original": len(os.listdir(i_dir)), "iframes_fast": len(maps),
              "pframes_original": len(os.listdir(p_dir)), "pframes_fast": len(mvs)}
    max_feat_diff = 0.0
    for f in os.listdir(i_dir):
        a = np.load(os.path.join(i_dir, f))
        b = maps[int(f[:-4])]
        assert a.shape == b.shape, (f, a.shape, b.shape)
        max_feat_diff = max(max_feat_diff, float(np.abs(a - b).max()))
    report["feature_map_max_abs_diff"] = max_feat_diff

    mv_identical = True
    for f in os.listdir(p_dir):
        a = np.load(os.path.join(p_dir, f), allow_pickle=True)
        b = mvs[int(f[:-4])]
        if a.dtype != b.dtype or not np.array_equal(a, b):
            mv_identical = False
            break
    report["mv_arrays_identical"] = mv_identical

    # Propagation: original propagate_bbox vs fast, over the whole video from frame-0 box
    box = (300.0, 500.0, 700.0, 1500.0)
    worst = 0.0
    a_box = b_box = box
    for f in sorted(mvs):
        arr = np.load(os.path.join(p_dir, f"{f}.npy"), allow_pickle=True)
        a_box = propagate_bbox(a_box, arr)
        b_box = propagate_bbox_fast(b_box, FrameMV(mvs[f]))
        worst = max(worst, max(abs(p - q) for p, q in zip(a_box, b_box)))
    report["propagation_max_abs_diff_px"] = worst
    shutil.rmtree(work_dir, ignore_errors=True)
    return report


if __name__ == "__main__":
    import json
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "data", "videos", "Walking.mp4")
    print(json.dumps(verify_against_original(video, os.path.join(BASE_DIR, "_verify_fast_ops")), indent=2))
