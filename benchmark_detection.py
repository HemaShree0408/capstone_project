"""
benchmark_detection.py
======================
Computational comparison: compressed-domain detection (ours) vs RGB YOLO.

Pipelines measured on the same video(s), same machine:

  [A] OURS        custom FFmpeg -> I-frame DCT feature maps + P-frame MVs
                  -> SSD300_H264 on I-frames only -> MV box propagation to P/B frames
  [B] YOLO-FULL   full RGB decode of every frame -> YOLO on every frame
                  (the conventional pixel-domain pipeline)
  [C] YOLO-KEY    RGB decode of key frames only -> YOLO on key frames
                  -> the SAME MV propagation as ours (ablation: separates the
                  benefit of the DCT input from the benefit of sparse detection)

Metrics:
  * Model complexity : parameters, GFLOPs per inference (same counting rule for
                       both models: 2 x multiply-accumulates of conv + dense layers),
                       weight-file size, model-only latency (batch 1)
  * End-to-end       : wall time per stage, total time, FPS, ms/frame,
                       real-time factor, concurrent real-time streams
  * Compute          : detector calls, total detector GFLOPs, GFLOPs per video second,
                       CPU-seconds (incl. FFmpeg child processes), extra peak RAM,
                       GPU peak memory and energy (if available)
  * Data volume      : bytes handed to the vision stack (DCT maps + MVs vs RGB frames)
  * Accuracy         : precision / recall / F1 @ IoU 0.5 and mean IoU against the
                       ground-truth <video>.txt files (if present)
  * Cost (optional)  : USD per hour of video for a given $/hour machine price

Outputs (results/benchmark/):
  report.md                 paper-ready tables + ratios + caveats
  summary.csv               one row per (video, pipeline, device)
  model_complexity.csv
  raw_results.json          everything that was measured
  fig_*.png                 bar charts

Requirements (in addition to the project venv):
  pip install ultralytics psutil
  pip install pynvml        (optional: GPU power / energy)

Usage:
  python benchmark_detection.py --videos data/videos/Walking.mp4 data/videos/Clapping.mp4 data/videos/Meet1.mp4
  python benchmark_detection.py --videos data/videos/Walking.mp4 --yolo_models models/yolo11n.pt yolov8n.pt yolo11s.pt
  python benchmark_detection.py --videos data/videos/Walking.mp4 --yolo_devices cpu cuda --runs 3
"""

import os
import sys
import re
import csv
import copy
import json
import time
import shutil
import argparse
import platform
import threading
import subprocess
from collections import defaultdict

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Existing modules print symbols like "✓"; a Windows cp1252 console would crash on them
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = os.path.join(BASE_DIR, "H264_Compressed_Detector")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DETECTOR_DIR)

import torch  # imported before TensorFlow (GPU memory growth set above)

# Existing project code (used as-is)
from prepare_action_dataset import ActionDatasetPreprocessor, get_video_frame_count
import i_frame_extractor as ife
import p_frame_extractor as pfe
from bbox_propagator import propagate_bbox
import tensorflow as tf
import fast_compressed_ops as fco


# ============================================================
# MEASUREMENT HELPERS
# ============================================================

class StageTimer:
    """Accumulates wall-clock time per named stage"""

    def __init__(self):
        self.stages = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self_):
                self_.t0 = time.perf_counter()

            def __exit__(self_, *exc):
                timer.stages[name] = timer.stages.get(name, 0.0) + time.perf_counter() - self_.t0

        return _Ctx()

    @property
    def total(self):
        return sum(self.stages.values())


class ResourceMonitor:
    """
    Samples (in a background thread) while a pipeline runs:
      - peak RSS of this process + children (FFmpeg), relative to the start
      - CPU-seconds of this process + children
      - GPU power (NVML) -> energy in joules
    """

    def __init__(self, interval=0.05, use_gpu=False):
        self.interval = interval
        self.use_gpu = use_gpu and pynvml is not None
        self.proc = psutil.Process() if psutil else None

    def __enter__(self):
        self._stop = threading.Event()
        self.base_rss = self._rss() if self.proc else 0
        self.peak_rss = self.base_rss
        self.child_cpu = {}
        self.power = []
        self.cpu0 = self.proc.cpu_times() if self.proc else None
        self.t0 = time.perf_counter()
        self._nvml = None
        if self.use_gpu:
            try:
                pynvml.nvmlInit()
                self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _rss(self):
        total = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            if self.proc:
                try:
                    self.peak_rss = max(self.peak_rss, self._rss())
                    for child in self.proc.children(recursive=True):
                        try:
                            ct = child.cpu_times()
                            self.child_cpu[child.pid] = ct.user + ct.system
                        except Exception:
                            pass
                except Exception:
                    pass
            if self._nvml is not None:
                try:
                    self.power.append((time.perf_counter(),
                                       pynvml.nvmlDeviceGetPowerUsage(self._nvml) / 1000.0))
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.wall = time.perf_counter() - self.t0
        if self.proc:
            cpu1 = self.proc.cpu_times()
            own = (cpu1.user - self.cpu0.user) + (cpu1.system - self.cpu0.system)
            self.cpu_seconds = own + sum(self.child_cpu.values())
            self.extra_peak_rss_mb = (self.peak_rss - self.base_rss) / 1024 ** 2
        else:
            self.cpu_seconds = None
            self.extra_peak_rss_mb = None
        if len(self.power) >= 2:
            t = np.array([p[0] for p in self.power])
            w = np.array([p[1] for p in self.power])
            self.energy_j = float(np.sum((w[1:] + w[:-1]) / 2 * np.diff(t)))
            self.avg_power_w = float(w.mean())
        else:
            self.energy_j = None
            self.avg_power_w = None


def latency_stats(times_s):
    a = np.array(times_s) * 1000.0
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "std_ms": float(a.std()), "n": int(len(a))}


# ============================================================
# MODEL COMPLEXITY
# ============================================================

def keras_flops(model):
    """2 x MACs of every Conv2D and Dense layer (same rule as torch_flops)"""
    flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, c_out = layer.output.shape
            kh, kw = layer.kernel_size
            c_in = layer.input.shape[-1]
            flops += 2 * h * w * c_out * kh * kw * c_in // layer.groups
        elif isinstance(layer, tf.keras.layers.Dense):
            flops += 2 * layer.input.shape[-1] * layer.units
    return int(flops)


def torch_flops(module, input_tensor):
    """2 x MACs of every Conv2d and Linear layer, counted with forward hooks"""
    total = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += 2 * out.numel() // out.shape[0] * (m.in_channels // m.groups) * k

    def linear_hook(m, inp, out):
        total[0] += 2 * m.in_features * m.out_features * (out.numel() // out.shape[0] // m.out_features)

    hooks = []
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    with torch.inference_mode():
        module(input_tensor)
    for h in hooks:
        h.remove()
    return int(total[0])


def bench_ssd_model(model, reps, warmup=10):
    """Model-only latency of SSD300_H264 (batch 1, graph mode, no I/O)"""
    x = tf.constant(np.random.randn(1, 300, 300, 3).astype(np.float32))
    fn = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup):
        fn(x).numpy()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x).numpy()
        times.append(time.perf_counter() - t0)
    return latency_stats(times)


def bench_yolo_model(yolo, device, imgsz, reps, warmup=10):
    """Model-only latency of the YOLO network (batch 1, letterboxed size, no pre/post-processing)"""
    yolo.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)
    net = yolo.predictor.model  # fused AutoBackend used by predict()
    dtype = torch.float16 if getattr(net, "fp16", False) else torch.float32
    x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        sync()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            net(x)
            sync()
            times.append(time.perf_counter() - t0)
    return latency_stats(times)


# ============================================================
# VIDEO / GT HELPERS
# ============================================================

def ffmpeg_frames(video_path, width, height, keyframes_only=False):
    """
    Decodes the video to raw BGR frames with a STOCK FFmpeg (imageio-ffmpeg's binary),
    so the custom coefficient-dumping build is not involved in the RGB baselines.
    """
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-noautorotate"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-i", video_path, "-an", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=frame_bytes * 4)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def get_keyframe_indices(video_path):
    """Display-order indices of key frames (what -skip_frame nokey decodes)"""
    cmd = [ife.FFPROBE_PATH, "-v", "quiet", "-select_streams", "v:0", "-show_frames",
           "-show_entries", "frame=key_frame", "-of", "csv=p=0", video_path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return [i for i, line in enumerate(out.splitlines()) if line.strip().startswith("1")]


def load_gt(txt_path, width, height):
    """Ground truth: '# resolution: W,H' then frame,track_id,class,x1_norm,y1_norm,x2_norm,y2_norm"""
    gt = defaultdict(list)
    res_w, res_h = width, height
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.search(r"resolution:\s*(\d+)\s*,\s*(\d+)", line)
                if m:
                    res_w, res_h = int(m.group(1)), int(m.group(2))
                continue
            parts = line.split(",")
            if len(parts) < 7 or not parts[0].strip().isdigit():
                continue
            x1, y1, x2, y2 = (float(v) for v in parts[3:7])
            gt[int(parts[0])].append((x1 * res_w, y1 * res_h, x2 * res_w, y2 * res_h))
    return dict(gt)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate_boxes(pred_by_frame, gt_by_frame, iou_thr=0.5):
    """Greedy one-to-one matching per GT frame -> P / R / F1 @ IoU and mean IoU of matches"""
    tp = fp = fn = 0
    ious = []
    for f, gts in gt_by_frame.items():
        preds = list(pred_by_frame.get(f, []))
        pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
                       reverse=True)
        used_p, used_g = set(), set()
        for v, i, j in pairs:
            if v < iou_thr:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            ious.append(v)
        tp += len(used_g)
        fp += len(preds) - len(used_p)
        fn += len(gts) - len(used_g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "gt_frames": len(gt_by_frame)}


def propagate_timeline(total_frames, key_boxes, p_frame_files, width, height):
    """
    Same box timeline logic as ActionDatasetPreprocessor.process_single_video:
    key frame -> detections replace boxes (if any); P-frame -> MV propagation + clamp;
    other frames (B / no MV) -> hold.
    """
    boxes_by_frame = {}
    current = []
    for f_idx in range(total_frames):
        if f_idx in key_boxes:
            if key_boxes[f_idx]:
                current = list(key_boxes[f_idx])
        elif f_idx in p_frame_files and current:
            mv_array = np.load(p_frame_files[f_idx], allow_pickle=True)
            propagated = []
            for bbox in current:
                nb = propagate_bbox(bbox, mv_array)
                propagated.append((
                    max(0.0, min(float(width), nb[0])), max(0.0, min(float(height), nb[1])),
                    max(0.0, min(float(width), nb[2])), max(0.0, min(float(height), nb[3])),
                ))
            current = propagated
        if current:
            boxes_by_frame[f_idx] = list(current)
    return boxes_by_frame


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) if os.path.isdir(path) else 0


# ============================================================
# PIPELINES
# ============================================================

def run_ours(video_path, work_dir, preprocessor, width, height, total_frames):
    """[A] Compressed-domain pipeline, timed stage by stage (existing functions, unchanged)"""
    i_dir = os.path.join(work_dir, "I_frames")
    mv_dir = os.path.join(work_dir, "MVS")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(i_dir)
    os.makedirs(mv_dir)
    key = os.path.splitext(os.path.basename(video_path))[0]

    timer = StageTimer()
    det_times = []
    with ResourceMonitor() as mon:
        with timer("1_ffprobe_frame_types"):
            abs_i = ife.get_i_frame_abs_indices(video_path)
        with timer("2_iframe_coeff_extraction_ffmpeg"):
            ok = ife.run_ffmpeg_on_video(video_path)
        if not ok:
            raise RuntimeError("Custom FFmpeg did not produce I-frame coefficients")
        with timer("3_dct_feature_map_build"):
            ife.convert_bin_to_npy_files(i_dir, abs_i)
        ife.clear_bin()

        # Same steps as p_frame_extractor.process_video, timed separately so codec cost
        # and Python parsing overhead can be told apart (module functions used unchanged)
        project_dir = BASE_DIR
        ffmpeg_exe = os.path.join(project_dir, "FFmpeg", "FFmpeg", "ffmpeg.exe")
        bin_path = os.path.join(project_dir, "STRICT_FINAL_P.bin")
        with timer("4a_mv_extraction_ffmpeg"):
            # -threads 1: with frame threading the custom build interleaves MV records
            # (corrupted, non-deterministic STRICT_FINAL_P.bin) - see fast_compressed_ops.py
            res = subprocess.run([ffmpeg_exe, "-hide_banner", "-loglevel", "info", "-threads", "1", "-y", "-i", video_path,
                                  "-an", "-f", "null", "NUL"], cwd=project_dir, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0 or not os.path.isfile(bin_path):
            raise RuntimeError(f"Custom FFmpeg MV extraction failed:\n{res.stderr[-2000:]}")
        with timer("4b_ffprobe_p_frame_index"):
            abs_p = pfe._get_p_frame_abs_indices(video_path, project_dir)
        with timer("4c_mv_bin_parse_python"):
            mv_frames = pfe._parse_p_bin(bin_path)
        with timer("4d_mv_npy_write_python"):
            for seq_idx in sorted(mv_frames):
                abs_frame = abs_p[seq_idx] if seq_idx < len(abs_p) else seq_idx
                pfe._write_frame_npy(mv_frames[seq_idx], os.path.join(mv_dir, f"{abs_frame}.npy"))
        del mv_frames
        os.remove(bin_path)

        i_files = {int(f[:-4]): os.path.join(i_dir, f) for f in os.listdir(i_dir) if f.endswith(".npy")}
        p_files = {int(f[:-4]): os.path.join(mv_dir, f) for f in os.listdir(mv_dir) if f.endswith(".npy")}

        key_boxes = {}
        with timer("5_ssd_detection_iframes"):
            for f_idx in sorted(i_files):
                t0 = time.perf_counter()
                key_boxes[f_idx] = preprocessor.detect_on_i_frame(i_files[f_idx], width, height)
                det_times.append(time.perf_counter() - t0)
        with timer("6_mv_box_propagation"):
            boxes = propagate_timeline(total_frames, key_boxes, p_files, width, height)

    feat_shape = list(np.load(next(iter(i_files.values()))).shape) if i_files else None
    return {
        "stages_s": timer.stages,
        "total_s": timer.total,
        "detector_calls": len(i_files),
        "detector_call_latency": latency_stats(det_times) if det_times else None,
        "data_bytes": dir_bytes(i_dir) + dir_bytes(mv_dir),
        "data_detail": {"dct_feature_maps_bytes": dir_bytes(i_dir), "motion_vectors_bytes": dir_bytes(mv_dir),
                        "dct_feature_map_shape": feat_shape, "num_p_frames": len(p_files)},
        "cpu_seconds": mon.cpu_seconds,
        "extra_peak_rss_mb": mon.extra_peak_rss_mb,
        "energy_j": None,
        "gpu_peak_mem_mb": None,
        "boxes": boxes,
        "_p_files": p_files,
    }


def run_ours_fast(video_path, detector, width, height, total_frames, conf_thresh):
    """
    [A'] Same method as run_ours, with the vectorized front end (fast_compressed_ops.py):
    one single-threaded FFmpeg pass, one ffprobe call, NumPy parsing, in-memory data,
    vectorized propagation. Outputs are identical to the original code (verified).
    """
    timer = StageTimer()
    det_times = []
    with ResourceMonitor() as mon:
        with timer("1_ffprobe_frame_types"):
            abs_i, abs_p = fco.get_frame_type_indices(video_path)
        with timer("2_single_pass_extraction_ffmpeg"):
            i_bin, p_bin = fco.run_single_pass_extraction(video_path)
        with timer("3_dct_feature_map_build_numpy"):
            maps = fco.build_iframe_feature_maps(i_bin, abs_i)
        with timer("4_mv_parse_numpy"):
            mvs = fco.parse_p_bin_fast(p_bin, abs_p)
        fco.cleanup_bins()

        key_boxes = {}
        with timer("5_ssd_detection_iframes"):
            for f_idx in sorted(maps):
                t0 = time.perf_counter()
                key_boxes[f_idx] = fco.detect_on_feature_map(detector, maps[f_idx], width, height, conf_thresh)
                det_times.append(time.perf_counter() - t0)
        with timer("6_mv_box_propagation"):
            boxes = fco.propagate_timeline_fast(total_frames, key_boxes, mvs, width, height)

    feat_bytes = sum(m.nbytes for m in maps.values())
    mv_bytes = sum(m.nbytes for m in mvs.values())
    return {
        "stages_s": timer.stages,
        "total_s": timer.total,
        "detector_calls": len(maps),
        "detector_call_latency": latency_stats(det_times) if det_times else None,
        "data_bytes": feat_bytes + mv_bytes,
        "data_detail": {"dct_feature_maps_bytes": feat_bytes, "motion_vectors_bytes": mv_bytes,
                        "dct_feature_map_shape": list(next(iter(maps.values())).shape) if maps else None,
                        "num_p_frames": len(mvs)},
        "cpu_seconds": mon.cpu_seconds,
        "extra_peak_rss_mb": mon.extra_peak_rss_mb,
        "energy_j": None,
        "gpu_peak_mem_mb": None,
        "boxes": boxes,
        "_p_mvs": mvs,
    }


def yolo_detect(yolo, frame, imgsz, conf, device):
    res = yolo.predict(frame, imgsz=imgsz, conf=conf, classes=[0], device=device, verbose=False)[0]
    return [tuple(b) for b in res.boxes.xyxy.cpu().numpy().tolist()], res.speed


def run_yolo_full(video_path, yolo, device, imgsz, conf, width, height):
    """[B] Decode every frame to RGB + YOLO on every frame"""
    timer = StageTimer()
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    with ResourceMonitor(use_gpu=is_cuda) as mon:
        # Decode cost alone (decode runs in a separate process, so it is measured on its own pass)
        with timer("1_rgb_decode_all_frames"):
            n_decoded = sum(1 for _ in ffmpeg_frames(video_path, width, height))

        # Detection on every frame (decode overlaps with inference here)
        boxes, det_times, speeds = {}, [], defaultdict(float)
        t_e2e = time.perf_counter()
        for f_idx, frame in enumerate(ffmpeg_frames(video_path, width, height)):
            t0 = time.perf_counter()
            boxes[f_idx], sp = yolo_detect(yolo, frame, imgsz, conf, device)
            det_times.append(time.perf_counter() - t0)
            for k, v in sp.items():
                speeds[k] += v / 1000.0
        overlapped_e2e = time.perf_counter() - t_e2e
        timer.stages["2_yolo_detection_all_frames"] = float(np.sum(det_times))

    return {
        "stages_s": timer.stages,
        "total_s": timer.total,
        "overlapped_decode_plus_detect_s": overlapped_e2e,
        "detector_calls": len(det_times),
        "detector_call_latency": latency_stats(det_times),
        "yolo_internal_breakdown_s": dict(speeds),
        "data_bytes": n_decoded * width * height * 3,
        "data_detail": {"decoded_rgb_frames": n_decoded, "rgb_frame_bytes": width * height * 3},
        "cpu_seconds": mon.cpu_seconds,
        "extra_peak_rss_mb": mon.extra_peak_rss_mb,
        "energy_j": mon.energy_j,
        "gpu_peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if is_cuda else None,
        "boxes": boxes,
    }


def run_yolo_keyframes(video_path, yolo, device, imgsz, conf, width, height, total_frames,
                       p_files, mv_extraction_s):
    """
    [C] Decode key frames only + YOLO on them + the same MV propagation as ours.
    mv_extraction_s: {stage_name: seconds} of the MV extraction steps measured in [A]
    """
    timer = StageTimer()
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    with ResourceMonitor(use_gpu=is_cuda) as mon:
        with timer("1_ffprobe_keyframe_index"):
            key_idx = get_keyframe_indices(video_path)
        with timer("2_rgb_decode_keyframes"):
            key_frames = list(ffmpeg_frames(video_path, width, height, keyframes_only=True))
        if len(key_frames) != len(key_idx):
            print(f"  [warn] {len(key_frames)} decoded key frames vs {len(key_idx)} indexed; using the shorter")
        key_boxes, det_times = {}, []
        with timer("3_yolo_detection_keyframes"):
            for f_idx, frame in zip(key_idx, key_frames):
                t0 = time.perf_counter()
                key_boxes[f_idx], _ = yolo_detect(yolo, frame, imgsz, conf, device)
                det_times.append(time.perf_counter() - t0)
        # MV extraction is identical to ours -> reuse its measured time and files
        for k, v in mv_extraction_s.items():
            timer.stages[f"{k} (from A)"] = v
        in_memory = bool(p_files) and not isinstance(next(iter(p_files.values())), str)
        with timer("5_mv_box_propagation"):
            if in_memory:
                boxes = fco.propagate_timeline_fast(total_frames, key_boxes, p_files, width, height)
            else:
                boxes = propagate_timeline(total_frames, key_boxes, p_files, width, height)

    n_key = len(det_times)
    mv_bytes = sum(v.nbytes if hasattr(v, "nbytes") else os.path.getsize(v) for v in p_files.values())
    return {
        "stages_s": timer.stages,
        "total_s": timer.total,
        "detector_calls": n_key,
        "detector_call_latency": latency_stats(det_times) if det_times else None,
        "data_bytes": n_key * width * height * 3 + mv_bytes,
        "data_detail": {"decoded_rgb_keyframes": n_key},
        "cpu_seconds": mon.cpu_seconds,
        "extra_peak_rss_mb": mon.extra_peak_rss_mb,
        "energy_j": mon.energy_j,
        "gpu_peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if is_cuda else None,
        "boxes": boxes,
    }


# ============================================================
# REPORTING
# ============================================================

def derive_metrics(r, video_meta, gflops_per_call, price_per_hour):
    """Adds throughput / compute / cost metrics to a pipeline result"""
    frames = video_meta["frames"]
    duration = video_meta["duration_s"]
    total = r["total_s"]
    r["fps"] = frames / total if total > 0 else None
    r["ms_per_frame"] = 1000.0 * total / frames
    r["realtime_factor"] = duration / total if total > 0 else None
    r["realtime_streams"] = r["fps"] / video_meta["fps"] if r["fps"] else None
    r["detector_gflops_total"] = r["detector_calls"] * gflops_per_call
    r["detector_gflops_per_video_second"] = r["detector_gflops_total"] / duration
    r["detector_calls_per_100_frames"] = 100.0 * r["detector_calls"] / frames
    r["compute_seconds_per_video_hour"] = total / duration * 3600.0
    r["usd_per_video_hour"] = (r["compute_seconds_per_video_hour"] / 3600.0 * price_per_hour
                               if price_per_hour else None)
    return r


STAGE_CATEGORIES = [
    ("Codec / bitstream (FFmpeg, ffprobe)", ("ffmpeg", "ffprobe", "rgb_decode")),
    ("Python data handling (prototype overhead)", ("feature_map_build", "_python", "_numpy")),
    ("Detector (model + NMS + pre/post)", ("detection",)),
    ("MV box tracking", ("propagation",)),
]


def stage_categories(stages):
    """Groups stage times into codec / Python overhead / detector / tracking"""
    out = {name: 0.0 for name, _ in STAGE_CATEGORIES}
    for stage, secs in stages.items():
        for name, keys in STAGE_CATEGORIES:
            if any(k in stage for k in keys):
                out[name] += secs
                break
    return out


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


def ratio(a, b):
    return a / b if a and b else None


def make_figures(out_dir, rows, complexity):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figs = []

    for video in sorted({r["video"] for r in rows}):
        vr = [r for r in rows if r["video"] == video]
        labels = [f"{r['pipeline']}\n({r['device']})" for r in vr]

        # Stacked stage times
        fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(vr)), 5))
        bottoms = np.zeros(len(vr))
        stage_names = []
        for r in vr:
            for s in r["stages_s"]:
                if s not in stage_names:
                    stage_names.append(s)
        cmap = plt.get_cmap("tab20")
        for k, s in enumerate(stage_names):
            vals = np.array([r["stages_s"].get(s, 0.0) for r in vr])
            if vals.sum() == 0:
                continue
            ax.bar(labels, vals, bottom=bottoms, label=s, color=cmap(k % 20))
            bottoms += vals
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"End-to-end time by stage - {video}")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig_stage_time_{video}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs.append(path)

        # Detector GFLOPs per video second
        fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(vr)), 4))
        vals = [r["detector_gflops_per_video_second"] for r in vr]
        ax.bar(labels, vals, color=["#2a9d8f" if r["pipeline"].startswith("OURS") else "#e76f51" for r in vr])
        ax.set_yscale("log")
        ax.set_ylabel("Detector GFLOPs per second of video (log)")
        ax.set_title(f"Detection compute - {video}")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig_gflops_{video}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs.append(path)

    # Model-only latency
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, vals = [], []
    for c in complexity:
        for dev, stats in c["latency"].items():
            labels.append(f"{c['model']}\n{dev}")
            vals.append(stats["median_ms"])
    ax.bar(labels, vals, color=["#2a9d8f" if "SSD300_H264" in l else "#e76f51" for l in labels])
    ax.set_ylabel("Median latency per inference (ms, batch 1)")
    ax.set_title("Model-only inference latency")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_model_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)
    return figs


def write_report(out_dir, env, complexity, rows, args):
    lines = ["# Compressed-domain detection vs RGB YOLO - computational benchmark", ""]
    lines += ["## Environment", ""]
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## 1. Model complexity", "",
              "| Model | Input | Params (M) | GFLOPs / inference | Weights file (MB) | Median latency (ms) |",
              "|---|---|---|---|---|---|"]
    for c in complexity:
        lat = ", ".join(f"{d}: {s['median_ms']:.2f}" for d, s in c["latency"].items())
        lines.append(f"| {c['model']} | {c['input']} | {c['params_m']:.2f} | {c['gflops']:.2f} | "
                     f"{fmt(c['weights_mb'])} | {lat} |")
    lines += ["", "GFLOPs = 2 x multiply-accumulates of all convolution and dense layers, counted the "
              "same way for both models.", ""]

    lines += ["## 2. End-to-end results", ""]
    for video in sorted({r["video"] for r in rows}):
        vr = [r for r in rows if r["video"] == video]
        meta = vr[0]["video_meta"]
        lines += [f"### {video}  ({meta['width']}x{meta['height']}, {meta['frames']} frames, "
                  f"{meta['fps']:.2f} fps, {meta['duration_s']:.1f} s, "
                  f"bitstream {meta['bitstream_mb']:.2f} MB)", "",
                  "| Pipeline | Device | Detector calls | Total time (s) | FPS | ms/frame | Real-time x | "
                  "RT streams | Detector GFLOPs (video) | GFLOPs / video-s | Data to vision stack (MB) | "
                  "CPU-s | Extra RAM (MB) | GPU mem (MB) | Energy (J) | F1@0.5 | mIoU |",
                  "|" + "---|" * 17]
        for r in vr:
            acc = r.get("accuracy") or {}
            lines.append(
                f"| {r['pipeline']} | {r['device']} | {r['detector_calls']} | {fmt(r['total_s'])} | "
                f"{fmt(r['fps'])} | {fmt(r['ms_per_frame'])} | {fmt(r['realtime_factor'])} | "
                f"{fmt(r['realtime_streams'])} | {fmt(r['detector_gflops_total'])} | "
                f"{fmt(r['detector_gflops_per_video_second'])} | {fmt(r['data_bytes'] / 1024 ** 2)} | "
                f"{fmt(r['cpu_seconds'])} | {fmt(r['extra_peak_rss_mb'])} | {fmt(r['gpu_peak_mem_mb'])} | "
                f"{fmt(r['energy_j'])} | {fmt(acc.get('f1'), 3)} | {fmt(acc.get('mean_iou'), 3)} |")
        if args.price_per_hour:
            lines += ["", f"Cost at ${args.price_per_hour}/machine-hour: " + ", ".join(
                f"{r['pipeline']} ({r['device']}) = ${r['usd_per_video_hour']:.4f} per video-hour" for r in vr)]

        lines += ["", "**Stage breakdown (s)**", ""]
        for r in vr:
            stages = ", ".join(f"{k} = {v:.3f}" for k, v in r["stages_s"].items())
            lines.append(f"- {r['pipeline']} ({r['device']}): {stages}")

        lines += ["", "**Time by cost category (s)**", "",
                  "| Pipeline | Device | " + " | ".join(n for n, _ in STAGE_CATEGORIES) + " |",
                  "|---|---|" + "---|" * len(STAGE_CATEGORIES)]
        for r in vr:
            cats = stage_categories(r["stages_s"])
            lines.append(f"| {r['pipeline']} | {r['device']} | " +
                         " | ".join(fmt(cats[n], 3) for n, _ in STAGE_CATEGORIES) + " |")

        if any((r.get("accuracy") or {}).get("f1", 0) >= 0.99 for r in vr if not r["pipeline"].startswith("OURS")):
            lines += ["", "> **Ground-truth caveat:** a YOLO baseline matches the ground truth almost perfectly "
                      "(F1 >= 0.99), which indicates the annotations were produced by YOLO itself. For this "
                      "video the F1 / mIoU columns measure *agreement with YOLO*, not true accuracy; use "
                      "human-annotated boxes for accuracy claims."]

        ours = next((r for r in vr if r["pipeline"].startswith("OURS-VEC")), None) or             next((r for r in vr if r["pipeline"].startswith("OURS")), None)
        if ours:
            lines += ["", f"**{ours['pipeline']} relative to each other row** (>1 = ours better)", "",
                      "| Baseline | Device | Speed-up (time) | Detector-FLOPs reduction | Detector-call reduction | "
                      "Data-volume reduction | CPU-s reduction | F1 difference (ours - baseline) |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in vr:
                if r is ours:
                    continue
                f1_o = (ours.get("accuracy") or {}).get("f1")
                f1_b = (r.get("accuracy") or {}).get("f1")
                lines.append(
                    f"| {r['pipeline']} | {r['device']} | {fmt(ratio(r['total_s'], ours['total_s']))}x | "
                    f"{fmt(ratio(r['detector_gflops_total'], ours['detector_gflops_total']))}x | "
                    f"{fmt(ratio(r['detector_calls'], ours['detector_calls']))}x | "
                    f"{fmt(ratio(r['data_bytes'], ours['data_bytes']))}x | "
                    f"{fmt(ratio(r['cpu_seconds'], ours['cpu_seconds']))}x | "
                    f"{fmt(f1_o - f1_b, 3) if f1_o is not None and f1_b is not None else '-'} |")
        lines.append("")

    lines += ["## 3. Notes for interpreting the numbers", "",
              "- *Total time* of OURS is the sum of its sequential stages. For YOLO-FULL it is RGB decode "
              "(measured on its own pass) + YOLO time; the pipelined wall time with decode overlapping "
              "inference is in raw_results.json (`overlapped_decode_plus_detect_s`).",
              "- The custom FFmpeg dumps coefficients/MVs right after entropy decoding and skips pixel "
              "reconstruction of I-frames, but P/B-frames are still fully reconstructed (the original code "
              "also decodes the video twice: one pass for I-frame coefficients, one for MVs; OURS-VEC "
              "decodes once). Skipping P/B reconstruction too would make extraction cheaper; report the "
              "measured times and present that saving as potential, not achieved.",
              "- OURS (original code) parses the binaries with per-macroblock Python loops; OURS-VEC "
              "(fast_compressed_ops.py) does the same computation with NumPy and produces identical feature "
              "maps, motion vectors and boxes (verified bit-exact). The difference between the two rows is "
              "implementation overhead, not a property of the method.",
              "- The custom FFmpeg must run with `-threads 1`: with frame-threaded decoding its MV dump "
              "interleaves records from several threads (corrupted, non-deterministic STRICT_FINAL_P.bin). "
              "Both OURS rows use -threads 1 for the MV pass.",
              "- YOLO-KEY uses the same detection schedule and MV tracking as OURS, so OURS vs YOLO-KEY "
              "isolates the effect of the DCT input; OURS vs YOLO-FULL shows the gain of the full approach.",
              "- Compare pipelines on the same device. TensorFlow has no native-Windows GPU support "
              "(TF >= 2.11), so on Windows the SSD runs on CPU; compare it with the YOLO CPU rows, or "
              "run under WSL2/Linux for GPU-vs-GPU numbers.",
              "- Detector FLOPs exclude NMS, pre- and post-processing; those are inside the measured times.",
              ""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compressed-domain SSD vs RGB YOLO computational benchmark")
    parser.add_argument("--videos", nargs="+", default=[os.path.join(BASE_DIR, "data", "videos", "Walking.mp4")])
    parser.add_argument("--gt", nargs="*", default=None,
                        help="Ground-truth txt per video (default: <video>.txt next to the video or in data/annotations/)")
    parser.add_argument("--detector_weights", default=os.path.join(BASE_DIR, "models", "best_model.h5"))
    parser.add_argument("--ssd_conf", type=float, default=0.5)
    parser.add_argument("--yolo_models", nargs="+", default=[os.path.join(BASE_DIR, "models", "yolo11n.pt")],
                        help="Ultralytics weights, e.g. yolo11n.pt yolov8n.pt yolo11s.pt (auto-downloaded)")
    parser.add_argument("--yolo_devices", nargs="+", default=None,
                        help="Devices for YOLO, e.g. cpu cuda (default: cpu, plus cuda if available)")
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--skip_keyframe_baseline", action="store_true", help="Do not run [C] YOLO-KEY")
    parser.add_argument("--impl", choices=["original", "fast", "both"], default="both",
                        help="Our pipeline with the original Python front end, the vectorized one, or both")
    parser.add_argument("--runs", type=int, default=1, help="Repeat each pipeline; the median-time run is kept")
    parser.add_argument("--micro_reps", type=int, default=100, help="Repetitions for model-only latency")
    parser.add_argument("--price_per_hour", type=float, default=None,
                        help="Optional machine price (USD/hour) to report cost per video-hour")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "results", "benchmark"))
    parser.add_argument("--keep_work", action="store_true", help="Keep extracted feature/MV files")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is required for the YOLO baselines:  pip install ultralytics psutil")
    if psutil is None:
        print("[warn] psutil not installed: CPU-seconds and RAM will not be reported (pip install psutil)")

    yolo_devices = args.yolo_devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    os.makedirs(args.output_dir, exist_ok=True)

    tf_gpus = tf.config.list_physical_devices("GPU")
    ssd_device = "GPU (TensorFlow)" if tf_gpus else "cpu"
    env = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor() or platform.machine(),
        "CPU cores (logical)": os.cpu_count(),
        "RAM (GB)": f"{psutil.virtual_memory().total / 1024 ** 3:.1f}" if psutil else "-",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "TensorFlow": f"{tf.__version__} (SSD device: {ssd_device})",
        "PyTorch": torch.__version__,
        "Python": platform.python_version(),
    }
    print("Environment:", json.dumps(env, indent=2))

    # ---------------- Models & complexity ----------------
    print("\nLoading SSD300_H264 ...")
    preprocessor = ActionDatasetPreprocessor(model_weights=args.detector_weights, conf_thresh=args.ssd_conf)
    ssd = preprocessor.detector
    ssd_gflops = keras_flops(ssd) / 1e9
    complexity = [{
        "model": "SSD300_H264 (ours)",
        "input": "300x300x3 DCT energy map",
        "params_m": ssd.count_params() / 1e6,
        "gflops": ssd_gflops,
        "weights_mb": os.path.getsize(args.detector_weights) / 1024 ** 2,
        "latency": {ssd_device: bench_ssd_model(ssd, args.micro_reps)},
    }]
    print(f"  SSD: {complexity[0]['params_m']:.2f} M params, {ssd_gflops:.2f} GFLOPs")

    yolos = {}
    for name in args.yolo_models:
        print(f"\nLoading {name} ...")
        y = YOLO(name)
        flop_model = copy.deepcopy(y.model).float().cpu().eval()
        gflops = torch_flops(flop_model, torch.zeros(1, 3, args.yolo_imgsz, args.yolo_imgsz)) / 1e9
        del flop_model
        weights_path = getattr(y, "ckpt_path", None) or name
        entry = {
            "model": os.path.splitext(os.path.basename(name))[0],
            "input": f"{args.yolo_imgsz}x{args.yolo_imgsz}x3 RGB",
            "params_m": sum(p.numel() for p in y.model.parameters()) / 1e6,
            "gflops": gflops,
            "weights_mb": os.path.getsize(weights_path) / 1024 ** 2 if os.path.exists(weights_path) else None,
            "latency": {},
        }
        # One instance per device: Ultralytics keeps a predictor bound to its first device
        instances = {}
        for dev in yolo_devices:
            instances[dev] = YOLO(name)
            entry["latency"][dev] = bench_yolo_model(instances[dev], dev, args.yolo_imgsz, args.micro_reps)
        complexity.append(entry)
        yolos[name] = (instances, gflops)
        print(f"  {entry['model']}: {entry['params_m']:.2f} M params, {gflops:.2f} GFLOPs")

    # ---------------- Per-video pipelines ----------------
    rows = []
    gt_paths = args.gt or []
    for v_i, video in enumerate(args.videos):
        video = os.path.abspath(video)
        vname = os.path.splitext(os.path.basename(video))[0]
        print(f"\n{'#' * 70}\n# Video: {vname}\n{'#' * 70}")
        fps, _, width, height = ife.get_fps_and_timebase(video)
        frames = get_video_frame_count(video) or sum(1 for _ in ffmpeg_frames(video, width, height))
        meta = {"width": width, "height": height, "fps": fps, "frames": frames,
                "duration_s": frames / fps if fps else 0.0,
                "bitstream_mb": os.path.getsize(video) / 1024 ** 2}

        gt_path = gt_paths[v_i] if v_i < len(gt_paths) else os.path.splitext(video)[0] + ".txt"
        if v_i >= len(gt_paths) and not os.path.exists(gt_path):
            gt_path = os.path.join(BASE_DIR, "data", "annotations", vname + ".txt")
        gt = load_gt(gt_path, width, height) if os.path.exists(gt_path) else None
        print(f"  Ground truth: {gt_path if gt else 'not found (accuracy skipped)'}")

        work = os.path.join(args.output_dir, "work", vname)

        def best_of(fn):
            runs = [fn() for _ in range(max(1, args.runs))]
            runs.sort(key=lambda r: r["total_s"])
            chosen = runs[len(runs) // 2]
            chosen["all_run_totals_s"] = [r["total_s"] for r in runs]
            return chosen

        results = []
        if args.impl in ("original", "both"):
            print("\n[A] OURS (original code): compressed-domain SSD + MV propagation")
            ours = best_of(lambda: run_ours(video, work, preprocessor, width, height, frames))
            p_files = ours.pop("_p_files")
            ours.update(pipeline="OURS (DCT-SSD + MV, original code)", device=ssd_device)
            results.append(derive_metrics(ours, meta, ssd_gflops, args.price_per_hour))
            mv_s = {k: v for k, v in ours["stages_s"].items() if k.startswith("4")}
        if args.impl in ("fast", "both"):
            print("\n[A'] OURS-VEC (vectorized front end): compressed-domain SSD + MV propagation")
            fast = best_of(lambda: run_ours_fast(video, ssd, width, height, frames, args.ssd_conf))
            # YOLO-KEY reuses the best available MV front end
            p_files = fast.pop("_p_mvs")
            fast.update(pipeline="OURS-VEC (DCT-SSD + MV, vectorized)", device=ssd_device)
            results.append(derive_metrics(fast, meta, ssd_gflops, args.price_per_hour))
            mv_s = {k: v for k, v in fast["stages_s"].items() if k.startswith(("2_", "4_"))}

        for name, (instances, gflops) in yolos.items():
            tag = os.path.splitext(os.path.basename(name))[0]
            for dev in yolo_devices:
                y = instances[dev]
                print(f"\n[B] {tag}-FULL on {dev}")
                r = best_of(lambda: run_yolo_full(video, y, dev, args.yolo_imgsz, args.yolo_conf, width, height))
                r.update(pipeline=f"{tag}-FULL (RGB every frame)", device=dev)
                results.append(derive_metrics(r, meta, gflops, args.price_per_hour))

                if not args.skip_keyframe_baseline:
                    print(f"\n[C] {tag}-KEY on {dev}")
                    r = best_of(lambda: run_yolo_keyframes(video, y, dev, args.yolo_imgsz, args.yolo_conf,
                                                           width, height, frames, p_files, mv_s))
                    r.update(pipeline=f"{tag}-KEY (RGB keyframes + MV)", device=dev)
                    results.append(derive_metrics(r, meta, gflops, args.price_per_hour))

        for r in results:
            r["video"] = vname
            r["video_meta"] = meta
            r["accuracy"] = evaluate_boxes(r["boxes"], gt) if gt else None
            r["frames_with_boxes"] = len(r.pop("boxes"))
            print(f"  {r['pipeline']:<38} {r['device']:<16} total {r['total_s']:8.2f} s  "
                  f"fps {r['fps']:7.1f}  detector calls {r['detector_calls']:4d}  "
                  f"GFLOPs {r['detector_gflops_total']:9.1f}"
                  + (f"  F1 {r['accuracy']['f1']:.3f}" if r["accuracy"] else ""))
        rows.extend(results)

        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)

    if not args.keep_work:
        shutil.rmtree(os.path.join(args.output_dir, "work"), ignore_errors=True)
    fco.cleanup_bins()  # the custom ffprobe leaves (empty) STRICT_FINAL_*.bin files behind

    # ---------------- Save ----------------
    with open(os.path.join(args.output_dir, "raw_results.json"), "w", encoding="utf-8") as f:
        json.dump({"environment": env, "complexity": complexity, "results": rows,
                   "args": vars(args)}, f, indent=2, default=str)

    cols = ["video", "pipeline", "device", "detector_calls", "total_s", "fps", "ms_per_frame",
            "realtime_factor", "realtime_streams", "detector_gflops_total",
            "detector_gflops_per_video_second", "data_bytes", "cpu_seconds", "extra_peak_rss_mb",
            "gpu_peak_mem_mb", "energy_j", "compute_seconds_per_video_hour", "usd_per_video_hour"]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["precision", "recall", "f1", "mean_iou"] + ["stages_s"])
        for r in rows:
            acc = r["accuracy"] or {}
            w.writerow([r.get(c) for c in cols] +
                       [acc.get("precision"), acc.get("recall"), acc.get("f1"), acc.get("mean_iou")] +
                       [json.dumps(r["stages_s"])])

    with open(os.path.join(args.output_dir, "model_complexity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "input", "params_m", "gflops", "weights_mb", "device", "median_ms", "mean_ms", "p95_ms"])
        for c in complexity:
            for dev, s in c["latency"].items():
                w.writerow([c["model"], c["input"], c["params_m"], c["gflops"], c["weights_mb"],
                            dev, s["median_ms"], s["mean_ms"], s["p95_ms"]])

    write_report(args.output_dir, env, complexity, rows, args)
    figs = make_figures(args.output_dir, rows, complexity)

    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE")
    print(f"  Report   : {os.path.join(args.output_dir, 'report.md')}")
    print(f"  Summary  : {os.path.join(args.output_dir, 'summary.csv')}")
    print(f"  Raw data : {os.path.join(args.output_dir, 'raw_results.json')}")
    for p in figs:
        print(f"  Figure   : {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
