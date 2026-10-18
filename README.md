# Compressed-Domain Person Detection, Tracking & Action Recognition (H.264)

Detect people, track them and recognise their action **directly from the H.264 bitstream**,
without running a detector on decoded RGB frames:

1. A custom FFmpeg build dumps **DCT coefficients of I-frames** and **motion vectors (MVs) of P-frames**.
2. A small CNN (**SSD300_H264**) detects people on the **I-frames only**, from DCT energy maps.
3. The boxes are **propagated through P-frames with the motion vectors** (B-frames hold the box).
4. The MVs around each person are fed to a **motion-vector CNN (TemporalCNN)** that classifies the action.
5. The result is rendered as a normal RGB video with boxes and action labels.

```
            ┌──────────────────── custom FFmpeg (one decode pass) ────────────────────┐
 video.mp4 ─┤ I-frames → DCT coefficients → 3-band energy map → SSD300_H264 → boxes   │
            │ P-frames → motion vectors ──────────────┐                               │
            └─────────────────────────────────────────┼───────────────────────────────┘
                                                      ▼
                         boxes on I-frames ──► MV box propagation (P) / hold (B)
                                                      ▼
                         MVs cropped around the person (+15 % margin)
                                                      ▼
                         TemporalCNN on 10-frame MV clips ──► action label
                                                      ▼
                         annotated RGB video + JSON + per-frame box CSV
```

---

## Table of contents
1. [Folder structure](#1-folder-structure)
2. [Installation](#2-installation)
3. [Quick start: one video in, annotated video out](#3-quick-start-one-video-in-annotated-video-out)
4. [Step-by-step workflow](#4-step-by-step-workflow)
   - 4.1 [Train the I-frame detector (SSD300_H264)](#41-train-the-i-frame-detector-ssd300_h264-optional)
   - 4.2 [Prepare the action-recognition dataset](#42-prepare-the-action-recognition-dataset)
   - 4.3 [Train the action-recognition model](#43-train-the-action-recognition-model)
   - 4.4 [Run the full pipeline](#44-run-the-full-pipeline)
   - 4.5 [Benchmark against YOLO](#45-benchmark-against-yolo-computational-comparison)
   - 4.6 [Other tools](#46-other-tools)
5. [How every component works](#5-how-every-component-works)
6. [File reference (every file explained)](#6-file-reference-every-file-explained)
7. [Benchmark results (Walking.mp4)](#7-benchmark-results-walkingmp4)
8. [Known issues & important notes](#8-known-issues--important-notes)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Folder structure

```
Capstone_Final/
├── README.md                      ← this file
├── requirements.txt               ← Python dependencies
│
├── run_pipeline.py                ★ MAIN ENTRY: video in → annotated video out
├── prepare_action_dataset.py      ★ videos → boxes CSV + cropped MVs (action dataset)
├── benchmark_detection.py         ★ computational comparison vs YOLO
├── fast_compressed_ops.py           vectorized compressed-domain front end (used by the 3 above)
├── feature_map.py                   DCT coefficients → 3-band energy map (original implementation)
├── visualize_pipeline.py            demo: SSD + MV tracking vs ground truth on one video
├── video_metadata.json              auto-written fps / time_base per video (used by sync_labels.py)
│
├── H264_Compressed_Detector/      ← compressed-domain extraction + SSD detector
│   ├── i_frame_extractor.py         custom FFmpeg → I-frame coefficients → feature-map .npy
│   ├── p_frame_extractor.py         custom FFmpeg → P-frame motion vectors → .npy
│   ├── bbox_propagator.py           box propagation with motion vectors
│   ├── dataset_builder_coeffs.py    SSD training data, step 1 (extract a whole dataset)
│   ├── sync_labels.py               SSD training data, step 2 (pair I-frames with labels)
│   ├── deep_check.py                sanity checks of extracted feature maps
│   ├── fix_dll.py                   one-off helper to install zlibwapi.dll (Windows cuDNN)
│   ├── zlibwapi.dll                 DLL required by cuDNN on Windows (GPU training)
│   └── model/
│       ├── model.py                 SSD300_H264 network definition
│       ├── data_loader.py           Keras data generator for FINAL_SSD_DATASET
│       ├── train.py                 SSD training → models/best_model.h5
│       └── evaluate_accuracy.py     SSD precision / recall / F1
│
├── Action_Recognition/            ← motion-vector action recognition (PyTorch)
│   ├── main.py                      CLI: `train` / `inference`
│   ├── config/config.yaml           all dataset / model / training settings
│   ├── src/
│   │   ├── model.py                 TemporalCNN (two-stream temporal net, 20-channel MV input)
│   │   ├── data_loader.py           MV clips: rasterise → crop to person → normalise → stack
│   │   ├── train.py                 training loop (GPU, mixed precision, checkpoints)
│   │   ├── inference.py             video-level prediction (averages clip predictions)
│   │   └── utils.py                 config, logging, seeding, metrics helpers
│   ├── data/raw/                    (created by prepare_action_dataset.py)
│   ├── models/                      (created by training: best_model.pth, …)
│   └── logs/                        (created by training)
│
├── ffmpeg_patch/                  ← the FFmpeg modification: patch + modified files + build guide
├── FFmpeg/FFmpeg/                 ← CUSTOM FFmpeg build (ffmpeg.exe / ffprobe.exe) [not in git:
│                                    build it from ffmpeg_patch/, see ffmpeg_patch/README.md]
│
├── models/                        [weights not in git]
│   ├── best_model.h5                trained SSD300_H264 weights (train with step 4.1)
│   └── yolo11n.pt                   YOLO11n weights (benchmark only; auto-downloaded)
│
├── data/
│   ├── videos/                      input videos, e.g. Walking.mp4 [not in git: add your own]
│   └── annotations/                 Walking.txt, Clapping.txt, Meet1.txt (per-frame boxes)
│
├── results/
│   ├── benchmark/                   report.md, summary.csv, figures (benchmark_detection.py)
│   ├── visualizations/              *_Visualization.mp4 (visualize_pipeline.py)
│   ├── pipeline_output/             (created by run_pipeline.py)
│   └── previous_outputs/            earlier results (Sept. GPU run): comparison images,
│                                    confusion matrix, SSD training log, model_comparison.mp4
│
├── docs/
│   └── Claims_and_Proof.docx        project claims document
│
├── archive/                       ← nothing here is used by the code
│   ├── zips/                        old snapshot zips (all_files, FFmpeg, output, …)
│   ├── ffmpeg_old_build_2026-04/    older FFmpeg build (I-frame-only extraction)
│   └── legacy/                      temp_I_frames/, temp_MVS/, early Action_Recognition MV test
│
└── venv/                          ← Python virtual environment
```

**Not in the git repository** (too large or private): `FFmpeg/` (build it from `ffmpeg_patch/`),
model weights in `models/`, videos, `archive/`, `venv/`, and everything generated
(`Action_Recognition/data|models|logs`, `results/pipeline_output`, rendered videos).

**Reference paper:** A. Duché, C. Chatelain, G. Gasso, *"See Without Decoding: Motion-Vector-Based
Tracking in Compressed Video"*, arXiv:2602.00153, 2026 — https://arxiv.org/abs/2602.00153

**Rule of thumb:** code stays where it is (the extractors locate `FFmpeg/` and `feature_map.py`
relative to their own position, so do not move `H264_Compressed_Detector/`, `FFmpeg/` or
`feature_map.py`). Inputs go in `data/`, weights in `models/`, everything generated in `results/`.

---

## 2. Installation

**Requirements:** Windows 10/11, Python 3.12, and the custom FFmpeg build in `FFmpeg/FFmpeg/`
(`ffmpeg.exe`, `ffprobe.exe` + DLLs). It is not stored in git: build it once from
`ffmpeg_patch/` by following [ffmpeg_patch/README.md](ffmpeg_patch/README.md).

```bat
cd "D:\Capstone Folder\Capstone_Final"
python -m venv venv
venv\Scripts\activate

:: NVIDIA GPU (for action-model training): install CUDA PyTorch FIRST
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Check the installation:

```bat
python -c "import tensorflow, torch, cv2, imageio, yaml; print('ok', torch.cuda.is_available())"
FFmpeg\FFmpeg\ffmpeg.exe -version
```

> **TensorFlow and GPUs on Windows:** TensorFlow ≥ 2.11 has no native-Windows GPU support,
> so the SSD detector runs on the CPU there (it is cheap: ~15 ms per I-frame). PyTorch (action
> model) uses the NVIDIA GPU normally. For TensorFlow on GPU use WSL2 / Linux.

All commands below are run from the project root with the venv activated.

---

## 3. Quick start: one video in, annotated video out

Requires the custom FFmpeg (section 2) and the two trained models: `models/best_model.h5`
(SSD detector, step 4.1; not distributed in the repository) and
`Action_Recognition/models/best_model.pth` (step 4.3). Put your videos in `data/videos/`.

```bat
python run_pipeline.py --video data\videos\Walking.mp4
```

Output in `results/pipeline_output/Walking/`:

| File | Content |
|---|---|
| `Walking_action.mp4` | RGB video: green person boxes, action + confidence on the main person, banner with the overall action |
| `Walking_result.json` | overall action, top-5 classes, per-clip predictions (frame ranges), runtime |
| `bboxes/Walking_bboxes.csv` | person box for every frame (pixel + normalised coordinates) |

Useful options: `--device cpu`, `--conf_thresh 0.5` (I-frame detection threshold),
`--clip_stride 5` (smaller = finer action timeline), `--keep_intermediate` (keep cropped MVs),
`--output_dir`, `--action_model`, `--detector_weights`.

---

## 4. Step-by-step workflow

### 4.1 Train the I-frame detector (SSD300_H264) *(optional)*

The trained weights (`models/best_model.h5`, 507 MB) are not in the repository: train them
here, or copy an existing `best_model.h5` into `models/`.
The training data is the *Human Activity Recognition – Video Dataset* with per-video box
annotations. Place these folders in the **project root**:

```
Human Activity Recognition - Video Dataset/                  <class>/<clip>.mp4
Human Activity Recognition - Video Dataset_annotations/      <class>/<clip>.txt
```

Annotation format (same as `data/annotations/*.txt`):

```
# resolution: 1080,1920
frame,track_id,class,x1_norm,y1_norm,x2_norm,y2_norm
0,1,0,0.3566,0.0966,0.8179,0.9187
```

Then:

```bat
:: 1. extract I-frame feature maps (+ MVs) for every video
::    -> "...Video Dataset__DCT/", "...Video Dataset__MVS/", video_metadata.json
python H264_Compressed_Detector\dataset_builder_coeffs.py

:: 2. pair every I-frame feature map with its boxes -> FINAL_SSD_DATASET/maps + labels
python H264_Compressed_Detector\sync_labels.py

:: (optional) sanity-check the extracted feature maps
python H264_Compressed_Detector\deep_check.py

:: 3. train -> models/best_model.h5 (best val loss), models/final_model.h5
python H264_Compressed_Detector\model\train.py

:: 4. evaluate on the held-out 5 % -> precision / recall / F1 @ IoU 0.5
python H264_Compressed_Detector\model\evaluate_accuracy.py
```

### 4.2 Prepare the action-recognition dataset

Put the training videos in **one sub-folder per action** (the folder name becomes the label):

```
my_videos/
├── Walking/      a.mp4  b.mp4 …
├── Clapping/     a.mp4 …
└── Sitting/      …
```

```bat
python prepare_action_dataset.py --video_dir my_videos
```

For every video this runs extraction → I-frame detection → MV propagation → MV cropping, and
writes into `Action_Recognition/data/raw/`:

| Output | Content |
|---|---|
| `bboxes/<Class>_<video>_bboxes.csv` | person box per frame |
| `motion_vectors/<Class>_<video>/<frame>.npy` | MVs of the macroblocks inside the box + 15 % margin |
| `video_manifest.csv` | video → label, resolution, fps, frame count, source path |

Options: `--padding_ratio 0.15` (margin around the box; must be ≥ `bbox_padding` in
`config.yaml`), `--conf_thresh 0.5`, `--output_dir`, `--keep_temp` (save full-frame maps/MVs).
Speed: about 12 s per 10-second 1080p video on CPU.

### 4.3 Train the action-recognition model

```bat
cd Action_Recognition
..\venv\Scripts\python main.py train --config config/config.yaml
```

* Labels come from `video_manifest.csv` (class folder) → or `dataset.labels_csv` → or the video name.
* Split: whole videos are held out per class (needs ≥ 2 videos per class); otherwise each
  video is split in time (first 80 % train / last 20 % val, optimistic numbers).
* Requires an NVIDIA GPU by default (`device: "cuda"`): it stops with a clear message if CUDA
  is unavailable. Use `--device cpu` to force CPU.
* Overrides: `--epochs`, `--batch-size`, `--lr`, `--device`, `--resume models/last_model.pth`.

Outputs in `Action_Recognition/`:

| File | Content |
|---|---|
| `models/best_model.pth` | best checkpoint (weights + class names + config: self-contained) |
| `models/last_model.pth` | last epoch (for `--resume`) |
| `models/training_history.json` | loss / accuracy per epoch |
| `models/dataset_split.json` | classes, train/val videos, clips per class |
| `logs/training_log.txt` | full log incl. confusion matrix |

Standalone inference on a prepared video:

```bat
..\venv\Scripts\python main.py inference --model models/best_model.pth ^
    --motion-dir data/raw/motion_vectors/Walking_a --bboxes-csv data/raw/bboxes/Walking_a_bboxes.csv --show-clips
```

### 4.4 Run the full pipeline

See [section 3](#3-quick-start-one-video-in-annotated-video-out). Back in the project root:

```bat
python run_pipeline.py --video data\videos\Clapping.mp4
```

### 4.5 Benchmark against YOLO (computational comparison)

```bat
python benchmark_detection.py --videos data\videos\Walking.mp4 --runs 3
python benchmark_detection.py --videos data\videos\Walking.mp4 data\videos\Clapping.mp4 data\videos\Meet1.mp4
python benchmark_detection.py --yolo_models models\yolo11n.pt yolov8n.pt yolo11s.pt --yolo_devices cpu cuda
```

Pipelines compared on the same video and machine:

| Row | Pipeline |
|---|---|
| **OURS-VEC** | single FFmpeg pass → DCT-SSD on I-frames → MV propagation (vectorized code) |
| **OURS (original code)** | same method with the original per-macroblock Python front end |
| **YOLO-FULL** | decode every frame to RGB → YOLO on every frame (conventional pipeline) |
| **YOLO-KEY** | decode key frames only → YOLO on them → the same MV propagation (ablation) |

Measured: parameters, GFLOPs per inference, model-only latency, time per stage, total time,
FPS, real-time factor, concurrent streams, detector calls, detector GFLOPs per video, data volume
fed to the detector, CPU-seconds, RAM, GPU memory/energy (if available), precision/recall/F1/mIoU
against `data/annotations/<video>.txt`. Options: `--impl original|fast|both`, `--runs N`,
`--price_per_hour X` (cost per video-hour), `--skip_keyframe_baseline`, `--output_dir`.

Output in `results/benchmark/`: `report.md` (paper-ready tables, ratios, notes),
`summary.csv`, `model_complexity.csv`, `raw_results.json`, `fig_*.png`.

### 4.6 Other tools

| Command | Purpose |
|---|---|
| `python visualize_pipeline.py` | SSD + MV tracking (green) vs annotation boxes (red) on `data/videos/Walking.mp4` → `results/visualizations/Walking_Visualization.mp4` (uses the original extractors) |
| `python fast_compressed_ops.py data\videos\Walking.mp4` | verifies that the vectorized front end gives exactly the same feature maps / MVs / boxes as the original code |
| `python feature_map.py` | plots the low / mid / high-frequency maps of one frame from a `coeffs.bin` |

---

## 5. How every component works

### 5.1 Custom FFmpeg (`FFmpeg/FFmpeg/libavcodec/h264_coeff_extract.c`)
The H.264 decoder (`h264_slice.c`, `h264dec.c`) was modified to dump, right after entropy
decoding of each macroblock (full details and build guide: `ffmpeg_patch/README.md`):
* **`<prefix>_I.bin`**: for every macroblock of every I-frame, a 24-byte header
  (`pts, mb_x, mb_y, qp, slice_type`) + 16×48×2 quantised transform coefficients (int16) +
  luma DC. **Pixel reconstruction of I-frame macroblocks is skipped.**
* **`<prefix>_P.bin`**: for every macroblock of every P-frame an 80-byte record
  (`frame_idx, mb_x, mb_y, 16 × (dx, dy)` quarter-pixel list-0 motion vectors).

`<prefix>` is the environment variable `H264_COEFF_EXTRACT_FILE` (default `STRICT_FINAL` in the
current directory). Both files are opened whenever an H.264 decoder starts, including inside `ffprobe`.

### 5.2 I-frame feature map (`feature_map.py`, `fast_compressed_ops.build_iframe_feature_maps`)
For each 4×4 luma block the 16 coefficients are split into frequency bands
(low = {0,1,4}, mid = {2,3,5,6,8}, high = the rest); band energy `log(1 + Σ c²)` gives a
**(H/4) × (W/4) × 3** map (e.g. 480 × 272 × 3 for 1080 × 1920 video).

### 5.3 SSD300_H264 detector (`H264_Compressed_Detector/model/model.py`)
Input: the feature map, standardised and resized to 300 × 300 × 3. Five 3×3 conv layers
(32 → 512 channels, each followed by 2×2 max-pooling), Dense 1024, Dense 245 → a **7 × 7 grid ×
[objectness, cx, cy, w, h]** (YOLO-style). Boxes with objectness ≥ 0.5 → NMS (IoU 0.4) → scaled to
video pixels. 44.3 M parameters, **3.47 GFLOPs** per inference.

### 5.4 Motion-vector propagation (`bbox_propagator.py`, `fast_compressed_ops.propagate_bbox_fast`)
On each P-frame: take the macroblocks overlapping the previous box, keep the ones that move,
take the **median** displacement and shift the box by its negative (H.264 MVs point backwards
to the reference frame). I-frames reset the boxes with fresh detections; B-frames hold the box.

### 5.5 Action-recognition input (`Action_Recognition/src/data_loader.py`)
Per P-frame: MVs rasterised to a (H/16) × (W/16) × 2 grid in pixels → cropped to the main
person's box (+15 %) → resized to 224 × 224 → clipped to ±20 px and scaled to [-1, 1].
10 consecutive P-frames are stacked to a **20 × 224 × 224** tensor (dx₀, dy₀, dx₁, dy₁ …).
Training uses overlapping clips (stride 4) and random horizontal flips (dx negated).

### 5.6 TemporalCNN (`Action_Recognition/src/model.py`)
The temporal stream of the two-stream network (`temporal_cls.prototxt`): conv1 7×7/2 (96) →
LRN → pool → conv2 5×5/2 (256) → LRN → pool → conv3-5 3×3 (512) → pool → fc6 4096 → fc7 2048
→ fc8 (classes), PReLU activations, Caffe-style ceil pooling (6 × 6 × 512 into fc6).
Video prediction = average of the clip probabilities.

---

## 6. File reference (every file explained)

### Project root

| File | What it does | Run it? |
|---|---|---|
| `run_pipeline.py` | Full chain for one video: action model loading → `ActionDatasetPreprocessor.process_single_video` (extraction, detection, propagation, MV crop) → `ActionRecognizer` → rendering with imageio. Sets UTF-8 console output and TF GPU memory growth (TF + PyTorch share the process). | **Yes**, main entry |
| `prepare_action_dataset.py` | `ActionDatasetPreprocessor`: loads the SSD; `process_single_video` = vectorized front end (default); `process_single_video_original` = original two-pass implementation kept for reference. CLI walks a folder of class sub-folders and writes boxes CSVs, cropped MVs and `video_manifest.csv`. | **Yes** (step 4.2) |
| `benchmark_detection.py` | Ours vs YOLO computational benchmark (section 4.5). | **Yes** (step 4.5) |
| `fast_compressed_ops.py` | Vectorized NumPy version of the front end: single-threaded single-pass extraction, one ffprobe call, `np.fromfile` parsing, row-wise medians, vectorized propagation; plus `verify_against_original()`. Produces identical outputs to the original modules, ~24× faster. | Library (run it to verify) |
| `feature_map.py` | Original I-frame feature-map builder (`build_and_save_npy`, used by `i_frame_extractor.py`) + plotting function. Must stay in the root. | Library |
| `visualize_pipeline.py` | Demo video of detection + tracking against annotation boxes. | Optional |
| `video_metadata.json` | fps / time_base / resolution per processed video, written by `i_frame_extractor.py`, read by `sync_labels.py`. | Auto-generated |
| `requirements.txt` | Python dependencies. | — |

### `H264_Compressed_Detector/`

| File | What it does |
|---|---|
| `i_frame_extractor.py` | `process_video()`: ffprobe (fps, resolution, I-frame positions) → custom FFmpeg with `H264_COEFF_EXTRACT_FILE` → `feature_map.build_and_save_npy` for each I-frame → `<frame>.npy`; deletes the bin. |
| `p_frame_extractor.py` | `process_video()`: custom FFmpeg → `STRICT_FINAL_P.bin` → per-P-frame structured arrays (`src_x, src_y, w, h, motion_x, motion_y, motion_scale…`) → `<frame>.npy`. |
| `bbox_propagator.py` | `block_overlaps_bbox`, `propagate_bbox` (median of moving overlapping MVs), YOLO ↔ pixel box helpers. |
| `dataset_builder_coeffs.py` | Runs both extractors over the whole HAR dataset (SSD training data). |
| `sync_labels.py` | Matches I-frame feature maps with annotation boxes (via PTS) → `FINAL_SSD_DATASET/maps/*.npy` + `labels/*.txt`. |
| `deep_check.py` | Prints statistics of extracted maps (channel ranges, I-frames differ, …). |
| `fix_dll.py` | One-off helper that extracts `zlibwapi.dll` from a zip in `C:\Windows\System32`. |
| `zlibwapi.dll` | Needed by cuDNN on Windows for GPU training. |
| `model/model.py` | `SSD300_H264` Keras model. |
| `model/data_loader.py` | `H264DatasetGenerator`: loads maps + labels, builds 7×7×5 targets, 95 / 5 split. |
| `model/train.py` | Custom YOLO-style loss, Adam, checkpoint → `models/best_model.h5`, early stopping. |
| `model/evaluate_accuracy.py` | TP / FP / FN at IoU 0.5 on the held-out split. |

### `Action_Recognition/`

| File | What it does |
|---|---|
| `main.py` | `python main.py train …` / `python main.py inference …` |
| `config/config.yaml` | Every setting: clip length, crop margin, normalisation, split strategy, dropout, optimiser, LR schedule, early stopping, mixed precision, device, workers. |
| `src/model.py` | `TemporalCNN`, `load_model` (reads class count from the checkpoint). |
| `src/data_loader.py` | `VideoMotionData` (per-video rasterised crops in RAM), `MotionClipDataset`, label resolution, video-level / temporal split. |
| `src/train.py` | `Trainer`: class-weighted cross-entropy with label smoothing, Adam/SGD, ReduceLROnPlateau, early stopping, AMP (bf16/fp16), clip- and video-level accuracy, confusion matrix, resume. |
| `src/inference.py` | `ActionRecognizer.predict()`: same clip construction as training, averages clip probabilities, returns top-5 and per-clip labels. |
| `src/utils.py` | YAML config, logger, seeds, device selection (fails loudly without CUDA), meters, early stopping. |

### Data, models, results

| Path | Content |
|---|---|
| `data/videos/*.mp4` | Input videos (H.264 MP4). Not in the repository: the benchmark/README examples use `Walking.mp4`, `Clapping.mp4`, `Meet1.mp4`; add your own. |
| `data/annotations/*.txt` | Per-frame person boxes for `Walking`, `Clapping`, `Meet1`. **Generated by YOLO** (see section 8). |
| `models/best_model.h5` | Trained SSD300_H264 weights (507 MB: the H5 also stores optimiser state). Not in the repository. |
| `models/yolo11n.pt` | YOLO11n weights for the benchmark. |
| `results/benchmark/` | Latest benchmark (Walking): `report.md`, CSVs, figures, log. |
| `results/visualizations/` | Output of `visualize_pipeline.py` (videos are not in the repository). |
| `results/previous_outputs/` | Earlier (September) results: per-class comparison images, confusion matrix, evaluation log, SSD GPU training log, `model_comparison.mp4`. |

---

## 7. Benchmark results (Walking.mp4)

1080 × 1920, 292 frames, 9.7 s, CPU only, median of 3 runs. Full report: `results/benchmark/report.md`.

| Pipeline | Detector calls | Total time | FPS | × real time | Detector GFLOPs | CPU-s | F1* |
|---|---|---|---|---|---|---|---|
| **OURS-VEC** | 3 | **5.12 s** | **57.0** | **1.90×** | **10.4** | 6.1 | 0.836 |
| OURS (original code) | 3 | 124.46 s | 2.3 | 0.08× | 10.4 | 191.3 | 0.836 |
| YOLO11n on every frame | 292 | 26.36 s | 11.1 | 0.37× | 1 892 | 213.4 | 1.000* |
| YOLO11n keyframes + MV | 3 | 5.36 s | 54.5 | 1.82× | 19.4 | 3.8 | 0.880 |

Ours vs YOLO-FULL: **5.1× faster, 182× less detection compute, 97× fewer detector calls,
32× less data to the detector, 35× fewer CPU-seconds**; per inference 3.47 vs 6.48 GFLOPs and
14.7 ms vs 44.1 ms. Most of the end-to-end gain comes from detecting only on I-frames and tracking
with MVs (YOLO-KEY is almost as fast); the DCT input adds 1.9× fewer FLOPs and 3× faster
inference per detection and needs no RGB decode of key frames.

\* The annotations were produced by YOLO, so F1 measures agreement with YOLO, not true accuracy.

---

## 8. Known issues & important notes

1. **Multithreaded MV corruption (original extractor).** With FFmpeg's default frame-threaded
   decoding, several threads write `STRICT_FINAL_P.bin` at the same time: the file differs on
   every run and parses into ~1.5 M bogus "frames". `fast_compressed_ops.py` runs the custom
   FFmpeg with `-threads 1` (deterministic, and 4× faster). `prepare_action_dataset.py` and
   `run_pipeline.py` use it. `p_frame_extractor.py` itself is unchanged; MVs extracted earlier
   with it may be affected. (I-frame coefficients are not affected.)
2. **The custom ffprobe also writes the `.bin` files** (it decodes frames too). Always call
   ffprobe *before* extracting, never between extraction and parsing. Empty
   `STRICT_FINAL_*.bin` files in the project root are harmless leftovers and can be deleted.
3. **Decoding is only partly avoided.** The modified decoder skips pixel reconstruction for
   I-frames, but P- and B-frames are still fully reconstructed (inverse transform + motion
   compensation) although only their motion vectors are used, and every frame is entropy-decoded.
   The whole extraction pass takes 1.6 s for the 292-frame 1080p video. Skipping reconstruction of
   P/B-frames too would cut this further. State it this way in the paper.
4. **Annotations are YOLO output.** `data/annotations/*.txt` (and the SSD training labels:
   `sync_labels.py` converts "YOLO 0-based" classes) come from YOLO. Use human-labelled boxes
   for accuracy claims.
5. **Model size.** SSD300_H264 has 44.3 M parameters (42.5 M in the first Dense layer): its
   advantage is FLOPs and how rarely it runs, not parameter count.
6. **TensorFlow on Windows runs on the CPU** (see section 2). Compare like with like (CPU vs CPU).

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` | Run `chcp 65001` or `set PYTHONIOENCODING=utf-8` (the main scripts already handle it). |
| `CUDA GPU requested but not available` | Install CUDA PyTorch (section 2), check `nvidia-smi`, or pass `--device cpu`. |
| `Action model not found` | Train it first (step 4.3) or pass `--action_model`. |
| `Custom FFmpeg extraction failed` | Check `FFmpeg\FFmpeg\ffmpeg.exe -version` and that the DLLs next to it are present. |
| `Found only one action class` | Training needs ≥ 2 classes: use class sub-folders in step 4.2. |
| No boxes on a video | Lower `--conf_thresh`; the detector only runs on I-frames, so a person absent from the first I-frame is picked up at the next one. |
| Out of GPU memory during training | `--batch-size 16` (or set `training.mixed_precision: true`). |
