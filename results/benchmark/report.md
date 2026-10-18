# Compressed-domain detection vs RGB YOLO - computational benchmark

## Environment

- **OS**: Windows 11
- **CPU**: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- **CPU cores (logical)**: 18
- **RAM (GB)**: 15.4
- **GPU**: none
- **TensorFlow**: 2.21.0 (SSD device: cpu)
- **PyTorch**: 2.14.1+cpu
- **Python**: 3.12.10

## 1. Model complexity

| Model | Input | Params (M) | GFLOPs / inference | Weights file (MB) | Median latency (ms) |
|---|---|---|---|---|---|
| SSD300_H264 (ours) | 300x300x3 DCT energy map | 44.29 | 3.47 | 506.91 | cpu: 14.71 |
| yolo11n | 640x640x3 RGB | 2.62 | 6.48 | 5.35 | cpu: 44.05 |

GFLOPs = 2 x multiply-accumulates of all convolution and dense layers, counted the same way for both models.

## 2. End-to-end results

### Walking  (1080x1920, 292 frames, 30.00 fps, 9.7 s, bitstream 9.29 MB)

| Pipeline | Device | Detector calls | Total time (s) | FPS | ms/frame | Real-time x | RT streams | Detector GFLOPs (video) | GFLOPs / video-s | Data to vision stack (MB) | CPU-s | Extra RAM (MB) | GPU mem (MB) | Energy (J) | F1@0.5 | mIoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OURS (DCT-SSD + MV, original code) | cpu | 3 | 124.46 | 2.35 | 426.23 | 0.08 | 0.08 | 10.42 | 1.07 | 54.05 | 191.31 | 1,053 | - | - | 0.836 | 0.670 |
| OURS-VEC (DCT-SSD + MV, vectorized) | cpu | 3 | 5.12 | 56.98 | 17.55 | 1.90 | 1.90 | 10.42 | 1.07 | 53.96 | 6.05 | 672.44 | - | - | 0.836 | 0.670 |
| yolo11n-FULL (RGB every frame) | cpu | 292 | 26.36 | 11.08 | 90.26 | 0.37 | 0.37 | 1,892 | 194.43 | 1,732 | 213.41 | 373.51 | - | - | 1.000 | 0.936 |
| yolo11n-KEY (RGB keyframes + MV) | cpu | 3 | 5.36 | 54.48 | 18.35 | 1.82 | 1.82 | 19.44 | 2.00 | 67.28 | 3.77 | 119.73 | - | - | 0.880 | 0.754 |

**Stage breakdown (s)**

- OURS (DCT-SSD + MV, original code) (cpu): 1_ffprobe_frame_types = 1.608, 2_iframe_coeff_extraction_ffmpeg = 6.488, 3_dct_feature_map_build = 5.166, 4a_mv_extraction_ffmpeg = 1.541, 4b_ffprobe_p_frame_index = 1.656, 4c_mv_bin_parse_python = 2.503, 4d_mv_npy_write_python = 95.652, 5_ssd_detection_iframes = 0.347, 6_mv_box_propagation = 9.497
- OURS-VEC (DCT-SSD + MV, vectorized) (cpu): 1_ffprobe_frame_types = 1.677, 2_single_pass_extraction_ffmpeg = 1.601, 3_dct_feature_map_build_numpy = 0.116, 4_mv_parse_numpy = 1.476, 5_ssd_detection_iframes = 0.191, 6_mv_box_propagation = 0.063
- yolo11n-FULL (RGB every frame) (cpu): 1_rgb_decode_all_frames = 7.464, 2_yolo_detection_all_frames = 18.892
- yolo11n-KEY (RGB keyframes + MV) (cpu): 1_ffprobe_keyframe_index = 1.765, 2_rgb_decode_keyframes = 0.233, 3_yolo_detection_keyframes = 0.221, 2_single_pass_extraction_ffmpeg (from A) = 1.601, 4_mv_parse_numpy (from A) = 1.476, 5_mv_box_propagation = 0.063

**Time by cost category (s)**

| Pipeline | Device | Codec / bitstream (FFmpeg, ffprobe) | Python data handling (prototype overhead) | Detector (model + NMS + pre/post) | MV box tracking |
|---|---|---|---|---|---|
| OURS (DCT-SSD + MV, original code) | cpu | 11.293 | 103.321 | 0.347 | 9.497 |
| OURS-VEC (DCT-SSD + MV, vectorized) | cpu | 3.278 | 1.593 | 0.191 | 0.063 |
| yolo11n-FULL (RGB every frame) | cpu | 7.464 | 0.000 | 18.892 | 0.000 |
| yolo11n-KEY (RGB keyframes + MV) | cpu | 3.599 | 1.476 | 0.221 | 0.063 |

> **Ground-truth caveat:** a YOLO baseline matches the ground truth almost perfectly (F1 >= 0.99), which indicates the annotations were produced by YOLO itself. For this video the F1 / mIoU columns measure *agreement with YOLO*, not true accuracy; use human-annotated boxes for accuracy claims.

**OURS-VEC (DCT-SSD + MV, vectorized) relative to each other row** (>1 = ours better)

| Baseline | Device | Speed-up (time) | Detector-FLOPs reduction | Detector-call reduction | Data-volume reduction | CPU-s reduction | F1 difference (ours - baseline) |
|---|---|---|---|---|---|---|---|
| OURS (DCT-SSD + MV, original code) | cpu | 24.29x | 1.00x | 1.00x | 1.00x | 31.64x | 0.000 |
| yolo11n-FULL (RGB every frame) | cpu | 5.14x | 181.70x | 97.33x | 32.10x | 35.29x | -0.164 |
| yolo11n-KEY (RGB keyframes + MV) | cpu | 1.05x | 1.87x | 1.00x | 1.25x | 0.62x | -0.045 |

## 3. Notes for interpreting the numbers

- *Total time* of OURS is the sum of its sequential stages. For YOLO-FULL it is RGB decode (measured on its own pass) + YOLO time; the pipelined wall time with decode overlapping inference is in raw_results.json (`overlapped_decode_plus_detect_s`).
- The custom FFmpeg dumps coefficients/MVs right after entropy decoding and skips pixel reconstruction of I-frames, but P/B-frames are still fully reconstructed (the original code also decodes the video twice: one pass for I-frame coefficients, one for MVs; OURS-VEC decodes once). Skipping P/B reconstruction too would make extraction cheaper; report the measured times and present that saving as potential, not achieved.
- OURS (original code) parses the binaries with per-macroblock Python loops; OURS-VEC (fast_compressed_ops.py) does the same computation with NumPy and produces identical feature maps, motion vectors and boxes (verified bit-exact). The difference between the two rows is implementation overhead, not a property of the method.
- The custom FFmpeg must run with `-threads 1`: with frame-threaded decoding its MV dump interleaves records from several threads (corrupted, non-deterministic STRICT_FINAL_P.bin). Both OURS rows use -threads 1 for the MV pass.
- YOLO-KEY uses the same detection schedule and MV tracking as OURS, so OURS vs YOLO-KEY isolates the effect of the DCT input; OURS vs YOLO-FULL shows the gain of the full approach.
- Compare pipelines on the same device. TensorFlow has no native-Windows GPU support (TF >= 2.11), so on Windows the SSD runs on CPU; compare it with the YOLO CPU rows, or run under WSL2/Linux for GPU-vs-GPU numbers.
- Detector FLOPs exclude NMS, pre- and post-processing; those are inside the measured times.
