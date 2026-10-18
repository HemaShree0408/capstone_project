# src/data_loader.py
"""
Data loading for compressed-domain action recognition.

Consumes the output of prepare_action_dataset.py:
  <data_root>/bboxes/<video>_bboxes.csv         per-frame person boxes (pixel + normalized)
  <data_root>/motion_vectors/<video>/<f>.npy    cropped P-frame macroblock MVs
                                                (structured array: src_x, src_y, w, h,
                                                 motion_x, motion_y, motion_scale, ...)
  <data_root>/video_manifest.csv                (optional) video -> label, width, height

A training sample is a clip of `num_frames` consecutive P-frames of one video:
  1. each frame's macroblock MVs are rasterised to a (H/16, W/16, 2) grid in pixels
  2. the grid is cropped to the padded person box of that frame
  3. the crop is resized to (frame_size, frame_size) and normalized to [-1, 1]
  4. frames are stacked as channels [dx_0, dy_0, dx_1, dy_1, ...]
giving a (2 * num_frames, frame_size, frame_size) tensor for TemporalCNN.
"""

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import normalize_motion_values

MB_SIZE = 16  # H.264 macroblock size in pixels


# ============================================================
# CSV / MANIFEST HELPERS
# ============================================================

def load_bbox_csv(csv_path):
    """Returns {frame_idx: [(x1, y1, x2, y2), ...]} in pixels"""
    boxes = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            boxes[int(row['frame'])].append((
                float(row['x1']), float(row['y1']),
                float(row['x2']), float(row['y2'])
            ))
    return dict(boxes)


def infer_resolution_from_csv(csv_path):
    """
    Recovers the video resolution from the pixel / normalized column pairs
    written by prepare_action_dataset.py (x2 / x2_norm == width).
    """
    widths, heights = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if float(row['x2_norm']) > 0.05:
                widths.append(float(row['x2']) / float(row['x2_norm']))
            if float(row['y2_norm']) > 0.05:
                heights.append(float(row['y2']) / float(row['y2_norm']))
    if not widths or not heights:
        return None, None
    return int(round(np.median(widths))), int(round(np.median(heights)))


def load_manifest(manifest_path):
    """Returns {video_name: row_dict} from video_manifest.csv (or {} if missing)"""
    if not manifest_path or not Path(manifest_path).exists():
        return {}
    with open(manifest_path, newline='', encoding='utf-8') as f:
        return {row['video_name']: row for row in csv.DictReader(f)}


def load_labels_csv(labels_path):
    """Optional user-provided labels: CSV with columns video_name,label"""
    with open(labels_path, newline='', encoding='utf-8') as f:
        return {row['video_name']: row['label'].strip() for row in csv.DictReader(f)}


def infer_label_from_name(video_name):
    """
    Fallback label from the video name:
      v_Walking_g01_c02 -> walking   (UCF-101 style)
      Clapping_03       -> clapping
      walking12         -> walking
    """
    ucf = re.match(r'^v_([A-Za-z]+)_g\d+_c\d+$', video_name)
    if ucf:
        return ucf.group(1).lower()
    stripped = re.sub(r'[\s_\-\.]*\d+$', '', video_name)
    return (stripped or video_name).lower()


def resolve_labels(video_names, manifest, labels_csv=None):
    """
    Label priority: explicit labels CSV > manifest (class sub-folder of the
    source video) > inferred from video name.
    """
    explicit = load_labels_csv(labels_csv) if labels_csv else {}
    labels = {}
    for name in video_names:
        if name in explicit:
            labels[name] = explicit[name].lower()
        elif manifest.get(name, {}).get('label'):
            labels[name] = manifest[name]['label'].lower()
        else:
            labels[name] = infer_label_from_name(name)
    return labels


# ============================================================
# MOTION VECTOR RASTERISATION
# ============================================================

def rasterise_mvs(mv_array, grid_h, grid_w):
    """Sparse macroblock MVs -> dense (grid_h, grid_w, 2) displacement grid in pixels"""
    grid = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    if mv_array is None or len(mv_array) == 0:
        return grid

    scale = mv_array['motion_scale'].astype(np.float32)
    scale[scale == 0] = 4.0
    cols = mv_array['src_x'].astype(np.int64) // MB_SIZE
    rows = mv_array['src_y'].astype(np.int64) // MB_SIZE
    valid = (cols < grid_w) & (rows < grid_h)

    grid[rows[valid], cols[valid], 0] = mv_array['motion_x'][valid] / scale[valid]
    grid[rows[valid], cols[valid], 1] = mv_array['motion_y'][valid] / scale[valid]
    return grid


def select_region(boxes, pad_ratio, width, height, box_mode='largest'):
    """
    Picks the crop region for one frame:
      'largest' -> the largest detected person (main actor)
      'union'   -> the box enclosing every detected person
    then pads it by pad_ratio on every side (same margin as prepare_action_dataset.py).
    """
    if box_mode == 'union':
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
    else:
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    pad_w = max(1.0, x2 - x1) * pad_ratio
    pad_h = max(1.0, y2 - y1) * pad_ratio
    return (max(0.0, x1 - pad_w), max(0.0, y1 - pad_h),
            min(float(width), x2 + pad_w), min(float(height), y2 + pad_h))


def crop_grid(grid, region):
    """Crops a macroblock grid to a pixel region (always at least 1x1 cell)"""
    grid_h, grid_w = grid.shape[:2]
    x1, y1, x2, y2 = region
    c1 = int(np.clip(np.floor(x1 / MB_SIZE), 0, grid_w - 1))
    r1 = int(np.clip(np.floor(y1 / MB_SIZE), 0, grid_h - 1))
    c2 = int(np.clip(np.ceil(x2 / MB_SIZE), c1 + 1, grid_w))
    r2 = int(np.clip(np.ceil(y2 / MB_SIZE), r1 + 1, grid_h))
    return grid[r1:r2, c1:c2]


def build_clip_tensor(crops, frame_size, motion_norm_range, flip=False):
    """
    Stacks per-frame MV crops into a (2*T, frame_size, frame_size) float32 array.
    flip=True mirrors the clip horizontally (and negates dx so motion stays consistent).
    """
    stacked = np.empty((2 * len(crops), frame_size, frame_size), dtype=np.float32)
    for t, crop in enumerate(crops):
        flow = cv2.resize(crop.astype(np.float32), (frame_size, frame_size),
                          interpolation=cv2.INTER_LINEAR)
        if flow.ndim == 2:  # cv2 drops the channel axis for 1-pixel-wide edge cases
            flow = flow[:, :, None].repeat(2, axis=2)
        flow = normalize_motion_values(flow, -motion_norm_range, motion_norm_range)
        if flip:
            flow = flow[:, ::-1]
            flow[:, :, 0] = -flow[:, :, 0]
        stacked[2 * t] = flow[:, :, 0]
        stacked[2 * t + 1] = flow[:, :, 1]
    return stacked


# ============================================================
# PER-VIDEO MOTION DATA
# ============================================================

class VideoMotionData:
    """
    All motion data of one video, rasterised and cropped once, kept in memory
    as small macroblock-resolution crops (float16).
    """

    def __init__(self, name, mv_dir, bbox_csv, width=None, height=None,
                 pad_ratio=0.15, box_mode='largest', max_frame_gap=4):
        self.name = name
        mv_dir = Path(mv_dir)

        if not width or not height:
            width, height = infer_resolution_from_csv(bbox_csv)
        if not width or not height:
            raise ValueError(f"Cannot determine resolution of video '{name}'")
        self.width, self.height = int(width), int(height)

        grid_h = int(np.ceil(self.height / MB_SIZE))
        grid_w = int(np.ceil(self.width / MB_SIZE))
        boxes_by_frame = load_bbox_csv(bbox_csv)

        frame_ids = sorted(int(p.stem) for p in mv_dir.glob('*.npy') if p.stem.isdigit())

        self.frame_ids = []
        self.crops = []
        for f_idx in frame_ids:
            boxes = boxes_by_frame.get(f_idx)
            if not boxes:
                continue
            mv_array = np.load(mv_dir / f"{f_idx}.npy", allow_pickle=True)
            grid = rasterise_mvs(mv_array, grid_h, grid_w)
            region = select_region(boxes, pad_ratio, self.width, self.height, box_mode)
            self.crops.append(crop_grid(grid, region).astype(np.float16))
            self.frame_ids.append(f_idx)

        # Split into continuous segments so a clip never jumps over a long gap
        # (e.g. frames where the person was lost)
        self.segments = []
        start = 0
        for i in range(1, len(self.frame_ids) + 1):
            if i == len(self.frame_ids) or self.frame_ids[i] - self.frame_ids[i - 1] > max_frame_gap:
                if i > start:
                    self.segments.append((start, i))
                start = i

    def __len__(self):
        return len(self.frame_ids)

    def clip_starts(self, num_frames, stride, lo=0, hi=None):
        """Start positions of all clips fully inside [lo, hi) and inside one segment"""
        hi = len(self.frame_ids) if hi is None else hi
        starts = []
        for seg_start, seg_end in self.segments:
            s, e = max(seg_start, lo), min(seg_end, hi)
            starts.extend(range(s, e - num_frames + 1, max(1, stride)))
        return starts

    def clip(self, start, num_frames, frame_size, motion_norm_range, flip=False):
        return build_clip_tensor(self.crops[start:start + num_frames],
                                 frame_size, motion_norm_range, flip=flip)


# ============================================================
# DATASET
# ============================================================

class MotionClipDataset(Dataset):
    """
    Clip-level dataset.
    Returns (motion_tensor (2T, S, S), class_index, video_index)
    """

    def __init__(self, videos, clips, video_labels, config, train=False):
        """
        Args:
            videos: list of VideoMotionData
            clips: list of (video_index, start_position)
            video_labels: list of class indices aligned with videos
        """
        self.videos = videos
        self.clips = clips
        self.video_labels = video_labels
        self.num_frames = config['dataset']['num_frames']
        self.frame_size = config['dataset']['frame_size']
        self.motion_norm_range = float(config['dataset']['motion_norm_range'])
        self.hflip = train and config['training'].get('augment_hflip', True)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid_idx, start = self.clips[idx]
        flip = self.hflip and random.random() < 0.5
        clip = self.videos[vid_idx].clip(
            start, self.num_frames, self.frame_size, self.motion_norm_range, flip=flip
        )
        return torch.from_numpy(clip), self.video_labels[vid_idx], vid_idx

    def clip_labels(self):
        return [self.video_labels[v] for v, _ in self.clips]


# ============================================================
# SPLITTING
# ============================================================

def split_videos(video_names, labels, val_ratio, seed):
    """
    Stratified video-level split (no clip of a val video is ever seen in training).
    Returns None if some class has fewer than 2 videos.
    """
    by_class = defaultdict(list)
    for name in video_names:
        by_class[labels[name]].append(name)
    if any(len(v) < 2 for v in by_class.values()):
        return None

    rng = random.Random(seed)
    train, val = [], []
    for cls in sorted(by_class):
        names = sorted(by_class[cls])
        rng.shuffle(names)
        n_val = min(len(names) - 1, max(1, int(round(len(names) * val_ratio))))
        val.extend(names[:n_val])
        train.extend(names[n_val:])
    return sorted(train), sorted(val)


def build_datasets(config, root_dir, logger=print):
    """
    Loads every prepared video, assigns labels, splits train/val and builds clip lists.

    Returns:
        train_set, val_set, info (dict with class_names, split details)
    """
    ds_cfg = config['dataset']
    split_cfg = config['data_split']
    num_frames = ds_cfg['num_frames']

    def resolve(p):
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else Path(root_dir) / p

    mv_root = resolve(ds_cfg['motion_vectors_dir'])
    bbox_root = resolve(ds_cfg['bboxes_dir'])
    manifest = load_manifest(resolve(ds_cfg.get('manifest_csv')))

    if not mv_root.exists():
        raise FileNotFoundError(
            f"Motion vector directory not found: {mv_root}\n"
            f"Run prepare_action_dataset.py first."
        )

    # ---------- Load videos ----------
    candidate_names = sorted(p.name for p in mv_root.iterdir() if p.is_dir())
    videos, skipped = [], []
    for name in candidate_names:
        bbox_csv = bbox_root / f"{name}_bboxes.csv"
        if not bbox_csv.exists():
            skipped.append((name, 'no bbox csv'))
            continue
        meta = manifest.get(name, {})
        try:
            video = VideoMotionData(
                name, mv_root / name, bbox_csv,
                width=int(meta['width']) if meta.get('width') else None,
                height=int(meta['height']) if meta.get('height') else None,
                pad_ratio=ds_cfg.get('bbox_padding', 0.15),
                box_mode=ds_cfg.get('box_mode', 'largest'),
                max_frame_gap=ds_cfg.get('max_frame_gap', 4)
            )
        except ValueError as e:
            skipped.append((name, str(e)))
            continue
        if not video.clip_starts(num_frames, 1):
            skipped.append((name, f'fewer than {num_frames} consecutive frames with boxes'))
            continue
        videos.append(video)

    for name, reason in skipped:
        logger(f"  [skip] {name}: {reason}")
    if not videos:
        raise RuntimeError(f"No usable videos found in {mv_root}")

    # ---------- Labels ----------
    names = [v.name for v in videos]
    labels = resolve_labels(names, manifest, resolve(ds_cfg.get('labels_csv')))
    class_names = sorted(set(labels.values()))
    if len(class_names) < 2:
        raise RuntimeError(
            f"Found only one action class ({class_names}). Training needs at least 2.\n"
            f"Put videos in class sub-folders (e.g. videos/Walking/*.mp4) before running "
            f"prepare_action_dataset.py, or provide dataset.labels_csv."
        )
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    video_labels = [class_to_idx[labels[n]] for n in names]

    # ---------- Split ----------
    strategy = split_cfg.get('strategy', 'auto')
    val_ratio = split_cfg.get('val_ratio', 0.2)
    seed = split_cfg.get('random_seed', 42)
    train_stride = ds_cfg.get('clip_stride', 4)
    val_stride = ds_cfg.get('val_clip_stride', num_frames)

    video_split = None
    if strategy in ('auto', 'video'):
        video_split = split_videos(names, labels, val_ratio, seed)
        if video_split is None and strategy == 'video':
            raise RuntimeError("strategy='video' needs at least 2 videos per class")

    train_clips, val_clips = [], []
    if video_split is not None:
        used_strategy = 'video'
        val_names = set(video_split[1])
        for i, v in enumerate(videos):
            if v.name in val_names:
                val_clips += [(i, s) for s in v.clip_starts(num_frames, val_stride)]
            else:
                train_clips += [(i, s) for s in v.clip_starts(num_frames, train_stride)]
    else:
        # Too few videos per class: split each video in time
        # (first part -> train, last part -> val, no overlapping frames)
        used_strategy = 'temporal'
        for i, v in enumerate(videos):
            cut = int(len(v) * (1 - val_ratio))
            tr = v.clip_starts(num_frames, train_stride, hi=cut)
            va = v.clip_starts(num_frames, val_stride, lo=cut)
            if not tr or not va:
                tr = v.clip_starts(num_frames, train_stride)
                va = []
            train_clips += [(i, s) for s in tr]
            val_clips += [(i, s) for s in va]

    if not val_clips:
        raise RuntimeError("Validation split is empty - add more data or lower num_frames")

    train_set = MotionClipDataset(videos, train_clips, video_labels, config, train=True)
    val_set = MotionClipDataset(videos, val_clips, video_labels, config, train=False)

    info = {
        'class_names': class_names,
        'split_strategy': used_strategy,
        'video_labels': {n: labels[n] for n in names},
        'train_videos': sorted({videos[v].name for v, _ in train_clips}),
        'val_videos': sorted({videos[v].name for v, _ in val_clips}),
        'num_train_clips': len(train_clips),
        'num_val_clips': len(val_clips),
        'train_clips_per_class': dict(Counter(class_names[l] for l in train_set.clip_labels())),
        'val_clips_per_class': dict(Counter(class_names[l] for l in val_set.clip_labels())),
        'skipped_videos': skipped,
    }
    return train_set, val_set, info


def get_dataloaders(config, root_dir, logger=print):
    """Create train and validation dataloaders (+ dataset info)"""
    train_set, val_set, info = build_datasets(config, root_dir, logger)
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', False) and torch.cuda.is_available()

    # Workers build clips (resize/normalize) in parallel so the GPU is not starved
    worker_kwargs = {'persistent_workers': True, 'prefetch_factor': 4} if num_workers > 0 else {}

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=len(train_set) > batch_size, **worker_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, **worker_kwargs)
    return train_loader, val_loader, info


def save_split_info(info, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
