# src/inference.py
"""
Inference script for action recognition.
Predicts the action of a video prepared by prepare_action_dataset.py, using
exactly the same clip construction as training (src/data_loader.py).

Usage (from the Action_Recognition folder):
    python main.py inference --model models/best_model.pth \
        --motion-dir data/raw/motion_vectors/<video> \
        --bboxes-csv data/raw/bboxes/<video>_bboxes.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import load_model
from .data_loader import VideoMotionData
from .utils import get_device


class ActionRecognizer:
    """Inference pipeline for action recognition"""

    def __init__(self, model_path, device='cuda', clip_stride=None, batch_size=32):
        """
        Args:
            model_path: checkpoint written by src/train.py (contains classes + config)
            device: 'cuda' or 'cpu'
        """
        self.device = get_device(device)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']
        self.class_names = checkpoint['class_names']

        self.model = load_model(model_path, device=self.device)
        self.model.eval()

        ds_cfg = self.config['dataset']
        self.num_frames = ds_cfg['num_frames']
        self.frame_size = ds_cfg['frame_size']
        self.motion_norm_range = float(ds_cfg['motion_norm_range'])
        self.pad_ratio = ds_cfg.get('bbox_padding', 0.15)
        self.box_mode = ds_cfg.get('box_mode', 'largest')
        self.max_frame_gap = ds_cfg.get('max_frame_gap', 4)
        self.clip_stride = clip_stride or self.config.get('inference', {}).get('clip_stride', 5)
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, motion_vectors_dir, bboxes_csv, width=None, height=None):
        """
        Predict the action of one video by averaging the probabilities of all its clips.

        Returns:
            result: dict with the prediction, top-k and per-clip predictions
        """
        video = VideoMotionData(
            Path(motion_vectors_dir).name, motion_vectors_dir, bboxes_csv,
            width=width, height=height, pad_ratio=self.pad_ratio,
            box_mode=self.box_mode, max_frame_gap=self.max_frame_gap
        )
        starts = video.clip_starts(self.num_frames, self.clip_stride)
        if not starts:
            raise ValueError(
                f"Video needs at least {self.num_frames} consecutive P-frames with "
                f"bounding boxes, found {len(video)} usable frames"
            )

        all_probs = []
        for i in range(0, len(starts), self.batch_size):
            batch = np.stack([
                video.clip(s, self.num_frames, self.frame_size, self.motion_norm_range)
                for s in starts[i:i + self.batch_size]
            ])
            probs = self.model(torch.from_numpy(batch).to(self.device))
            all_probs.append(probs.cpu().numpy())
        clip_probs = np.concatenate(all_probs)
        probs = clip_probs.mean(axis=0)

        predicted_class = int(np.argmax(probs))
        top_k = np.argsort(probs)[::-1][:5]
        return {
            'predicted_class_id': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': float(probs[predicted_class]),
            'all_probabilities': dict(zip(self.class_names, probs.tolist())),
            'top_5': [
                {'rank': r + 1, 'class_id': int(c), 'class_name': self.class_names[c],
                 'probability': float(probs[c])}
                for r, c in enumerate(top_k)
            ],
            'clips': [
                {'start_frame': video.frame_ids[s],
                 'end_frame': video.frame_ids[s + self.num_frames - 1],
                 'class_name': self.class_names[int(p.argmax())],
                 'confidence': float(p.max())}
                for s, p in zip(starts, clip_probs)
            ],
        }


def main():
    parser = argparse.ArgumentParser(description='Perform action recognition inference')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--motion-dir', required=True, help='Path to the video motion vectors directory')
    parser.add_argument('--bboxes-csv', required=True, help='Path to the video bboxes CSV file')
    parser.add_argument('--device', default='cuda', help="'cuda' or 'cpu'")
    parser.add_argument('--show-clips', action='store_true', help='Print per-clip predictions')

    args, _ = parser.parse_known_args()

    recognizer = ActionRecognizer(model_path=args.model, device=args.device)

    print("Performing inference...")
    result = recognizer.predict(args.motion_dir, args.bboxes_csv)

    print("\n" + "=" * 60)
    print(f"Predicted Action: {result['predicted_class_name']}")
    print(f"Confidence: {result['confidence']:.4f}  (averaged over {len(result['clips'])} clips)")
    print("\nTop Predictions:")
    for pred in result['top_5']:
        print(f"  {pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")
    if args.show_clips:
        print("\nPer-clip predictions:")
        for c in result['clips']:
            print(f"  frames {c['start_frame']:>5}-{c['end_frame']:<5} "
                  f"{c['class_name']} ({c['confidence']:.2f})")
    print("=" * 60)


if __name__ == '__main__':
    main()
