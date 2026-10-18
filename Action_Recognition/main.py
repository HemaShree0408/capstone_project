# main.py
"""
Main entry point for the action recognition project
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='Motion Vector CNN for Action Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Prepare data (from the project root)
  python prepare_action_dataset.py --video_dir <videos_root_with_class_subfolders>

  # 2. Train model (labels and train/val split are created automatically)
  python main.py train --config config/config.yaml

  # 3. Run inference
  python main.py inference --model models/best_model.pth \\
                          --motion-dir data/raw/motion_vectors/video_name \\
                          --bboxes-csv data/raw/bboxes/video_name_bboxes.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', default=str(Path(__file__).parent / 'config' / 'config.yaml'))
    train_parser.add_argument('--epochs', type=int)
    train_parser.add_argument('--batch-size', type=int)
    train_parser.add_argument('--lr', type=float)
    train_parser.add_argument('--device')
    train_parser.add_argument('--resume')

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--model', required=True)
    inference_parser.add_argument('--motion-dir', required=True)
    inference_parser.add_argument('--bboxes-csv', required=True)
    inference_parser.add_argument('--device', default='cuda')
    inference_parser.add_argument('--show-clips', action='store_true')
    inference_parser.add_argument('--config', help='Unused (config is stored in the checkpoint)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Sub-modules re-parse sys.argv themselves; import lazily so that
    # each command only needs its own dependencies.
    if args.command == 'train':
        from src.train import main as train_main
        train_main()
    elif args.command == 'inference':
        from src.inference import main as inference_main
        inference_main()


if __name__ == '__main__':
    main()
