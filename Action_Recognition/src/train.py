# src/train.py
"""
Training script for the motion-vector action recognition model (TemporalCNN).

Usage (from the Action_Recognition folder):
    python main.py train --config config/config.yaml
    python main.py train --config config/config.yaml --epochs 50 --lr 5e-5
    python main.py train --config config/config.yaml --resume models/last_model.pth
"""

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .model import TemporalCNN, freeze_backbone
from .data_loader import get_dataloaders, save_split_info
from .utils import (
    load_config, setup_logger, set_seed, ensure_dir,
    get_device, count_parameters, AverageMeter, EarlyStopping
)

# Action_Recognition/ - relative paths in the config are resolved against it
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Trainer:
    """Trainer class for model training"""

    def __init__(self, config, device, logger, class_names, class_weights=None):
        self.config = config
        self.device = device
        self.logger = logger
        self.class_names = class_names
        train_cfg = config['training']

        # Create model
        self.logger.info("Creating model...")
        self.model = TemporalCNN(
            num_classes=len(class_names),
            input_channels=2 * config['dataset']['num_frames'],
            dropout_fc6=config['model']['dropout_fc6'],
            dropout_fc7=config['model']['dropout_fc7']
        )
        if config['model'].get('freeze_backbone', False):
            freeze_backbone(self.model)
        self.model.to(device)
        count_parameters(self.model)

        # Loss function (operates on logits, NOT on softmax output)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=train_cfg.get('label_smoothing', 0.0)
        )

        # Optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if train_cfg.get('optimizer', 'adam').lower() == 'sgd':
            self.optimizer = SGD(
                trainable_params,
                lr=train_cfg['learning_rate'],
                momentum=train_cfg['momentum'],
                weight_decay=train_cfg['weight_decay']
            )
        else:
            self.optimizer = Adam(
                trainable_params,
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg['weight_decay']
            )

        # Learning rate scheduler
        sched_cfg = train_cfg['scheduler']
        if sched_cfg['type'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=sched_cfg.get('step_size', 10),
                                    gamma=sched_cfg['factor'])
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_cfg['factor'],
                patience=sched_cfg['patience'],
                min_lr=float(sched_cfg['min_lr'])
            )

        # Early stopping
        es_cfg = train_cfg['early_stopping']
        self.early_stopping = EarlyStopping(patience=es_cfg['patience'], verbose=True) \
            if es_cfg.get('enabled', True) else None

        # Checkpointing
        self.checkpoint_dir = PROJECT_ROOT / train_cfg['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)

        # Mixed precision (GPU only): bf16 on Ampere+ (RTX 30xx/40xx/50xx), else fp16 + loss scaling
        self.use_amp = bool(train_cfg.get('mixed_precision', True)) and device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() \
            else torch.float16
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        if self.use_amp:
            self.logger.info(f"Mixed precision: {self.amp_dtype}")

        self.start_epoch = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.history = []

    # --------------------------------------------------------
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        log_every = max(1, len(train_loader) // 10)

        for batch_idx, (motion_vectors, labels, _) in enumerate(train_loader):
            motion_vectors = motion_vectors.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass - use logits for the loss
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                _, logits = self.model(motion_vectors, return_logits=True)
                loss = self.criterion(logits, labels)

            # Backward pass (scaler is a no-op unless fp16 AMP is active)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['gradient_clip']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            n = labels.size(0)
            acc = logits.argmax(1).eq(labels).sum().item() / n
            loss_meter.update(loss.item(), n)
            acc_meter.update(acc, n)

            if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == len(train_loader):
                print(f"  [train] batch {batch_idx + 1}/{len(train_loader)} "
                      f"loss {loss_meter.avg:.4f} acc {acc_meter.avg:.2%} "
                      f"lr {self.optimizer.param_groups[0]['lr']:.2e}", flush=True)

        return loss_meter.avg, acc_meter.avg

    # --------------------------------------------------------
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate on clips.
        Returns clip-level loss/accuracy plus video-level accuracy
        (clip probabilities averaged per video) and the confusion matrix.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        num_classes = len(self.class_names)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        video_probs = defaultdict(list)
        video_label = {}

        for motion_vectors, labels, vid_idx in val_loader:
            motion_vectors = motion_vectors.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                probs, logits = self.model(motion_vectors, return_logits=True)
                loss = self.criterion(logits, labels)
            loss_meter.update(loss.item(), labels.size(0))

            preds = logits.argmax(1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t, p] += 1
            for v, prob, t in zip(vid_idx.tolist(), probs.float().cpu().numpy(), labels.tolist()):
                video_probs[v].append(prob)
                video_label[v] = t

        clip_acc = np.trace(confusion) / max(1, confusion.sum())
        video_correct = [int(np.mean(video_probs[v], axis=0).argmax() == video_label[v])
                         for v in video_probs]
        video_acc = float(np.mean(video_correct)) if video_correct else 0.0
        return loss_meter.avg, float(clip_acc), video_acc, confusion

    # --------------------------------------------------------
    def save_checkpoint(self, path, epoch, acc):
        """Save model checkpoint (self-contained: includes classes and config)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'accuracy': acc,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'class_names': self.class_names,
            'config': self.config,
            'history': self.history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Resume training from a checkpoint saved by save_checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get('class_names') != self.class_names:
            raise RuntimeError(
                f"Checkpoint classes {checkpoint.get('class_names')} do not match "
                f"dataset classes {self.class_names}"
            )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict') and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint.get('history', [])
        self.logger.info(f"Resumed from {path} (epoch {self.start_epoch}, best acc {self.best_acc:.2%})")

    # --------------------------------------------------------
    def log_confusion(self, confusion):
        width = max(len(c) for c in self.class_names) + 2
        self.logger.info("Confusion matrix (rows = true, cols = predicted):")
        self.logger.info(" " * width + "".join(f"{c[:8]:>9}" for c in self.class_names))
        for name, row in zip(self.class_names, confusion):
            per_class = row[self.class_names.index(name)] / max(1, row.sum())
            self.logger.info(f"{name:<{width}}" + "".join(f"{v:>9d}" for v in row) +
                             f"   acc {per_class:.2%}")

    def train(self, train_loader, val_loader, num_epochs):
        """Train for multiple epochs"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)

        # 'clip' or 'video' accuracy drives model selection / LR schedule / early stopping
        metric = self.config['training'].get('selection_metric', 'clip')
        best_confusion = None

        for epoch in range(self.start_epoch, num_epochs):
            t0 = time.time()
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.logger.info("-" * 60)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_clip_acc, val_video_acc, confusion = self.validate(val_loader)
            val_acc = val_video_acc if metric == 'video' else val_clip_acc

            self.logger.info(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            self.logger.info(f"Val   | Loss: {val_loss:.4f} | Clip Acc: {val_clip_acc:.2%} "
                             f"| Video Acc: {val_video_acc:.2%} | {time.time() - t0:.0f}s")

            self.history.append({
                'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_clip_acc': val_clip_acc,
                'val_video_acc': val_video_acc, 'lr': self.optimizer.param_groups[0]['lr']
            })

            # Save best model (always save after the very first epoch of a fresh run)
            if val_acc > self.best_acc or (epoch == 0 and not self.history[:-1]):
                self.best_acc = val_acc
                self.best_epoch = epoch
                best_confusion = confusion
                self.save_checkpoint(self.checkpoint_dir / 'best_model.pth', epoch, val_acc)
                self.logger.info(f"[BEST] New best model! Val acc: {val_acc:.2%} -> "
                                 f"{self.checkpoint_dir / 'best_model.pth'}")
            else:
                self.logger.info(f"Best Acc: {self.best_acc:.2%} (epoch {self.best_epoch + 1})")

            self.save_checkpoint(self.checkpoint_dir / 'last_model.pth', epoch, val_acc)

            # Step scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_acc)
                if self.early_stopping.early_stop:
                    self.logger.info(f"[STOP] Early stopping at epoch {epoch + 1}")
                    break

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Best Val Accuracy ({metric}): {self.best_acc:.2%} (Epoch {self.best_epoch + 1})")
        if best_confusion is not None:
            self.log_confusion(best_confusion)
        self.logger.info("=" * 60)

        with open(self.checkpoint_dir / 'training_history.json', 'w', encoding='utf-8') as f:
            json.dump({'class_names': self.class_names, 'best_acc': self.best_acc,
                       'best_epoch': self.best_epoch + 1, 'selection_metric': metric,
                       'history': self.history}, f, indent=2)


def compute_class_weights(train_set, num_classes):
    """Inverse-frequency class weights over training clips (mean weight = 1)"""
    counts = np.bincount(train_set.clip_labels(), minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return weights.tolist()


def main():
    parser = argparse.ArgumentParser(
        description='Train Temporal CNN for action recognition'
    )
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
                        help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, help='Override training.num_epochs')
    parser.add_argument('--batch-size', type=int, help='Override training.batch_size')
    parser.add_argument('--lr', type=float, help='Override training.learning_rate')
    parser.add_argument('--device', help="Override device ('cuda' or 'cpu')")
    parser.add_argument('--resume', help='Resume from a checkpoint (e.g. models/last_model.pth)')

    args, _ = parser.parse_known_args()

    # Load config (+ command line overrides)
    config = copy.deepcopy(load_config(args.config))
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['device'] = args.device

    # Setup
    set_seed(config['seed'])
    # Training is meant for an NVIDIA GPU: fail loudly instead of silently using the CPU
    device = get_device(config['device'], require_cuda=(config['device'] == 'cuda'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # fixed input size -> let cuDNN pick fastest kernels
    logger = setup_logger(
        PROJECT_ROOT / config['logging']['log_dir'],
        config['logging']['log_file']
    )
    logger.info("Configuration loaded from: " + args.config)
    logger.info(f"Device: {device}")

    # Create data loaders
    logger.info("Loading dataset (rasterising motion vectors)...")
    train_loader, val_loader, info = get_dataloaders(config, PROJECT_ROOT, logger.info)
    class_names = info['class_names']

    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Split strategy: {info['split_strategy']}")
    logger.info(f"Train: {len(info['train_videos'])} videos, {info['num_train_clips']} clips "
                f"{info['train_clips_per_class']}")
    logger.info(f"Val  : {len(info['val_videos'])} videos, {info['num_val_clips']} clips "
                f"{info['val_clips_per_class']}")
    if info['split_strategy'] == 'temporal':
        logger.info("NOTE: some class has <2 videos, so each video was split in time "
                    "(first part train / last part val). Validation accuracy will be "
                    "optimistic - add more videos per class for a real evaluation.")

    ensure_dir(PROJECT_ROOT / config['training']['checkpoint_dir'])
    save_split_info(info, PROJECT_ROOT / config['training']['checkpoint_dir'] / 'dataset_split.json')

    # Create trainer
    class_weights = compute_class_weights(train_loader.dataset, len(class_names)) \
        if config['training'].get('class_weighted_loss', True) else None
    trainer = Trainer(config, device, logger, class_names, class_weights)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader, num_epochs=config['training']['num_epochs'])


if __name__ == '__main__':
    main()
