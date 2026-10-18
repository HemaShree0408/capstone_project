# src/utils.py
"""
Utility functions for data processing, logging, and misc operations
"""

import os
import random
import yaml
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(log_dir, log_file):
    """Setup logging to file and console"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger('ActionRecognition')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_device(device_name='cuda', require_cuda=False):
    """
    Get torch device.
    require_cuda=True raises instead of silently falling back to the CPU.
    """
    if device_name == 'cuda' and not torch.cuda.is_available() and require_cuda:
        raise RuntimeError(
            "CUDA GPU requested but not available to PyTorch "
            f"(torch {torch.__version__}, CUDA build: {torch.version.cuda}).\n"
            "Install the CUDA build of PyTorch, e.g.:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "(or pick the command for your CUDA version at https://pytorch.org/get-started)\n"
            "and check the NVIDIA driver with `nvidia-smi`. "
            "Use --device cpu to train on the CPU anyway."
        )
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB, "
              f"CUDA {torch.version.cuda})")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Ratio: {100 * trainable / total:.1f}%")
    
    return trainable, total


def normalize_motion_values(flow, min_val=-100.0, max_val=100.0):
    """Normalize motion values to [-1, 1] range"""
    flow_normalized = (flow - min_val) / (max_val - min_val) * 2 - 1
    # Clip to ensure within [-1, 1]
    flow_normalized = np.clip(flow_normalized, -1.0, 1.0)
    return flow_normalized


def denormalize_motion_values(flow_norm, min_val=-100.0, max_val=100.0):
    """Denormalize from [-1, 1] back to original range"""
    flow = (flow_norm + 1) / 2 * (max_val - min_val) + min_val
    return flow


class AverageMeter:
    """Compute and store average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        Args:
            patience: How many epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        """
        Args:
            val_score: Validation metric (accuracy, loss, etc.)
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True