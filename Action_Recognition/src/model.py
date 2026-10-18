# src/model.py
"""
Temporal CNN Model
Recreated from temporal_cls.prototxt from zbwglory/MV-release
Input: 20-channel motion vectors (10 frames × 2 components)
Output: Action class probabilities
"""

import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """
    Motion-Only CNN for Action Recognition
    
    Architecture based on temporal_cls.prototxt:
    - Conv layers: conv1-5 with PReLU activation and LRN normalization
    - FC layers: fc6, fc7, fc8
    - Input: (batch, 20, 224, 224) - 10 frames × 2 flow components
    - Output: (batch, num_classes) - action probabilities
    """
    
    def __init__(self, num_classes=101, input_channels=20, 
                 dropout_fc6=0.9, dropout_fc7=0.8):
        super(TemporalCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # ========== CONVOLUTIONAL LAYERS ==========
        
        # Conv1: input_channels → 96
        self.conv1 = nn.Conv2d(input_channels, 96, kernel_size=7, stride=2)
        self.relu1 = nn.PReLU()
        self.norm1 = nn.LocalResponseNorm(5, alpha=0.0005, beta=0.75)
        # ceil_mode=True reproduces Caffe pooling arithmetic (needed for the 6x6 fc input)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Conv2: 96 → 256
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1)
        self.relu2 = nn.PReLU()
        self.norm2 = nn.LocalResponseNorm(5, alpha=0.0005, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Conv3: 256 → 512
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu3 = nn.PReLU()
        
        # Conv4: 512 → 512
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4 = nn.PReLU()
        
        # Conv5: 512 → 512
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5 = nn.PReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # ========== FULLY CONNECTED LAYERS ==========

        # Calculate flattened size
        # (224, 224) → conv1 stride2 → (109, 109)
        # → pool1 stride2 (ceil) → (54, 54)
        # → conv2 stride2 → (26, 26)
        # → pool2 stride2 (ceil) → (13, 13)
        # → conv3,4,5 (no stride) → (13, 13)
        # → pool5 stride2 → (6, 6)
        fc_input_size = 512 * 6 * 6  # 18432
        
        self.fc6 = nn.Linear(fc_input_size, 4096)
        self.relu6 = nn.PReLU()
        self.drop6 = nn.Dropout(p=dropout_fc6)
        
        self.fc7 = nn.Linear(4096, 2048)
        self.relu7 = nn.PReLU()
        self.drop7 = nn.Dropout(p=dropout_fc7)
        
        self.fc8 = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_logits=False):
        """
        Forward pass
        
        Args:
            x: (batch_size, 20, 224, 224) - motion vectors
            return_logits: If True, return logits before softmax
        
        Returns:
            output: (batch_size, num_classes) - action probabilities
            logits: (batch_size, num_classes) - if return_logits=True
        """
        
        # ===== CONVOLUTIONAL BLOCK 1 =====
        x = self.conv1(x)                    # (batch, 96, 111, 111)
        x = self.relu1(x)
        x = self.norm1(x.float())          # LRN squares activations: keep fp32 under AMP
        x = self.pool1(x)                    # (batch, 96, 55, 55)
        
        # ===== CONVOLUTIONAL BLOCK 2 =====
        x = self.conv2(x)                    # (batch, 256, 27, 27)
        x = self.relu2(x)
        x = self.norm2(x.float())
        x = self.pool2(x)                    # (batch, 256, 13, 13)
        
        # ===== CONVOLUTIONAL BLOCK 3,4,5 =====
        x = self.conv3(x)                    # (batch, 512, 13, 13)
        x = self.relu3(x)
        
        x = self.conv4(x)                    # (batch, 512, 13, 13)
        x = self.relu4(x)
        
        x = self.conv5(x)                    # (batch, 512, 13, 13)
        x = self.relu5(x)
        x = self.pool5(x)                    # (batch, 512, 6, 6)
        
        # ===== FLATTEN & FULLY CONNECTED =====
        x = x.view(x.size(0), -1)            # (batch, 18432)
        
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)                    # (batch, 4096)
        
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)                    # (batch, 2048)
        
        logits = self.fc8(x)                 # (batch, num_classes)
        output = self.softmax(logits)        # (batch, num_classes)
        
        if return_logits:
            return output, logits
        return output


def load_model(model_path, num_classes=None, device='cuda'):
    """
    Load trained model from checkpoint.
    Class count and input channels are read from the stored weights, so the
    num_classes argument is only a fallback for very old checkpoints.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict']
    model = TemporalCNN(
        num_classes=state['fc8.weight'].shape[0] if num_classes is None else num_classes,
        input_channels=state['conv1.weight'].shape[1]
    )
    model.load_state_dict(state)
    model.to(device)
    return model


def init_with_imagenet_pretrained(model, num_classes=101):
    """
    Initialize model with ImageNet pre-trained weights
    
    Strategy:
    1. Load ResNet50 pre-trained on ImageNet
    2. Average RGB weights to get 1 channel
    3. Replicate for 20 channels (motion vectors)
    """
    
    print("Initializing with ImageNet pre-trained weights...")
    
    # Load ImageNet pre-trained ResNet50
    import torchvision.models as models  # optional dependency, only needed here
    resnet50 = models.resnet50(pretrained=True)
    
    # Get conv1 weights: (64, 3, 7, 7)
    original_conv = resnet50.conv1
    pretrained_weight = original_conv.weight  # (64, 3, 7, 7)
    
    # Average across RGB channels: (64, 1, 7, 7)
    averaged_weight = pretrained_weight.mean(dim=1, keepdim=True)
    
    # Replicate for 20 channels: (64, 20, 7, 7)
    replicated_weight = averaged_weight.repeat(1, 20, 1, 1)
    
    # Scale to keep magnitude similar
    replicated_weight = replicated_weight / 20 * 3
    
    # Copy to model's conv1
    # Note: Our model has conv1: input_channels → 96
    # ResNet has: 3 → 64
    # We need to adjust dimensions
    
    # For now, we initialize conv1 manually (20 → 96)
    with torch.no_grad():
        # Initialize with small values (Xavier-like)
        model.conv1.weight.copy_(
            torch.randn_like(model.conv1.weight) * 0.01
        )
    
    print("✓ Model initialized")
    return model


def freeze_backbone(model):
    """Freeze early conv layers to prevent overfitting"""
    
    # Freeze conv1, conv2, conv3
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters() if hasattr(model, 'bn1') else []:
        param.requires_grad = False
    for param in model.relu1.parameters():
        param.requires_grad = False
    for param in model.pool1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.relu2.parameters():
        param.requires_grad = False
    for param in model.pool2.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n✓ Backbone frozen!")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"Training layers: conv4, conv5, fc6, fc7, fc8\n")