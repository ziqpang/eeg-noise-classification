import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    """CNN with simple time-wise attention for EEG noise classification"""
    def __init__(self, num_classes=3):
        super(AttentionClassifier, self).__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1)
        self.attn_norm = nn.LayerNorm(256)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, time]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch,1,time]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Prepare for self-attention: [batch, C, T] -> [T, batch, C]
        x2 = x.permute(2, 0, 1)
        attn_out, _ = self.attn(x2, x2, x2)
        # Residual and normalization
        x2 = self.attn_norm(x2 + attn_out)
        # back to [batch, C, T]
        x = x2.permute(1, 2, 0)
        # Global average pooling
        x = x.mean(dim=2)
        return self.classifier(x)