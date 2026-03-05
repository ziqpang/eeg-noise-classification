import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_length=512, num_classes=3):
        super(CNNLSTMClassifier, self).__init__()
        
        # First stage: Low and mid frequency feature extraction
        self.low_mid_features = nn.Sequential(
            # Low frequency block
            nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Mid frequency block
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Second stage: High frequency feature extraction (EMG-focused)
        self.high_freq_features = nn.Sequential(
            # Parallel small kernels for high frequency
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Residual connection with maxpool to match dimensions
        self.residual = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=1),
            nn.MaxPool1d(2)  # Match the spatial dimension
        )
        
        # Calculate sequence length after CNN
        self.seq_len = input_length // 8  # After 3 MaxPool layers
        
        # Enhanced LSTM for temporal modeling
        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=256,  # 128*2 from bidirectional
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Multiple pooling heads
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 256*2 from concatenated pools
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, time]
        
        # Low and mid frequency feature extraction
        x = self.low_mid_features(x)  # [batch, 64, time/4]
        
        # Store for residual connection
        identity = self.residual(x)  # [batch, 256, time/8]
        
        # High frequency feature extraction
        x = self.high_freq_features(x)  # [batch, 256, time/8]
        
        # Add residual connection
        x = x + identity  # Now dimensions match
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # [batch, time/8, 256]
        
        # First LSTM layer
        x, _ = self.lstm1(x)  # [batch, time/8, 256]
        
        # Second LSTM layer
        x, _ = self.lstm2(x)  # [batch, time/8, 256]
        
        # Transpose back for pooling
        x = x.transpose(1, 2)  # [batch, 256, time/8]
        
        # Multiple pooling operations
        max_pooled = self.max_pool(x).squeeze(-1)  # [batch, 256]
        avg_pooled = self.avg_pool(x).squeeze(-1)  # [batch, 256]
        
        # Concatenate pooled features
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # [batch, 512]
        
        # Classification
        output = self.classifier(pooled)
        
        return output 