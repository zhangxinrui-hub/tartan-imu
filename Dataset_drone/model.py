import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, has_downsample=False):
        super(ResidualBlock, self).__init__()
        
        # Main convolutions - using Sequential to match checkpoint structure
        # Structure: conv0, bn0, relu, conv1, bn1
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3,
                     padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        # Downsampling path
        self.downsample = None
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.convs(x)
        
        # Downsampling path
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual
        out += identity
        out = F.relu(out)
        return out


class ModelWithLSTM(nn.Module):
    def __init__(self, input_channels=6, lstm_hidden=64):
        super(ModelWithLSTM, self).__init__()
        
        # Input block - Conv + BN + ReLU (no bias in Conv)
        self.input_block = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual groups - using nested ModuleList to match checkpoint
        self.residual_groups = nn.ModuleList([
            # Group 0: 2 blocks, 64 channels, no downsample
            nn.ModuleList([
                ResidualBlock(64, 64, stride=1, has_downsample=False),
                ResidualBlock(64, 64, stride=1, has_downsample=False)
            ]),
            # Group 1: 2 blocks, 128 channels, with downsample
            nn.ModuleList([
                ResidualBlock(64, 128, stride=2, has_downsample=True),
                ResidualBlock(128, 128, stride=1, has_downsample=False)
            ]),
            # Group 2: 2 blocks, 256 channels, with downsample
            nn.ModuleList([
                ResidualBlock(128, 256, stride=2, has_downsample=True),
                ResidualBlock(256, 256, stride=1, has_downsample=False)
            ])
        ])
        
        # Post processing - 1x1 convolutions (no bias)
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # LSTM layer - part of model namespace
        self.lstm = nn.LSTM(1664, lstm_hidden, batch_first=True)
    
    def forward(self, x):
        # x shape: [batch, 6, seq_len]
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.input_block(x)
        
        for group in self.residual_groups:
            for block in group:
                x = block(x)
        
        x = self.resnet_post_pro(x)  # [batch, 128, seq_len]
        
        # Reshape for LSTM: [batch, seq_len, 1664]
        # [batch, 128, seq_len] -> [batch, seq_len, 128]
        x = x.transpose(1, 2)
        
        # [batch, seq_len, 128] -> [batch, seq_len, 1664]
        # Repeat the 128 features 13 times: 128 * 13 = 1664
        x = x.repeat(1, 1, 13)  # [batch, seq_len, 128*13] = [batch, seq_len, 1664]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 64]
        
        return lstm_out


class OutputHead(nn.Module):
    def __init__(self, input_dim=64):
        super(OutputHead, self).__init__()
        
        # Using ModuleList to match checkpoint structure (fcs.0, fcs.3, fcs.6)
        # Each MLP has 3 linear layers at indices 0, 3, 6 with ReLUs in between
        
        # Output block 1 - for 2D output
        self.output_block1 = nn.ModuleDict({
            'fcs': nn.ModuleList([
                nn.Linear(input_dim, 256),  # index 0
                nn.ReLU(inplace=True),      # index 1
                nn.Identity(),              # index 2 (placeholder)
                nn.Linear(256, 256),        # index 3
                nn.ReLU(inplace=True),      # index 4
                nn.Identity(),              # index 5 (placeholder)
                nn.Linear(256, 2)           # index 6
            ])
        })
        
        # Output block 2 - for 3D output
        self.output_block2 = nn.ModuleDict({
            'fcs': nn.ModuleList([
                nn.Linear(input_dim, 256),  # index 0
                nn.ReLU(inplace=True),      # index 1
                nn.Identity(),              # index 2 (placeholder)
                nn.Linear(256, 256),        # index 3
                nn.ReLU(inplace=True),      # index 4
                nn.Identity(),              # index 5 (placeholder)
                nn.Linear(256, 3)           # index 6
            ])
        })
        
        # Output block for Z coordinate
        self.output_block1_z = nn.ModuleDict({
            'fcs': nn.ModuleList([
                nn.Linear(input_dim, 256),  # index 0
                nn.ReLU(inplace=True),      # index 1
                nn.Identity(),              # index 2 (placeholder)
                nn.Linear(256, 256),        # index 3
                nn.ReLU(inplace=True),      # index 4
                nn.Identity(),              # index 5 (placeholder)
                nn.Linear(256, 1)           # index 6
            ])
        })
    
    def forward(self, x):
        # Output block 1
        out1 = self.output_block1['fcs'][0](x)
        out1 = self.output_block1['fcs'][1](out1)
        out1 = self.output_block1['fcs'][3](out1)
        out1 = self.output_block1['fcs'][4](out1)
        out1 = self.output_block1['fcs'][6](out1)
        
        # Output block 2
        out2 = self.output_block2['fcs'][0](x)
        out2 = self.output_block2['fcs'][1](out2)
        out2 = self.output_block2['fcs'][3](out2)
        out2 = self.output_block2['fcs'][4](out2)
        out2 = self.output_block2['fcs'][6](out2)
        
        # Output block z
        out_z = self.output_block1_z['fcs'][0](x)
        out_z = self.output_block1_z['fcs'][1](out_z)
        out_z = self.output_block1_z['fcs'][3](out_z)
        out_z = self.output_block1_z['fcs'][4](out_z)
        out_z = self.output_block1_z['fcs'][6](out_z)
        
        return out1, out2, out_z


class TartanIMUModel(nn.Module):
    def __init__(self, input_channels=6, lstm_hidden=64):
        super(TartanIMUModel, self).__init__()
        
        # Main model containing CNN + LSTM - named 'model' to match checkpoint
        self.model = ModelWithLSTM(input_channels, lstm_hidden)
        
        # Multiple heads for different classes
        self.heads = nn.ModuleDict()
        class_names = ['dog', 'human', 'car', 'drone']
        
        for class_name in class_names:
            self.heads[class_name] = OutputHead(lstm_hidden)
    
    def forward(self, x):
        # x shape: [batch, seq_len, 6] or [batch, 6, seq_len]
        
        # Ensure input is [batch, 6, seq_len]
        if x.dim() == 3 and x.size(1) != 6:
            x = x.transpose(1, 2)
        
        # CNN + LSTM feature extraction
        features = self.model(x)  # [batch, seq_len, 64]
        
        # Apply heads
        outputs = {}
        for class_name, head in self.heads.items():
            outputs[class_name] = head(features)
        
        return outputs


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load checkpoint into model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unknown checkpoint format")
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    # Create model
    model = TartanIMUModel()
    
    # Load checkpoint
    checkpoint_path = 'checkpoint_24.pt'
    try:
        model = load_checkpoint(model, checkpoint_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    
    # Set to eval mode
    model.eval()
    
    # Test forward pass
    batch_size = 1
    seq_len = 13  # Can be any length now due to repeat
    test_input = torch.randn(batch_size, 6, seq_len)
    
    with torch.no_grad():
        outputs = model(test_input)
        
    print("\nForward pass successful!")
    print("\nOutput shapes:")
    for class_name, (out1, out2, out_z) in outputs.items():
        print(f"{class_name}:")
        print(f"  output_block1: {out1.shape}")
        print(f"  output_block2: {out2.shape}")
        print(f"  output_block1_z: {out_z.shape}")
        
    # Print model structure
    print("\nModel structure:")
    print(model)