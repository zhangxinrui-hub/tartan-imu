import torch
import torch.nn as nn
import re


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), # convs.0
            nn.BatchNorm1d(out_channels),                                   # convs.1
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),# convs.3
            nn.BatchNorm1d(out_channels)                                    # convs.4
        ])
    
    def forward(self, x):
        out = self.convs[0](x)
        out = self.convs[1](out)
        out = self.convs[2](out)
        out = self.convs[3](out)
        return out

class Downsample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fcs = nn.ModuleList([
            nn.Linear(64, 256),     # fcs.0
            nn.ReLU(),              # fcs.1 (not in weights, so just activation)
            nn.Linear(256, 256),    # fcs.3
            nn.ReLU(),              # fcs.4 (not in weights, so just activation)
            nn.Linear(256, out_dim) # fcs.6
        ])
    def forward(self, x):
        out = self.fcs[0](x)
        out = self.fcs[1](out)
        out = self.fcs[2](out)
        out = self.fcs[3](out)
        out = self.fcs[4](out)
        return out

class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input Block
        self.input_block = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding=3),  # input_block.0.weight
            nn.BatchNorm1d(64)                           # input_block.1.weight
        )
        # Residual Groups
        self.residual_groups = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock1D(64, 64), ResidualBlock1D(64, 64)
            ]),
            nn.ModuleList([
                ResidualBlock1D(64, 128), ResidualBlock1D(128, 128)
            ]),
            nn.ModuleList([
                ResidualBlock1D(128, 256), ResidualBlock1D(256, 256)
            ])
        ])
        # Downsample layers
        self.downsample_layers = nn.ModuleList([
            Downsample1D(64, 128),
            Downsample1D(128, 256)
        ])
        # ResNet post-processing block
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128)
        )
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1664,  # 256 * 6.5 (你要根据上一层输出实际序列长度确定)
            hidden_size=256,
            batch_first=True
        )
        # Output heads
        def make_head():
            return nn.ModuleDict({
                "output_block1": OutputBlock(2),   # 类似 heads.car.output_block1.*, 输出2
                "output_block2": OutputBlock(3),   # 类似 heads.car.output_block2.*, 输出3
                "output_block1_z": OutputBlock(1), # 类似 heads.car.output_block1_z.*, 输出1
            })

        self.heads = nn.ModuleDict({
            "car": make_head(),
            "dog": make_head(),
            "human": make_head(),
            "drone": make_head()
        })

    def forward(self, x):
        out = self.input_block(x)
        # 残差块和下采样，需按参数名实际连接
        for i, group in enumerate(self.residual_groups):
            out = group[0](out)
            out = group[1](out)
            if i < 2:  # 对应1.0和2.0的downsample
                out = self.downsample_layers[i](out)
        out = self.resnet_post_pro(out)
        # 展平以适配LSTM
        out = out.transpose(1, 2)  # (B, seq_len, feat)
        out, _ = self.lstm(out)
        # 选择最后一个时间步，送去output heads
        out = out[:, -1, :]
        results = {
            head_name: {
                block: self.heads[head_name][block](out)
                for block in self.heads[head_name]
            }
            for head_name in self.heads
        }
        return results

# 加载checkpoint
model = MainModel()
ckpt = torch.load("checkpoint_24.pt", map_location="cpu")
state_dict = ckpt['model_state_dict']
new_state_dict = {re.sub(r"^model\.", "", k): v for k, v in state_dict.items()}
result = model.load_state_dict(new_state_dict, strict=False)
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)