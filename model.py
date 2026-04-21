import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================
# 1. Residual block (ResNet) - 保持不变
# ===========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, has_downsample=False):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = None
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.convs(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)

# ===========================================================
# 2. Transformer Block (IMU_Trunk) - [已修正命名匹配]
# ===========================================================
class Mlp(nn.Module):
    """
    专门定义的 MLP 类，为了匹配 Checkpoint 中的 'mlp.fc1' 和 'mlp.fc2' 命名
    """
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class IMUTransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_ff=256):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            bias=True,
            add_bias_kv=True   # ⭐ 关键：匹配 checkpoint
        )

        self.norm_1 = nn.LayerNorm(d_model)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_ff,
            out_features=d_model
        )

        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, 64]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm_1(x + attn_out)

        ffn_out = self.mlp(x)
        x = self.norm_2(x + ffn_out)

        return x

class IMU_Trunk(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()
        # Checkpoint 通常使用 module list
        self.blocks = nn.ModuleList([IMUTransformerBlock(d_model=64, nhead=4, dim_ff=256) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# ===========================================================
# 3. CNN + ResNet + LSTM - 保持不变
# ===========================================================
class ModelWithLSTM(nn.Module):
    def __init__(self, input_channels=6, lstm_hidden=64):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.residual_groups = nn.ModuleList([
            nn.ModuleList([ResidualBlock(64, 64), ResidualBlock(64, 64)]),
            nn.ModuleList([ResidualBlock(64, 128, stride=2, has_downsample=True), ResidualBlock(128, 128)]),
            nn.ModuleList([ResidualBlock(128, 256, stride=2, has_downsample=True), ResidualBlock(256, 256)])
        ])
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # 你的 Unfold 逻辑非常完美地解释了这里为什么是 1664 (128 * 13)
        self.lstm = nn.LSTM(input_size=1664, hidden_size=lstm_hidden, batch_first=True)
        self.IMU_Trunk = IMU_Trunk() # Transformer 接在 LSTM 后面

    def forward(self, x, hidden=None):
        """
        hidden: (h_n, c_n) 来自上一个 chunk 的 LSTM 隐状态，可选。
                传入后 LSTM 将从上一块末尾继续，而不是从零初始化。
        返回: (features, hidden_out) 其中 hidden_out 可传给下一块。
        """
        x = self.input_block(x)
        for group in self.residual_groups:
            for blk in group:
                x = blk(x)
        x = self.resnet_post_pro(x)
        
        B, C, T = x.shape
        win = 13
        if T < win:
            x = F.pad(x, (0, win - T))
            T = win
            
        x_unf = x.unfold(2, size=win, step=1)
        B, C, T_new, W = x_unf.shape
        x_unf = x_unf.permute(0, 2, 1, 3).reshape(B, T_new, C * W)
        
        lstm_out, hidden_out = self.lstm(x_unf, hidden)  # 传入/传出隐状态
        features = self.IMU_Trunk(lstm_out)
        return features, hidden_out

# ===========================================================
# 4. Output Heads - 保持不变
# ===========================================================
class OutputHead(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        # 这种写法完美匹配 Checkpoint 中的 fcs.0, fcs.3, fcs.6
        self.output_block1 = nn.ModuleDict({'fcs': nn.ModuleList([
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True), nn.Identity(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Identity(),
            nn.Linear(256, 2)
        ])})
        self.output_block2 = nn.ModuleDict({'fcs': nn.ModuleList([
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True), nn.Identity(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Identity(),
            nn.Linear(256, 3)
        ])})
        self.output_block1_z = nn.ModuleDict({'fcs': nn.ModuleList([
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True), nn.Identity(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Identity(),
            nn.Linear(256, 1)
        ])})

    def forward(self, x):
        def forward_block(block, x):
            x = block['fcs'][0](x)
            x = block['fcs'][1](x)
            x = block['fcs'][3](x)
            x = block['fcs'][4](x)
            x = block['fcs'][6](x)
            return x
        return forward_block(self.output_block1, x), forward_block(self.output_block2, x), forward_block(self.output_block1_z, x)

class TartanIMUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ModelWithLSTM()
        self.heads = nn.ModuleDict()
        for name in ['dog', 'human', 'car', 'drone']:
            self.heads[name] = OutputHead()

    def forward(self, x, hidden=None):
        if x.dim() == 3 and x.size(1) != 6:
            x = x.transpose(1, 2)
        features, hidden_out = self.model(x, hidden)
        outputs = {}
        for k in self.heads:
            outputs[k] = self.heads[k](features)
        return outputs, hidden_out

# ===========================================================
# 6. Checkpoint Loader (增强版)
# ===========================================================
def load_checkpoint(model, path, device='cpu'):
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # 1. 提取 state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    # 2. 处理 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # 3. 加载
    # strict=False 允许少量不匹配 (如 running_mean 的维度微小差异等)，但在验证阶段建议先尝试 strict=True
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Checkpoint Loaded Successfully (Strict Mode)!")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Retrying with strict=False...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0: print(f"Sample missing: {missing[:3]}")

    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TartanIMUModel().to(device)
    # 请确保文件名为用户上传的文件名
    model = load_checkpoint(model, "checkpoint_28.pt", device=device) 
    
    model.eval()
    # 模拟输入: [Batch, Channels, Time]
    dummy_input = torch.randn(1, 6, 200).to(device)
    
    with torch.no_grad():
        out = model(dummy_input)
        
    print("\nForward Pass Results:")
    for k, v in out.items():
        print(f"Robot: {k:5} | Vel 2D: {v[0].shape}")