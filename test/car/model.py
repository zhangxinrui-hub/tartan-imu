import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===========================================================
# 0. LoRA Layer Definition
# ===========================================================
class LoRALayer(nn.Module):
    def __init__(self, r=32, alpha=8):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

class LoRAConv1d(LoRALayer):
    """Low-rank adapter for Conv1d, matching checkpoint shape (Out, In*K)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, r=32, alpha=8):
        super().__init__(r, alpha)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.lora_dim = r * kernel_size
        self.input_dim = in_channels * kernel_size

        self.lora_A = nn.Parameter(torch.randn(self.lora_dim, self.input_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, self.lora_dim))

        self.conv.weight.requires_grad = False
        if bias:
            self.conv.bias.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        delta_w_flat = self.lora_B @ self.lora_A  # [Out, In*K]
        delta_w = delta_w_flat.view(self.conv.weight.shape) * self.scaling
        return F.conv1d(x,
                        self.conv.weight + delta_w,
                        self.conv.bias,
                        stride=self.conv.stride,
                        padding=self.conv.padding)

# ===========================================================
# 1. Residual block
# ===========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, has_downsample=False, use_lora=False):
        super().__init__()
        
        def ConvLayer(in_c, out_c, k, s, p, b=False):
            if use_lora:
                return LoRAConv1d(in_c, out_c, k, stride=s, padding=p, bias=b)
            return nn.Conv1d(in_c, out_c, k, stride=s, padding=p, bias=b)

        self.convs = nn.Sequential(
            ConvLayer(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            ConvLayer(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
        )
        
        self.downsample = None
        if has_downsample:
            self.downsample = nn.Sequential(
                ConvLayer(in_channels, out_channels, 1, stride, 0),
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
# 2. Transformer Block
# ===========================================================
class Mlp(nn.Module):
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
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, bias=True, add_bias_kv=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=dim_ff, out_features=d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm_1(x + attn_out)
        ffn_out = self.mlp(x)
        x = self.norm_2(x + ffn_out)
        return x

class IMU_Trunk(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([IMUTransformerBlock(d_model=64, nhead=4, dim_ff=256) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# ===========================================================
# 3. Model Core
# ===========================================================
class ModelWithLSTM(nn.Module):
    def __init__(self, input_channels=6, lstm_hidden=64, fine_tune_mode=None):
        super().__init__()
        use_lora = (fine_tune_mode == 'lora')
        
        def ConvLayer(in_c, out_c, k, s, p, b=False):
            if use_lora:
                return LoRAConv1d(in_c, out_c, k, stride=s, padding=p, bias=b)
            return nn.Conv1d(in_c, out_c, k, stride=s, padding=p, bias=b)

        self.input_block = nn.Sequential(
            ConvLayer(input_channels, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.residual_groups = nn.ModuleList([
            nn.ModuleList([ResidualBlock(64, 64, use_lora=use_lora), ResidualBlock(64, 64, use_lora=use_lora)]),
            nn.ModuleList([ResidualBlock(64, 128, stride=2, has_downsample=True, use_lora=use_lora), ResidualBlock(128, 128, use_lora=use_lora)]),
            nn.ModuleList([ResidualBlock(128, 256, stride=2, has_downsample=True, use_lora=use_lora), ResidualBlock(256, 256, use_lora=use_lora)])
        ])
        
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        
        self.lstm = nn.LSTM(input_size=1664, hidden_size=lstm_hidden, batch_first=True)
        self.IMU_Trunk = IMU_Trunk()

    def forward(self, x, hidden=None):
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
        lstm_out, hidden_out = self.lstm(x_unf, hidden)
        features = self.IMU_Trunk(lstm_out)
        return features, hidden_out

# ===========================================================
# 4. Output Head
# ===========================================================
class OutputHead(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
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

# ===========================================================
# 5. Main Model
# ===========================================================
class TartanIMUModel(nn.Module):
    def __init__(self, fine_tune_mode=None):
        super().__init__()
        self.model = ModelWithLSTM(fine_tune_mode=fine_tune_mode)
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
# 6. Checkpoint Loaders
# ===========================================================
def load_dual_checkpoints(model, base_ckpt_path, lora_ckpt_path, device='cuda'):
    """Load base (Stage 1) weights then overlay LoRA (Stage 2) parameters."""
    print(f"Loading base model: {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    base_state = base_ckpt.get('model_state_dict', base_ckpt)
    base_state = {k.replace('module.', ''): v for k, v in base_state.items()}

    # Remap Conv1d weights to LoRAConv1d's inner conv
    mapped_base_state = {}
    for k, v in base_state.items():
        if ('input_block' in k or 'residual_groups' in k) and 'weight' in k and 'bn' not in k:
            mapped_base_state[k.replace('.weight', '.conv.weight')] = v
        else:
            mapped_base_state[k] = v

    missing, _ = model.load_state_dict(mapped_base_state, strict=False)
    print(f"  Base loaded. Missing keys (expected LoRA params): {len(missing)}")

    print(f"Loading LoRA params: {lora_ckpt_path}")
    lora_ckpt = torch.load(lora_ckpt_path, map_location=device)
    lora_state = lora_ckpt.get('model_state_dict', lora_ckpt)
    lora_state = {k.replace('module.', ''): v for k, v in lora_state.items()}

    m2, u2 = model.load_state_dict(lora_state, strict=False)

    if len(m2) > 0:
        truly_missing = [k for k in m2 if k not in mapped_base_state and 'lora' not in k]
        if truly_missing:
            print(f"  WARNING: keys missing from both Base and LoRA: {truly_missing[:5]}")

    print("Model loaded successfully (Base + LoRA).")
    return model

def load_checkpoint(model, path, device='cpu'):
    """Load a Stage 1 checkpoint, handling 'module.' prefix from DataParallel."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    clean = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    try:
        model.load_state_dict(clean, strict=True)
        print(f"Checkpoint loaded (strict): {path}")
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        missing, unexpected = model.load_state_dict(clean, strict=False)
        real_missing = [k for k in missing if 'lora' not in k]
        if real_missing:
            print(f"ERROR: Missing non-LoRA keys: {real_missing[:5]}")
        else:
            print(f"Checkpoint loaded (strict=False, missing only LoRA keys): {path}")
    return model


if __name__ == "__main__":
    # Self-test
    print("Self-Testing Model Initialization...")
    model = TartanIMUModel(fine_tune_mode='lora')