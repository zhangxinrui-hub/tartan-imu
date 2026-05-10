import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUOdometryNet(nn.Module):
    """
    基于 CNN-LSTM 的 6-DOF 惯性里程计网络 (Sensors 2019, Lima et al.)
    论文: End-to-End Learning Framework for IMU-Based 6-DOF Odometry [cite: 4]
    """
    def __init__(self, sequence_length=200):
        super(IMUOdometryNet, self).__init__()
        
        # --- 1. 特征提取器 (CNN Encoders) [cite: 86] ---
        # 论文描述: 2层 1D卷积 (128 filters, kernel 11) + MaxPool
        # 这里的实现为了保持维度对齐，使用了 padding=5
        
        self.encoder_gyro = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1) 
        )
        
        self.encoder_acc = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        self.dropout = nn.Dropout(0.25) # [cite: 90]
        
        # --- 2. 时序建模 (Bidirectional LSTM) [cite: 88, 89] ---
        # 输入维度: 128 (gyro) + 128 (acc) = 256
        # 论文使用 2层堆叠的双向 LSTM
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.25 # 层间 Dropout
        )
        
        # LSTM 输出维度: hidden_size * 2 (bidirectional) = 256
        self.lstm_out_dim = 256
        
        # --- 3. 全连接输出层 (Fully Connected) [cite: 91] ---
        # 输出位姿变化 (Relative Pose): 平移 (3) + 四元数 (4)
        self.fc_pos = nn.Linear(self.lstm_out_dim, 3) 
        self.fc_quat = nn.Linear(self.lstm_out_dim, 4) 

    def forward(self, x_gyro, x_acc):
        """
        Args:
            x_gyro: [Batch, Sequence_Length, 3]
            x_acc:  [Batch, Sequence_Length, 3]
        Returns:
            pred_p: [Batch, 3] 平移向量
            pred_q: [Batch, 4] 单位四元数
        """
        # 转换维度适应 Conv1d: [B, L, C] -> [B, C, L]
        x_gyro = x_gyro.transpose(1, 2)
        x_acc = x_acc.transpose(1, 2)
        
        # CNN 编码
        feat_gyro = self.encoder_gyro(x_gyro)
        feat_acc = self.encoder_acc(x_acc)
        
        # Dropout
        feat_gyro = self.dropout(feat_gyro)
        feat_acc = self.dropout(feat_acc)
        
        # 特征融合: [B, 128, L] + [B, 128, L] -> [B, 256, L]
        feat = torch.cat((feat_gyro, feat_acc), dim=1)
        
        # 转换回 LSTM 维度: [B, 256, L] -> [B, L, 256]
        feat = feat.transpose(1, 2)
        
        # LSTM 前向传播
        # out: [Batch, Seq_Len, Hidden*2]
        out, _ = self.lstm(feat)
        
        # 取最后一个时间步的输出用于回归
        last_out = out[:, -1, :]
        
        # 回归位姿
        pred_p = self.fc_pos(last_out)
        raw_quat = self.fc_quat(last_out)
        
        # --- 关键步骤: 四元数归一化  ---
        # 强制输出为单位四元数，这对几何 Loss 计算至关重要
        pred_q = F.normalize(raw_quat, p=2, dim=1)
        
        return pred_p, pred_q


class Lima2019MultiTaskLoss(nn.Module):
    """
    实现了论文中效果最好的 Loss 组合: TMAE + QME + Multi-Task Weighting
    
    1. TMAE (Translation Mean Absolute Error): 平移 L1 Loss [cite: 155]
    2. QME (Quaternion Multiplicative Error): 四元数乘法误差 [cite: 155]
    3. MTL (Multi-Task Learning): 自动学习权重 sigma 
    """
    def __init__(self, init_log_vars=[0.0, -3.0]):
        super(Lima2019MultiTaskLoss, self).__init__()
        # log_vars = log(sigma^2)
        # 初始化建议: 给旋转项一个较小的 log_var (较大的权重)，因为旋转数值小但影响大
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    def quat_multiply(self, q1, q2):
        """
        计算四元数汉密尔顿积 (Hamilton Product)
        假设四元数格式为 [w, x, y, z] (实部在前)
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((w, x, y, z), dim=1)

    def quat_conjugate(self, q):
        """
        计算单位四元数的共轭 (即逆)
        q* = [w, -x, -y, -z]
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack((w, -x, -y, -z), dim=1)

    def forward(self, pred_p, pred_q, target_p, target_q):
        """
        计算加权后的总 Loss
        Args:
            pred_p, target_p: [Batch, 3]
            pred_q, target_q: [Batch, 4] (需已归一化)
        """
        # --- 1. 计算平移误差 (TMAE) [cite: 155] ---
        # 使用 L1 Loss (MAE)
        loss_tmae = torch.mean(torch.norm(pred_p - target_p, p=1, dim=1))

        # --- 2. 计算旋转误差 (QME) [cite: 155] ---
        # QME = 2 * || imag(q_pred * q_target_conj) ||_1
        
        # 计算 target 的共轭
        target_q_inv = self.quat_conjugate(target_q)
        
        # 计算差值旋转 (Error Quaternion)
        q_error = self.quat_multiply(pred_q, target_q_inv)
        
        # 取虚部 (x, y, z) 的 L1 范数
        # q_error[:, 1:] 取的是 [x, y, z] 部分
        loss_qme = 2.0 * torch.mean(torch.norm(q_error[:, 1:], p=1, dim=1))

        # --- 3. 多任务学习自动加权  ---
        # Total Loss = exp(-s_p) * L_p + s_p + exp(-s_q) * L_q + s_q
        
        s_p = self.log_vars[0]
        s_q = self.log_vars[1]
        
        precision_p = torch.exp(-s_p)
        loss_p_weighted = precision_p * loss_tmae + s_p
        
        precision_q = torch.exp(-s_q)
        loss_q_weighted = precision_q * loss_qme + s_q
        
        total_loss = loss_p_weighted + loss_q_weighted
        
        return total_loss, loss_tmae.item(), loss_qme.item()

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟数据
    batch_size = 32
    seq_len = 200
    gyro = torch.randn(batch_size, seq_len, 3)
    acc = torch.randn(batch_size, seq_len, 3)
    
    # 模拟 Ground Truth
    gt_pos = torch.randn(batch_size, 3)
    gt_quat = F.normalize(torch.randn(batch_size, 4), p=2, dim=1)

    # 1. 实例化
    model = IMUOdometryNet()
    criterion = Lima2019MultiTaskLoss()
    
    # 2. 优化器 (注意: 必须包含 criterion.parameters() 以学习权重)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=1e-4
    )

    # 3. 前向传播
    pred_p, pred_q = model(gyro, acc)
    
    # 4. 计算 Loss
    loss, l_p_raw, l_q_raw = criterion(pred_p, pred_q, gt_pos, gt_quat)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Raw Trans MAE: {l_p_raw:.4f} m")
    print(f"Raw Rot QME: {l_q_raw:.4f}")
    
    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()