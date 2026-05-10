import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 检查 model.py
try:
    from model import TartanIMUModel, load_checkpoint
except ImportError:
    print("[ERROR] 错误: 找不到 model.py")
    sys.exit(1)

# ==========================================
# 0. 重力追踪器 (核心修复: 动态去除重力)
# ==========================================
class GravityTracker:
    def __init__(self, dt, alpha=0.98):
        self.dt = dt
        self.alpha = alpha
        self.g_vector = np.array([0.0, 0.0, 1.0]) # 初始假设重力在 Z 轴 (归一化)

    def process(self, acc, gyr):
        """
        使用互补滤波追踪重力方向
        acc: [N, 3] m/s^2
        gyr: [N, 3] rad/s
        返回: 去重力后的加速度 [N, 3]
        """
        n_samples = acc.shape[0]
        acc_no_g = np.zeros_like(acc)
        
        # 初始化重力向量为第一帧加速度方向 (假设起步静止)
        if np.linalg.norm(acc[0]) > 0:
            self.g_vector = acc[0] / np.linalg.norm(acc[0])
            
        for i in range(1, n_samples):
            # 1. 陀螺仪积分预测重力方向变化 (g_new = g_old + (w x g_old) * dt)
            # 注意: 坐标系旋转导致向量反向旋转
            # delta_theta = gyr[i-1] * self.dt
            # 这里的 gyr 是机体角速度。重力向量在机体坐标系下的导数是: g_dot = -omega x g
            g_pred = self.g_vector + np.cross(self.g_vector, gyr[i-1]) * self.dt  # 近似积分
            g_pred = g_pred / np.linalg.norm(g_pred)
            
            # 2. 加速度计修正 (假设长期平均加速度就是重力)
            acc_norm = np.linalg.norm(acc[i])
            if 8.0 < acc_norm < 12.0: # 只在非剧烈运动时修正
                acc_dir = acc[i] / acc_norm
                # 互补滤波融合
                self.g_vector = self.alpha * g_pred + (1 - self.alpha) * acc_dir
            else:
                self.g_vector = g_pred
                
            self.g_vector = self.g_vector / np.linalg.norm(self.g_vector)
            
            # 3. 去除重力 (Acc_real = Acc_meas - g_vector * 9.81)
            # 注意: 加速度计测量的是 (a - g)，所以 a = Acc_meas + g? 
            # 不，静止时 Acc_meas = +9.8 (向上支持力). 所以 Acc_real = Acc_meas - (+9.8 * g_dir)
            acc_no_g[i] = acc[i] - self.g_vector * 9.81
            
        return acc_no_g

# ==========================================
# 1. 数据加载与处理
# ==========================================
class TartanValidationDataset:
    def __init__(self, file_path, is_gyro_deg=False):
        # 智能读取
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        # 提取原始数据
        acc_raw = df[['acc_x', 'acc_y', 'acc_z']].values
        gyr_raw = df[['gyr_x', 'gyr_y', 'gyr_z']].values
        
        # 时间戳
        if 'time after start [s]' in df.columns:
            self.ts = df['time after start [s]'].values
        elif 'time_s' in df.columns:
            self.ts = df['time_s'].values
        else:
            self.ts = np.arange(len(acc_raw)) * 0.005 # 200Hz

        dt = np.mean(np.diff(self.ts)) if len(self.ts) > 1 else 0.005
        print(f"数据采样率 dt: {dt:.4f}s")

        # [关键判断] 角速度单位
        # 如果数据里有 > 10 的值，基本不可能是 rad/s (那太快了)，肯定是 deg/s
        # 如果最大值只有 3-5，那极大概率已经是 rad/s 了
        max_gyr = np.max(np.abs(gyr_raw))
        if is_gyro_deg:
            print("强制模式: Gyro deg/s -> rad/s")
            gyr_rad = gyr_raw * (np.pi / 180.0)
        elif max_gyr > 15.0: 
            print(f"检测到 Gyro 最大值 {max_gyr:.1f} > 15, 判定为 deg/s -> 转 rad/s")
            gyr_rad = gyr_raw * (np.pi / 180.0)
        else:
            print(f"检测到 Gyro 最大值 {max_gyr:.1f} < 15, 判定为 rad/s (保持不变)")
            gyr_rad = gyr_raw

        # [坐标系对齐] User -> Model
        # Model: X前, Y左, Z上
        # User Waist: acc_x ≈ 9.8 (X上)
        # 假设: User Z -> Model X (前), User Y -> Model Y (左)
        
        acc_aligned = np.zeros_like(acc_raw)
        gyr_aligned = np.zeros_like(gyr_rad)
        
        # 映射尝试 (如果轨迹方向反了，改这里的符号)
        acc_aligned[:, 0] = acc_raw[:, 1]  # Mod X = User Z
        acc_aligned[:, 1] = acc_raw[:, 2]  # Mod Y = User Y
        acc_aligned[:, 2] = acc_raw[:, 0]  # Mod Z = User X
        
        gyr_aligned[:, 0] = gyr_rad[:, 1]
        gyr_aligned[:, 1] = gyr_rad[:, 2]
        gyr_aligned[:, 2] = gyr_rad[:, 0]
        
        # [动态去除重力]
        print("正在进行动态重力去除 (Complementary Filter)...")
        tracker = GravityTracker(dt=dt)
        acc_final = tracker.process(acc_aligned, gyr_aligned)
        
        self.features = np.concatenate([acc_final, gyr_aligned], axis=1)

    def get_sequence_data(self):
        x = torch.FloatTensor(self.features).permute(1, 0).unsqueeze(0) 
        return x, self.ts

# ==========================================
# 2. 真值加载 (修复缩放问题)
# ==========================================
def load_ground_truth(file_path, scale=1000.0):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 时间
    if 'time_s' in df.columns:
        ts = df['time_s'].values
    else:
        ts = df['time after start [s]'].values
        
    # 位置提取 & 缩放
    if 'gt_pos_global_x' in df.columns:
        pos = df[['gt_pos_global_x', 'gt_pos_global_y']].values
    else:
        # 备用列名搜索
        pos = df.iloc[:, 4:6].values # 盲猜第4,5列
        
    pos = pos * scale # * 1000
    print(f"真值已加载并缩放 x{scale}. 范围: X[{np.min(pos[:,0]):.2f}, {np.max(pos[:,0]):.2f}]")
    return ts, pos

# ==========================================
# 3. 推理函数 (修复 Tuple + 拼接逻辑)
# ==========================================
def run_inference(model, inputs, chunk_size=2000):
    batch, channels, total_len = inputs.shape
    outputs_list = []
    model.eval()
    
    print(f"开始推理 (长度 {total_len})...")
    start = 0
    while start < total_len:
        end = min(start + chunk_size, total_len)
        if end - start < 20: break # 丢弃过短尾部
        
        chunk = inputs[:, :, start:end]
        with torch.no_grad():
            res = model(chunk)
            # TartanIMU human head 输出: (vel_2d, vel_3d, vel_z)
            # 我们需要 vel_3d (index 1), 或者是拼接 vel_xy (0) 和 vel_z (2)
            # Model.py 里 output_block2 输出维度是 3，通常对应 3D 协方差
            # output_block1 是 2 (XY vel), output_block1_z 是 1 (Z vel)
            # 所以正确的速度是: cat(res['human'][0], res['human'][2])
            
            raw = res['human']
            if isinstance(raw, tuple):
                vel_xy = raw[0] # [B, T, 2]
                vel_z  = raw[2] # [B, T, 1]
                vel_3d = torch.cat([vel_xy, vel_z], dim=-1) # [B, T, 3]
            else:
                vel_3d = raw # fallback
                
        outputs_list.append(vel_3d.cpu())
        start = end
        sys.stdout.write(f"\r处理中: {end}/{total_len}")
        
    print("\n拼接结果...")
    # Linear 层输出通常是 [Batch, Time, Dim]，所以在 Dim=1 拼接
    return torch.cat(outputs_list, dim=1)

# ==========================================
# 4. 主流程
# ==========================================
def main():
    CHECKPOINT = "checkpoint_28.pt" 
    IMU_FILE = "4d91_long_waist_imu.xlsx"      
    GT_FILE = "4d91_long_ground_truth.xlsx"    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型
    model = TartanIMUModel().to(device)
    load_checkpoint(model, CHECKPOINT, device)
    
    # 2. 加载数据 (自动判断 Gyro 单位, 自动去重力)
    # is_gyro_deg=False 让程序自动判断。如果结果还是直线，尝试改为 True
    dataset = TartanValidationDataset(IMU_FILE, is_gyro_deg=False) 
    inp, _ = dataset.get_sequence_data()
    
    # 3. 加载真值 (x1000)
    gt_ts, gt_pos = load_ground_truth(GT_FILE, scale=1000.0)
    
    # 4. 推理
    pred_vel = run_inference(model, inp.to(device)).numpy()[0] # [T, 3]
    
    # 5. 积分
    dt = 0.005 # 200Hz
    pred_pos = np.cumsum(pred_vel * dt, axis=0)
    
    # 对齐
    pred_pos -= pred_pos[0]
    gt_pos -= gt_pos[0]
    
    # 6. 画图
    plt.figure(figsize=(10, 8))
    plt.plot(gt_pos[:,0], gt_pos[:,1], 'r--', label='GT (Scaled)', linewidth=2)
    
    # 截断绘图以匹配长度
    L = min(len(pred_pos), int(len(gt_pos) * (gt_ts[-1]/dataset.ts[-1]) * 1.1))
    # 或者直接画全部
    plt.plot(pred_pos[:,0], pred_pos[:,1], 'b-', label='Pred (IMU)', alpha=0.8)
    
    plt.title(f"Validation Result\nScale=1000, Gravity Removed")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('result_final.png')
    plt.show()

if __name__ == "__main__":
    main()