import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.transform import Rotation as R

try:
    from model import TartanIMUModel, load_checkpoint
except ImportError:
    print("[ERROR] 错误: 找不到 model.py")
    sys.exit(1)

# ==========================================
# 1. 简易 AHRS (计算姿态角)
# ==========================================
class SimpleAHRS:
    def __init__(self, dt):
        self.dt = dt
        self.yaw = 0.0 # 初始航向角
    
    def process(self, gyr_z):
        """
        简单的航向角积分: Yaw_t = Yaw_{t-1} + Gyr_z * dt
        gyr_z: [N] 弧度/秒
        """
        # 简单的去除零偏 (假设前100帧静止)
        if len(gyr_z) > 100:
            bias = np.mean(gyr_z[:100])
            if abs(bias) < 0.1: # 只有bias比较小时才去，防止其实是在动
                gyr_z = gyr_z - bias
        
        # 累积积分
        yaw_angles = np.cumsum(gyr_z * self.dt)
        return yaw_angles

# ==========================================
# 2. 数据加载 (User X=Up, User Z=Fwd)
# ==========================================
class DatasetWithRotation:
    def __init__(self, file_path):
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        acc_raw = df[['acc_x', 'acc_y', 'acc_z']].values
        gyr_raw = df[['gyr_x', 'gyr_y', 'gyr_z']].values
        
        # 自动转单位
        if np.max(np.abs(gyr_raw)) > 15.0:
            gyr_raw *= (np.pi / 180.0)

        # ---------------------------------------------------
        # 物理映射 (基于您的确认: X=Up, Z=Fwd)
        # ---------------------------------------------------
        self.acc_aligned = np.zeros_like(acc_raw)
        self.gyr_aligned = np.zeros_like(gyr_raw)

        # 1. Model X (前) <--- User Z
        self.acc_aligned[:, 0] = acc_raw[:, 2]  
        self.gyr_aligned[:, 0] = gyr_raw[:, 2]

        # 2. Model Y (左) <--- User Y
        self.acc_aligned[:, 1] = acc_raw[:, 1]  
        self.gyr_aligned[:, 1] = gyr_raw[:, 1]

        # 3. Model Z (上) <--- User X (垂直轴)
        self.acc_aligned[:, 2] = acc_raw[:, 0]  
        self.gyr_aligned[:, 2] = gyr_raw[:, 0]
        
        # 保存用于积分的原始 Yaw 角速度 (来自 User X / Model Z)
        # 注意：这里可能需要负号，取决于传感器是左手系还是右手系
        # 我们默认正向，如果反了，轨迹会镜像
        self.raw_yaw_rate = self.gyr_aligned[:, 2]

        # 去重力 (简化版，仅用于输入网络)
        self.acc_input = self.acc_aligned.copy()
        self.acc_input[:, 2] -= 9.81 
        
        self.features = np.concatenate([self.acc_input, self.gyr_aligned], axis=1)

    def get_data(self):
        return torch.FloatTensor(self.features).permute(1, 0).unsqueeze(0), self.raw_yaw_rate

# ==========================================
# 3. 主程序
# ==========================================
def main():
    CHECKPOINT = "checkpoint_28.pt" 
    IMU_FILE = "4d91_long_instep_imu.xlsx"      
    GT_FILE = "4d91_long_ground_truth.xlsx"    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 加载数据...")
    ds = DatasetWithRotation(IMU_FILE)
    inp, yaw_rate = ds.get_data()
    inp = inp.to(device)
    
    # 计算全局航向角 (Yaw)
    dt = 0.005 # 200Hz
    ahrs = SimpleAHRS(dt)
    yaw_angles = ahrs.process(yaw_rate) # [T]

    print("2. 模型推理 (预测机体速度)...")
    model = TartanIMUModel().to(device)
    load_checkpoint(model, CHECKPOINT, device)
    
    vel_list = []
    model.eval()
    total_len = inp.shape[2]
    
    for i in range(0, total_len, 2000):
        end = min(i+2000, total_len)
        if end-i < 20: break
        with torch.no_grad():
            res = model(inp[:,:,i:end])
            raw = res['human']
            if isinstance(raw, tuple):
                val = torch.cat([raw[0], raw[2]], dim=-1)
            else:
                val = raw
        vel_list.append(val.cpu())
    
    # 机体速度: [v_forward, v_left, v_up]
    body_vel = torch.cat(vel_list, dim=1).numpy()[0] # [T, 3]
    
    # 截断 Yaw 以匹配长度
    L = min(len(body_vel), len(yaw_angles))
    body_vel = body_vel[:L]
    yaw_angles = yaw_angles[:L]
    
    print("3. 坐标旋转 (Body -> Global) & 积分...")
    global_vel = np.zeros_like(body_vel)
    
    # 旋转公式:
    # V_global_x = V_body_x * cos(yaw) - V_body_y * sin(yaw)
    # V_global_y = V_body_x * sin(yaw) + V_body_y * cos(yaw)
    
    # 这里的 V_body_x 是 Forward (Model X), V_body_y 是 Left (Model Y)
    global_vel[:, 0] = body_vel[:, 0] * np.cos(yaw_angles) - body_vel[:, 1] * np.sin(yaw_angles)
    global_vel[:, 1] = body_vel[:, 0] * np.sin(yaw_angles) + body_vel[:, 1] * np.cos(yaw_angles)
    
    pred_pos = np.cumsum(global_vel[:, :2] * dt, axis=0) # 只取 XY
    pred_pos -= pred_pos[0]

    # 加载真值用于对比
    df_gt = pd.read_excel(GT_FILE, engine='openpyxl') if GT_FILE.endswith('xlsx') else pd.read_csv(GT_FILE)
    df_gt.columns = [c.strip() for c in df_gt.columns]
    if 'gt_pos_global_x' in df_gt.columns:
        gt_pos = df_gt[['gt_pos_global_x', 'gt_pos_global_y']].values * 1000.0
    else:
        gt_pos = df_gt.iloc[:, 4:6].values * 1000.0
    gt_pos -= gt_pos[0]

    # 画图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Trajectory Comparison (Raw)")
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'r--', label='Ground Truth')
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'b-', label='Predicted (with Rotation)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # SVD 自动对齐一下，方便看形状是否一致
    from scipy.interpolate import interp1d
    x_old = np.linspace(0, 1, len(pred_pos))
    x_new = np.linspace(0, 1, len(gt_pos))
    f = interp1d(x_old, pred_pos, axis=0)
    pred_resampled = f(x_new)
    
    mu_p = np.mean(pred_resampled, 0)
    mu_g = np.mean(gt_pos, 0)
    H = np.dot((pred_resampled - mu_p).T, (gt_pos - mu_g))
    U, S, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)
    if np.linalg.det(R_mat) < 0:
        Vt[1,:] *= -1
        R_mat = np.dot(Vt.T, U.T)
    pred_aligned = np.dot(pred_pos - mu_p, R_mat.T) + mu_g
    
    plt.subplot(1, 2, 2)
    plt.title("Trajectory Comparison (Aligned)")
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'r--', label='Ground Truth')
    plt.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'g-', label='Pred Aligned')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ins_results.png')
    print("[OK] 结果已保存: ins_results.png")
    plt.show()

if __name__ == "__main__":
    main()

