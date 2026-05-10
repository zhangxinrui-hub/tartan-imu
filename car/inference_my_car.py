import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from model import TartanIMUModel, load_checkpoint

# ================================================================
# 配置参数
# ================================================================
SCALE_FACTOR = 11.53  # 【关键参数】来自 Speed Analysis
FILTER_CUTOFF = 15.0  # 保留更多动态

# ================================================================
# 1. 滤波器 & GT 处理
# ================================================================
def low_pass_filter_6d(data, cutoff=15.0, fs=200.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = np.zeros_like(data)
    for i in range(data.shape[1]):
        y[:, i] = filtfilt(b, a, data[:, i])
    return y

def process_gt_recalc_yaw(gt_csv, target_timestamps):
    df = pd.read_csv(gt_csv)
    ts_gt = df['timestamp'].values
    pos_gt = df[['x_gt', 'y_gt', 'z_gt']].values
    
    # 重算 Yaw
    diffs = np.diff(pos_gt, axis=0)
    yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
    yaw = np.append(yaw, yaw[-1])
    new_r = R.from_euler('z', yaw, degrees=False)
    
    # 对齐插值
    f_pos = interp1d(ts_gt, pos_gt, axis=0, kind='linear', fill_value="extrapolate")
    pos_aligned = f_pos(target_timestamps)
    
    from scipy.spatial.transform import Slerp
    slerp = Slerp(ts_gt, new_r)
    valid_mask = (target_timestamps >= ts_gt[0]) & (target_timestamps <= ts_gt[-1])
    target_ts_valid = target_timestamps[valid_mask]
    r_interp = slerp(target_ts_valid)
    quat_aligned = r_interp.as_quat()
    
    return valid_mask, pos_aligned[valid_mask], quat_aligned

# ================================================================
# 2. 主推理流程
# ================================================================
def run_final_inference_scaled(ckpt_path, imu_csv, gt_csv, device='cuda', chunk_size=1000):
    print(f"Loading IMU: {imu_csv}")
    df_imu = pd.read_csv(imu_csv).iloc[::2].reset_index(drop=True)
    imu_data_raw = df_imu[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
    timestamps = df_imu['timestamp'].values
    
    print(f"应用 {FILTER_CUTOFF}Hz 滤波器...")
    imu_data_clean = low_pass_filter_6d(imu_data_raw, cutoff=FILTER_CUTOFF, fs=200.0)
    
    print("对齐 GT...")
    valid_mask, gt_pos, gt_quat = process_gt_recalc_yaw(gt_csv, timestamps)
    imu_data_clean = imu_data_clean[valid_mask]
    timestamps = timestamps[valid_mask]
    total_len = len(imu_data_clean)
    
    # 去 Bias
    static_bias = np.mean(imu_data_clean[:100], axis=0)
    processed_imu = imu_data_clean - static_bias
    
    # 模型推理
    model = TartanIMUModel().to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()
    
    all_vel = []
    print("开始推理...")
    with torch.no_grad():
        for i in range(0, total_len, chunk_size):
            end_idx = min(i + chunk_size, total_len)
            chunk_data = processed_imu[i:end_idx]
            imu_tensor = torch.FloatTensor(chunk_data).unsqueeze(0).permute(0, 2, 1).to(device)
            out = model(imu_tensor)['car']
            
            # 提取速度
            v_xy = out[0].cpu().numpy().squeeze()
            v_z  = out[2].cpu().numpy().squeeze()
            
            if v_xy.ndim == 1: v_xy = v_xy[np.newaxis, :]
            if v_z.ndim == 0: v_z = np.array([v_z])
            if v_z.ndim == 1: v_z = v_z[:, np.newaxis]
            
            # 【关键步骤】应用缩放因子 Scale Factor
            v_xy = v_xy * SCALE_FACTOR
            v_z  = v_z  * SCALE_FACTOR 
            
            all_vel.append(np.hstack([v_xy, v_z]))
            
    vel_body_raw = np.concatenate(all_vel, axis=0)
    
    # 插值恢复长度
    f_interp = interp1d(np.linspace(0, 1, len(vel_body_raw)), vel_body_raw, axis=0, kind='linear', fill_value="extrapolate")
    vel_body = f_interp(np.linspace(0, 1, total_len))

    # 约束
    vel_body[:, 1] = 0.0 # Vy = 0
    vel_body[:, 2] = 0.0 # Vz = 0
    
    # 旋转与积分
    r = R.from_quat(gt_quat)
    vel_world = r.apply(vel_body)
    
    dt = 1.0 / 200.0
    pred_pos = np.zeros((total_len, 3))
    gt_pos = gt_pos - gt_pos[0] # 归零起点
    curr_pos = np.zeros(3)
    
    for k in range(total_len):
        curr_pos += vel_world[k] * dt
        pred_pos[k] = curr_pos

    # 评估
    errors = np.linalg.norm(gt_pos - pred_pos, axis=1)
    ate = np.sqrt(np.mean(errors**2))
    path_len = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    drift = (ate / path_len) * 100 if path_len > 0 else 0
    
    print("="*40)
    print(f"FINAL RESULT (Scaled x{SCALE_FACTOR}):")
    print(f"ATE (RMSE): {ate:.4f} m")
    print(f"Path Length: {path_len:.2f} m")
    print(f"Drift Ratio: {drift:.2f} %")
    print("="*40)
    
    plt.figure(figsize=(10, 8))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', label='GT', linewidth=2)
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'r-', label='Pred (Scaled)', linewidth=2)
    plt.title(f"Final Trajectory with Scaling x{SCALE_FACTOR}\nATE: {ate:.2f}m")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig("final_trajectory_scaled.png")
    plt.show()

if __name__ == "__main__":
    run_final_inference_scaled("checkpoint_28.pt", "car_imu_data_full.csv", "car_ground_truth.csv")