import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from model import TartanIMUModel, load_checkpoint

# 复用之前的滤波和GT处理函数
def low_pass_filter_6d(data, cutoff=15.0, fs=200.0, order=4):
    """ 
    【修改】将截止频率提高到 15Hz，保留更多动态特征
    """
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
    diffs = np.diff(pos_gt, axis=0)
    yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
    yaw = np.append(yaw, yaw[-1])
    new_r = R.from_euler('z', yaw, degrees=False)
    f_pos = interp1d(ts_gt, pos_gt, axis=0, kind='linear', fill_value="extrapolate")
    pos_aligned = f_pos(target_timestamps)
    from scipy.spatial.transform import Slerp
    slerp = Slerp(ts_gt, new_r)
    valid_mask = (target_timestamps >= ts_gt[0]) & (target_timestamps <= ts_gt[-1])
    target_ts_valid = target_timestamps[valid_mask]
    r_interp = slerp(target_ts_valid)
    quat_aligned = r_interp.as_quat()
    return valid_mask, pos_aligned[valid_mask], quat_aligned

def debug_speed_and_scale(ckpt_path, imu_csv, gt_csv, device='cpu', chunk_size=1000):
    print("Loading data for Speed Analysis...")
    df_imu = pd.read_csv(imu_csv).iloc[::2].reset_index(drop=True)
    imu_data_raw = df_imu[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
    timestamps = df_imu['timestamp'].values
    
    # 1. 稍微放宽滤波器 (4Hz -> 15Hz)
    print("Filter: 15Hz (Allowing more dynamics)...")
    imu_data_clean = low_pass_filter_6d(imu_data_raw, cutoff=15.0, fs=200.0)
    
    valid_mask, gt_pos, gt_quat = process_gt_recalc_yaw(gt_csv, timestamps)
    imu_data_clean = imu_data_clean[valid_mask]
    timestamps = timestamps[valid_mask]
    
    # 去 Bias
    static_bias = np.mean(imu_data_clean[:100], axis=0)
    processed_imu = imu_data_clean - static_bias
    
    # 2. 推理
    model = TartanIMUModel().to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()
    
    all_vel = []
    total_len = len(processed_imu)
    
    with torch.no_grad():
        for i in range(0, total_len, chunk_size):
            end_idx = min(i + chunk_size, total_len)
            chunk_data = processed_imu[i:end_idx]
            imu_tensor = torch.FloatTensor(chunk_data).unsqueeze(0).permute(0, 2, 1).to(device)
            out = model(imu_tensor)['car']
            # 这里我们只取模长 (Speed) 来对比，不需要管方向
            vx = out[0].cpu().numpy().squeeze()
            if vx.ndim==1: vx = vx[:,None]
            all_vel.append(vx) # 只拿 Forward Velocity
            
    # 插值
    pred_vel_raw = np.concatenate(all_vel, axis=0)
    f_interp = interp1d(np.linspace(0,1,len(pred_vel_raw)), pred_vel_raw, axis=0, kind='linear', fill_value='extrapolate')
    pred_vel_x = f_interp(np.linspace(0,1,total_len)) # [N, 2]
    pred_speed = np.linalg.norm(pred_vel_x, axis=1) # 计算标量速度
    
    # 3. 计算 GT 速度
    dt = 0.005
    gt_vel_vec = np.diff(gt_pos, axis=0) / dt
    gt_vel_vec = np.vstack([gt_vel_vec, gt_vel_vec[-1]])
    gt_speed = np.linalg.norm(gt_vel_vec, axis=1)
    
    # 4. 寻找最佳缩放比例 (Optimal Scale Factor)
    # 简单的比率：GT平均速度 / 预测平均速度
    # 过滤掉静止部分
    moving_mask = gt_speed > 1.0
    if np.sum(moving_mask) > 100:
        scale_factor = np.mean(gt_speed[moving_mask]) / np.mean(pred_speed[moving_mask])
    else:
        scale_factor = 1.0
        
    print("="*40)
    print(f"检测到的最佳缩放比例 (Scale Factor): {scale_factor:.2f}")
    print("这表示模型预测的速度比真实速度小了这么多倍。")
    print("="*40)

    # 5. 画图
    plt.figure(figsize=(12, 6))
    plt.plot(gt_speed, 'k-', linewidth=1.5, alpha=0.6, label='GT Speed')
    plt.plot(pred_speed, 'r-', linewidth=1.5, label='Pred Speed (Original)')
    plt.plot(pred_speed * scale_factor, 'g--', linewidth=1.0, label=f'Pred Scaled (x{scale_factor:.1f})')
    
    plt.title(f"Speed Analysis: Does the model see the motion?\nScale Factor Needed: {scale_factor:.2f}")
    plt.xlabel("Time Steps")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid()
    plt.savefig("debug_speed_scale.png")
    plt.show()

if __name__ == "__main__":
    imu_f = "car_imu_data_full.csv"
    gt_f = "car_ground_truth.csv"
    ckpt = "checkpoint_28.pt"
    debug_speed_and_scale(ckpt, imu_f, gt_f)