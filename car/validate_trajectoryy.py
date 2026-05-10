import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d

try:
    from model import TartanIMUModel, load_checkpoint
except ImportError:
    print("[ERROR] 错误: 找不到 model.py")
    sys.exit(1)

def run_diagnosis():
    print("启动速度与航向分离诊断...")
    
    # ================= 配置 =================
    INPUT_GAIN = 5.0      # 唤醒模型的增益
    OUTPUT_SCALE = 3.17   # 里程校准系数
    # =======================================

    # 1. 加载数据
    df_imu = pd.read_csv('car_imu_data_full.csv')
    df_gt = pd.read_csv('car_ground_truth.csv')
    
    # 提取 IMU
    ts_raw = df_imu['timestamp'].values
    acc_raw = df_imu[['ax', 'ay', 'az']].values
    gyr_raw = df_imu[['gx', 'gy', 'gz']].values
    
    # 提取 GT
    gt_ts = df_gt['timestamp'].values
    gt_x = df_gt['x_gt'].values; gt_x -= gt_x[0]
    gt_y = df_gt['y_gt'].values; gt_y -= gt_y[0]
    # 计算 GT 速度 (用于对比)
    dt_gt = np.mean(np.diff(gt_ts))
    gt_vx = np.gradient(gt_x, gt_ts)
    gt_vy = np.gradient(gt_y, gt_ts)
    gt_speed = np.sqrt(gt_vx**2 + gt_vy**2)
    
    # 2. 预处理 (200Hz)
    target_fs = 200.0
    ts_new = np.arange(ts_raw[0], ts_raw[-1], 1.0/target_fs)
    ts_new = ts_new[ts_new <= ts_raw[-1]]
    
    f_acc = interp1d(ts_raw, acc_raw, axis=0, kind='linear')
    f_gyr = interp1d(ts_raw, gyr_raw, axis=0, kind='linear')
    acc_200 = f_acc(ts_new)
    gyr_200 = f_gyr(ts_new)
    
    if np.max(np.abs(gyr_200)) > 15.0: gyr_200 *= (np.pi / 180.0)

    # 简单去重力 + 输入增益
    acc_in = acc_200.copy()
    acc_in[:, 2] -= 9.81
    acc_in *= INPUT_GAIN
    
    # 3. 模型推理
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TartanIMUModel().to(device)
    load_checkpoint(model, 'checkpoint_28.pt', device)
    model.eval()
    
    feat = np.concatenate([acc_in, gyr_200], axis=1)
    inp = torch.FloatTensor(feat).permute(1, 0).unsqueeze(0).to(device)
    
    vel_list = []
    for i in range(0, inp.shape[2], 2000):
        end = min(i+2000, inp.shape[2])
        with torch.no_grad():
            res = model(inp[:,:,i:end])
            val = res['car']
            if isinstance(val, tuple): val = torch.cat([val[0], val[2]], dim=-1)
        vel_list.append(val.cpu())
            
    pred_vel_raw = torch.cat(vel_list, dim=1).numpy()[0]
    
    # 对齐 + 输出校准
    if pred_vel_raw.shape[0] != len(ts_new):
        t_old = np.linspace(0, 1, pred_vel_raw.shape[0])
        t_new = np.linspace(0, 1, len(ts_new))
        pred_vel = interp1d(t_old, pred_vel_raw, axis=0, kind='linear')(t_new)
    else:
        pred_vel = pred_vel_raw
        
    pred_vel *= OUTPUT_SCALE
    pred_speed = pred_vel[:, 0] # 假设 X 是前进轴
    
    # 4. 关键测试：使用 GT Yaw 进行积分 (Cheat Mode)
    # 我们需要把 GT Yaw 插值到 200Hz
    # 注意 GT 里的 yaw_gt 单位。如果是 deg 需要转 rad
    gt_yaw_raw = df_gt['yaw_gt'].values
    if np.max(np.abs(gt_yaw_raw)) > 7.0: # 超过 2pi 认为是度
        gt_yaw_raw = np.radians(gt_yaw_raw)
        
    # 处理 Yaw 的周期性 (Unwrap) 以便插值
    gt_yaw_unwrapped = np.unwrap(gt_yaw_raw)
    f_yaw = interp1d(gt_ts, gt_yaw_unwrapped, kind='linear', fill_value="extrapolate")
    yaw_cheat = f_yaw(ts_new)
    
    # 使用 预测速度 + GT航向 积分
    dt = 1.0/target_fs
    vx_cheat = pred_speed * np.cos(yaw_cheat)
    vy_cheat = pred_speed * np.sin(yaw_cheat)
    
    px_cheat = np.cumsum(vx_cheat * dt)
    py_cheat = np.cumsum(vy_cheat * dt)
    
    # 5. 画图诊断
    plt.figure(figsize=(14, 6))
    
    # 子图1: 速度对比 (检查模型预测的油门准不准)
    plt.subplot(1, 2, 1)
    # 把 GT 速度插值到同时间轴对比
    f_gt_speed = interp1d(gt_ts, gt_speed, kind='linear', fill_value="extrapolate")
    gt_speed_resamp = f_gt_speed(ts_new)
    
    plt.plot(ts_new, gt_speed_resamp, 'k-', alpha=0.6, label='GT Speed')
    plt.plot(ts_new, pred_speed, 'b-', alpha=0.8, label='Pred Speed (Model)')
    plt.title("Diagnosis 1: Speed Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True)
    
    # 子图2: 轨迹对比 (使用 GT Yaw)
    plt.subplot(1, 2, 2)
    plt.plot(gt_x, gt_y, 'k--', linewidth=2, label='Ground Truth')
    plt.plot(px_cheat, py_cheat, 'r-', linewidth=2, label='Pred (with GT Yaw)')
    plt.title("Diagnosis 2: Trajectory (using GT Yaw)")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('diagnosis_result.png')
    print("[OK] 诊断完成，请查看 diagnosis_result.png")
    print(f"Pred Mean Speed: {np.mean(pred_speed):.2f}, GT Mean Speed: {np.mean(gt_speed):.2f}")
    plt.show()

if __name__ == "__main__":
    run_diagnosis()