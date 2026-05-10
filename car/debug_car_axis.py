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

def run_robust_debug():
    print("1. 读取原始 CSV 数据...")
    # 强制重新读取 CSV，确保数据源一致
    df = pd.read_csv('car_imu_data_full.csv')
    ts_raw = df['timestamp'].values
    acc_raw = df[['ax', 'ay', 'az']].values # Ax=前, Ay=左, Az=上 (已确认)
    gyr_raw = df[['gx', 'gy', 'gz']].values
    
    # 降采样到 200Hz
    target_fs = 200.0
    ts_new = np.arange(ts_raw[0], ts_raw[-1], 1.0/target_fs)
    ts_new = ts_new[ts_new <= ts_raw[-1]] # 防止越界
    
    print(f"  - 目标时间轴长度: {len(ts_new)} (200Hz)")
    
    f_acc = interp1d(ts_raw, acc_raw, axis=0, kind='linear')
    f_gyr = interp1d(ts_raw, gyr_raw, axis=0, kind='linear')
    acc_200 = f_acc(ts_new)
    gyr_200 = f_gyr(ts_new)
    
    # 转 rad/s
    if np.max(np.abs(gyr_200)) > 15.0:
        gyr_200 *= (np.pi / 180.0)

    # 2. 准备多种输入方案
    
    # 方案 A: 完美去重力 (尝试加载，如果长度不对就跳过)
    acc_perfect = None
    if os.path.exists('processed_car_data_final.npy'):
        try:
            data_perfect = np.load('processed_car_data_final.npy')
            # 检查长度是否匹配，如果不匹配说明是旧文件，不能用
            if data_perfect.shape[0] == len(ts_new):
                acc_perfect = data_perfect[:, :3]
                print("  - 加载完美去重力数据成功")
            else:
                print(f"[WARN] npy文件长度 ({data_perfect.shape[0]}) 与当前时间轴 ({len(ts_new)}) 不一致，将使用实时计算的数据。")
        except:
            pass
            
    if acc_perfect is None:
        # 如果没加载到，暂时用简单数据代替，或者填0
        acc_perfect = acc_200.copy()
        acc_perfect[:, 2] -= 9.81 

    # 方案 B: 简单去重力 (Simple) - 只减 Z 轴均值
    acc_simple = acc_200.copy()
    acc_simple[:, 2] -= 9.81
    
    # 3. 定义测试配置
    configs = [
        # (数据源, 缩放倍数, 约束, 颜色, 标签)
        (acc_perfect, 1.0, True, 'blue', 'Perfect Gravity'),
        (acc_simple,  1.0, True, 'green', 'Simple Gravity'),
        (acc_simple,  1.0, False, 'orange', 'Simple (No Constraint)'),
        (acc_simple,  5.0, True, 'red', 'Simple x5 (Amplify Signal)'), 
    ]
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TartanIMUModel().to(device)
    load_checkpoint(model, 'checkpoint_28.pt', device)
    model.eval()
    
    plt.figure(figsize=(10, 8))
    
    # 加载真值画底图
    if os.path.exists('car_ground_truth.csv'):
        df_gt = pd.read_csv('car_ground_truth.csv')
        gt_x = df_gt['x_gt'].values; gt_y = df_gt['y_gt'].values
        gt_dist = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2))
        plt.plot(gt_x - gt_x[0], gt_y - gt_y[0], 'k--', linewidth=2, label=f'GT ({gt_dist/1000:.1f}km)', alpha=0.5)

    print("\n开始鲁棒性测试...")
    
    dt = 1.0 / target_fs
    
    for acc_in, scale, constraint, color, label in configs:
        # 准备 Tensor
        acc_scaled = acc_in * scale
        feat = np.concatenate([acc_scaled, gyr_200], axis=1)
        inp = torch.FloatTensor(feat).permute(1, 0).unsqueeze(0).to(device)
        
        # 推理
        vel_list = []
        for i in range(0, inp.shape[2], 2000):
            end = min(i+2000, inp.shape[2])
            with torch.no_grad():
                res = model(inp[:,:,i:end])
                raw = res['car']
                if isinstance(raw, tuple):
                    val = torch.cat([raw[0], raw[2]], dim=-1)
                else:
                    val = raw
            vel_list.append(val.cpu())
            
        pred_vel_raw = torch.cat(vel_list, dim=1).numpy()[0]
        
        # 核心修复: 形状对齐 
        target_len = len(ts_new)
        current_len = pred_vel_raw.shape[0]
        
        if current_len != target_len:
            # print(f"  - [{label}] 发现频率不匹配: 输出 {current_len} vs 目标 {target_len}。正在插值...")
            # 创建原始时间轴 (假设均匀分布)
            t_old = np.linspace(0, 1, current_len)
            t_new = np.linspace(0, 1, target_len)
            f_resample = interp1d(t_old, pred_vel_raw, axis=0, kind='linear')
            pred_vel = f_resample(t_new)
        else:
            pred_vel = pred_vel_raw

        # 打印状态
        avg_speed = np.mean(np.linalg.norm(pred_vel, axis=1))
        # print(f"  - [{label}] 平均速度: {avg_speed:.2f} m/s")

        # 约束
        if constraint:
            pred_vel[:, 1] = 0.0
            
        # 积分
        yaw_rate = gyr_200[:, 2] # Gz
        yaw_rate -= np.mean(yaw_rate[:100])
        yaw = np.cumsum(yaw_rate * dt)
        
        # 坐标变换
        vx = pred_vel[:, 0] * np.cos(yaw) - pred_vel[:, 1] * np.sin(yaw)
        vy = pred_vel[:, 0] * np.sin(yaw) + pred_vel[:, 1] * np.cos(yaw)
        
        px = np.cumsum(vx * dt)
        py = np.cumsum(vy * dt)
        
        total_dist = np.sum(np.sqrt(np.diff(px)**2 + np.diff(py)**2))
        print(f"[{label}] 总里程: {total_dist:.1f} 米 (平均速度 {avg_speed:.2f} m/s)")
        
        plt.plot(px, py, color=color, linewidth=2, label=f"{label} ({total_dist:.0f}m)")

    plt.title("Car Trajectory Diagnosis")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('car_robust_debug.png')
    print("\n[OK] 诊断完成，请查看 car_robust_debug.png")
    plt.show()

if __name__ == "__main__":
    run_robust_debug()