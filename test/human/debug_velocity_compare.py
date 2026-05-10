import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from model import TartanIMUModel, load_checkpoint

# ================= 配置 =================
CKPT_PATH = "checkpoint_28.pt"
DATA_PATH = "pretrain_1.npz"
NORM_FACTOR = 9.8675 # 使用之前取证得到的数值

def compare_velocity():
    # 1. 加载数据
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    imu_raw = data['retargetted_imu']
    gt_pos  = data['retargetted_pos']
    gt_quat = data['retargetted_quat']
    timestamps = np.squeeze(data['retargetted_ts'])
    
    # 截断对齐
    min_len = min(len(imu_raw), len(gt_pos), len(gt_quat), len(timestamps))
    imu_raw = imu_raw[:min_len]
    gt_pos  = gt_pos[:min_len]
    gt_quat = gt_quat[:min_len]
    timestamps = timestamps[:min_len]

    # 2. 准备模型输入
    print("Step 1: 预处理 (全序列)...")
    acc_raw = imu_raw[:, :3]
    gyro_raw = imu_raw[:, 3:]
    
    # 去重力 & 归一化
    g_world = np.array([0, 0, NORM_FACTOR]) 
    r = R.from_quat(gt_quat)
    g_body = r.inv().apply(g_world)
    acc_net = (acc_raw - g_body) / NORM_FACTOR
    
    imu_net = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

    # 3. 模型推理 (得到 预测速度)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TartanIMUModel().to(device)
    model = load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    
    print(f"Step 2: 模型推理...")
    with torch.no_grad():
        input_tensor = torch.from_numpy(imu_net).permute(1, 0).unsqueeze(0).to(device)
        outputs = model(input_tensor)
        head_out = outputs['human']
        
        v_xy = head_out[0].cpu().numpy().squeeze(0)
        v_z  = head_out[2].cpu().numpy().squeeze(0)
        if v_xy.ndim == 1: v_xy = v_xy[:, np.newaxis]
        if v_z.ndim == 1: v_z = v_z[:, np.newaxis]
        
        vel_body_pred = np.hstack([v_xy, v_z])

    # 4. 计算真值速度 (GT Body Velocity)
    # 这是“照妖镜”，我们用 GT 位置微分，然后转到 Body 系
    print(f"Step 3: 计算真值速度...")
    pred_len = len(vel_body_pred)
    
    # 下采样 GT 对齐 50Hz
    idxs = np.linspace(0, min_len-1, pred_len).astype(int)
    pos_aligned = gt_pos[idxs]
    quat_aligned = gt_quat[idxs]
    ts_aligned = timestamps[idxs]
    
    # 计算世界坐标系速度 V_world = dP / dt
    dt_seq = np.diff(ts_aligned)
    dt_seq = np.append(dt_seq, dt_seq[-1]) # 补齐长度
    # 避免 dt 为 0
    dt_seq[dt_seq < 1e-4] = 0.02 
    
    vel_world_gt = np.diff(pos_aligned, axis=0, prepend=pos_aligned[0:1]) / dt_seq[:, None]
    
    # 关键一步：把世界系真值速度 转回 Body系
    r_aligned = R.from_quat(quat_aligned)
    vel_body_gt = r_aligned.inv().apply(vel_world_gt)

    # 5. 绘图对比
    print("Step 4: 绘图对比...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    labels = ['Vx (Forward)', 'Vy (Left)', 'Vz (Up)']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(ts_aligned, vel_body_gt[:, i], 'k', alpha=0.5, label='Ground Truth', linewidth=1)
        ax.plot(ts_aligned, vel_body_pred[:, i], 'b', label='Prediction', linewidth=1.5)
        ax.set_title(f"{labels[i]} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Velocity (m/s)")
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("velocity_diagnosis.png")
    print("[OK] 诊断图已保存: velocity_diagnosis.png")
    plt.show()

if __name__ == "__main__":
    compare_velocity()