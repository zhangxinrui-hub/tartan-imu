import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from model import TartanIMUModel, load_checkpoint

# ================= 配置 =================
CKPT_PATH = "checkpoint_28.pt"
DATA_PATH = "pretrain_1.npz"
TARGET_FREQ = 200.0
NORMALIZE_INPUT = True  # 必须开启，保证形状正确

def resample_data(timestamps, data_dict):
    ts = timestamps
    # 计算新的长度
    duration = ts[-1] - ts[0]
    new_len = int(duration * TARGET_FREQ)
    new_ts = np.linspace(ts[0], ts[-1], new_len)
    
    resampled = {}
    for k, v in data_dict.items():
        f = interp1d(ts, v, axis=0, kind='linear', fill_value="extrapolate")
        resampled[k] = f(new_ts)
    return new_ts, resampled

def run_3d_final():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    ts_raw = np.squeeze(data['retargetted_ts'])
    
    # 1. 重采样 (200Hz)
    data_map = {
        'imu': data['retargetted_imu'], 
        'pos': data['retargetted_pos'], 
        'quat': data['retargetted_quat']
    }
    ts_200, d200 = resample_data(ts_raw, data_map)
    imu, gt_pos, gt_quat = d200['imu'], d200['pos'], d200['quat']
    gt_quat /= np.linalg.norm(gt_quat, axis=1, keepdims=True)

    # 2. 预处理
    acc_raw, gyro_raw = imu[:, :3], imu[:, 3:]
    g_world = np.array([0, 0, 9.81])
    r_obj = R.from_quat(gt_quat)
    g_body = r_obj.inv().apply(g_world)
    acc_net = acc_raw - g_body
    
    # 静态 Bias 扣除
    acc_net -= np.mean(acc_net[:200], axis=0)
    gyro_raw -= np.mean(gyro_raw[:200], axis=0)
    
    # 归一化
    acc_net /= 9.81
    imu_input = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

    # 3. 推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TartanIMUModel().to(device)
    model = load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    
    with torch.no_grad():
        input_t = torch.from_numpy(imu_input).permute(1, 0).unsqueeze(0).to(device)
        outputs = model(input_t)
        head = outputs['human']
        v_xy = head[0].cpu().numpy().squeeze(0)
        v_z = head[2].cpu().numpy().squeeze(0)
        if v_xy.ndim==1: v_xy = v_xy[:,None]
        if v_z.ndim==1: v_z = v_z[:,None]

    # 4. 自动比例校准 (Auto-Scaling)
    print("Step 3: 计算并应用比例因子...")
    vel_body_raw = np.hstack([v_xy, v_z])
    
    dt = (ts_200[-1] - ts_200[0]) / (len(vel_body_raw) - 1)
    
    # 对齐索引
    idxs = np.linspace(0, len(gt_pos)-1, len(vel_body_raw)).astype(int)
    gt_pos_al = gt_pos[idxs]
    
    # 计算路程比
    gt_dist = np.sum(np.linalg.norm(np.diff(gt_pos_al, axis=0), axis=1))
    pred_dist_unscaled = np.sum(np.linalg.norm(vel_body_raw[:-1] * dt, axis=1))
    
    scale_factor = gt_dist / pred_dist_unscaled
    print(f"   -> Scale Factor: {scale_factor:.4f}")
    
    # 应用 Scale (XY 和 Z 同时放大，保持物理一致性)
    vel_body_scaled = vel_body_raw * scale_factor

    # 5. 积分
    r_pred = R.from_quat(gt_quat[idxs])
    vel_world = r_pred.apply(vel_body_scaled)
    
    # Z轴去均值 (防止飞天，保留放大后的波形)
    vel_world[:, 2] -= np.mean(vel_world[:, 2])
    
    pred_pos = np.zeros_like(gt_pos_al)
    pred_pos[0] = gt_pos_al[0]
    for k in range(1, len(pred_pos)):
        pred_pos[k] = pred_pos[k-1] + vel_world[k-1] * dt

    # 6. 绘图 (3图合一)
    ate_3d = np.sqrt(np.mean(np.linalg.norm(gt_pos_al - pred_pos, axis=1)**2))
    ate_xy = np.sqrt(np.mean(np.linalg.norm(gt_pos_al[:,:2] - pred_pos[:,:2], axis=1)**2))
    
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Top View
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(gt_pos_al[:,0], gt_pos_al[:,1], 'k--', lw=2, label='GT')
    ax1.plot(pred_pos[:,0], pred_pos[:,1], 'b-', lw=2, label=f'Pred (x{scale_factor:.2f})')
    ax1.plot(gt_pos_al[0,0], gt_pos_al[0,1], 'g^', ms=12, label='Start')
    ax1.plot(gt_pos_al[-1,0], gt_pos_al[-1,1], 'kx', ms=10, label='End')
    ax1.set_title(f"Top View (XY ATE: {ate_xy:.2f}m)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.axis('equal'); ax1.grid(True); ax1.legend()

    # 2. Z-Axis View
    ax2 = fig.add_subplot(1, 3, 2)
    t = np.arange(len(pred_pos)) * dt
    ax2.plot(t, gt_pos_al[:,2], 'k--', lw=2, label='GT Height')
    ax2.plot(t, pred_pos[:,2], 'b-', lw=1, alpha=0.8, label='Pred Height')
    ax2.set_title("Z-Axis (Scaled & De-meaned)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Height (m)")
    ax2.set_ylim(-2, 2) # 聚焦看波动
    ax2.grid(True); ax2.legend()
    
    # 3. 3D View
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(gt_pos_al[:,0], gt_pos_al[:,1], gt_pos_al[:,2], 'k--', label='GT')
    ax3.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2], 'b-', label='Pred')
    ax3.set_title(f"3D Trajectory (Total ATE: {ate_3d:.2f}m)")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("result_3d_final.png")
    print("[OK] 最终结果已保存为 result_3d_final.png")
    plt.show()

if __name__ == "__main__":
    run_3d_final()