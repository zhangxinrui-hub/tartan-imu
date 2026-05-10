import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from model import TartanIMUModel, load_checkpoint

# ================= 配置 =================
CKPT_PATH = "checkpoint_28.pt"
DATA_PATH = "pretrain_1.npz"
TARGET_FREQ = 200.0

def apply_highpass(data, cutoff=0.8, fs=200.0):
    """ 高通滤波: 提取步伐波动，滤除高度变化 """
    nyq = 0.5 * fs
    b, a = butter(4, cutoff/nyq, btype='high')
    return filtfilt(b, a, data)

def run_waveform_analysis():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    ts_raw = np.squeeze(data['retargetted_ts'])
    
    # 1. 重采样 (200Hz)
    ts = ts_raw
    new_len = int((ts[-1] - ts[0]) * TARGET_FREQ)
    new_ts = np.linspace(ts[0], ts[-1], new_len)
    
    # 插值 GT
    f_pos = interp1d(ts, data['retargetted_pos'], axis=0, fill_value="extrapolate")
    gt_pos_200 = f_pos(new_ts)
    
    f_imu = interp1d(ts, data['retargetted_imu'], axis=0, fill_value="extrapolate")
    imu_200 = f_imu(new_ts)
    
    f_quat = interp1d(ts, data['retargetted_quat'], axis=0, fill_value="extrapolate")
    gt_quat_200 = f_quat(new_ts)
    gt_quat_200 /= np.linalg.norm(gt_quat_200, axis=1, keepdims=True)

    # 2. 预处理
    acc = imu_200[:, :3]
    gyro = imu_200[:, 3:]
    
    # 去重力
    g_world = np.array([0, 0, 9.81])
    r_obj = R.from_quat(gt_quat_200)
    g_body = r_obj.inv().apply(g_world)
    acc_net = acc - g_body
    
    # Bias 校准
    acc_net -= np.mean(acc_net[:200], axis=0)
    gyro -= np.mean(gyro[:200], axis=0)
    
    # 归一化
    acc_net /= 9.81
    imu_input = np.concatenate([acc_net, gyro], axis=1).astype(np.float32)

    # 3. 推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TartanIMUModel().to(device)
    model = load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    
    with torch.no_grad():
        inp = torch.from_numpy(imu_input).permute(1, 0).unsqueeze(0).to(device)
        out = model(inp)['human']
        v_xy = out[0].cpu().numpy().squeeze(0)
        v_z = out[2].cpu().numpy().squeeze(0)
        if v_xy.ndim==1: v_xy = v_xy[:,None]
        if v_z.ndim==1: v_z = v_z[:,None]

    # 4. 后处理：只看波形
    vel_body = np.hstack([v_xy, v_z])
    
    # 对齐长度
    pred_len = len(vel_body)
    idxs = np.linspace(0, len(gt_pos_200)-1, pred_len).astype(int)
    gt_pos_final = gt_pos_200[idxs]
    gt_quat_final = gt_quat_200[idxs]
    
    # 积分原始预测 (含漂移)
    dt = (new_ts[-1] - new_ts[0]) / (pred_len - 1)
    r_pred = R.from_quat(gt_quat_final)
    vel_world = r_pred.apply(vel_body)
    
    pred_pos = np.zeros_like(gt_pos_final)
    for k in range(1, len(pred_pos)):
        pred_pos[k] = pred_pos[k-1] + vel_world[k-1] * dt

    # 5. 【核心步骤】提取动态波形 (Both GT and Pred)
    # 截止频率 0.8Hz (人走路通常 1.5Hz-2Hz，所以能保留步伐，滤除起伏)
    z_gt_dyn = apply_highpass(gt_pos_final[:, 2], cutoff=0.8, fs=200.0)
    z_pred_dyn = apply_highpass(pred_pos[:, 2], cutoff=0.8, fs=200.0)

    # 6. 绘图 (Zoom In)
    # 选取中间一段 10秒 (比如 50s - 60s)
    center_idx = len(pred_pos) // 2
    window = int(10.0 / dt) # 10 seconds
    start, end = center_idx, center_idx + window
    
    time_zoom = np.arange(window) * dt
    
    fig = plt.figure(figsize=(12, 8))
    
    # 图1: XY 轨迹 (确认整体是对的)
    ax1 = fig.add_subplot(2, 1, 1)
    ate_xy = np.sqrt(np.mean(np.linalg.norm(gt_pos_final[:,:2] - pred_pos[:,:2], axis=1)**2))
    ax1.plot(gt_pos_final[:,0], gt_pos_final[:,1], 'k--', label='GT')
    ax1.plot(pred_pos[:,0], pred_pos[:,1], 'b-', lw=2, label='Pred')
    ax1.set_title(f"1. Overall Trajectory (XY ATE: {ate_xy:.2f}m)")
    ax1.axis('equal'); ax1.legend()
    
    # 图2: Z轴波形对比 (放大看细节)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(time_zoom, z_gt_dyn[start:end], 'k--', lw=2, alpha=0.7, label='GT Step Waveform')
    ax2.plot(time_zoom, z_pred_dyn[start:end], 'b-', lw=2, label='Pred Step Waveform')
    
    ax2.set_title("2. Z-Axis Step Dynamics (Zoomed In 10s Window)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vertical Dynamics (m)")
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("z_waveform_check.png")
    print("[OK] 结果已保存为 z_waveform_check.png")
    plt.show()

if __name__ == "__main__":
    run_waveform_analysis()