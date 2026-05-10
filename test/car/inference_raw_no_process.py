import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys

try:
    from model import TartanIMUModel, load_checkpoint
except ImportError:
    print("[ERROR] 错误: 找不到 model.py")
    sys.exit(1)

# ================= 配置 =================
CKPT_PATH = "checkpoint_28.pt"
DATA_PATH = "pretrain_1.npz" 
# 显存够大用 cuda，不够用 cpu (建议先试 cuda + chunk)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 1000  # 每次只处理 1000 帧，防止 28GB 显存溢出

def compute_ate(pred, gt):
    pred_aligned = pred - pred[0]
    gt_aligned = gt - gt[0]
    n = min(len(pred_aligned), len(gt_aligned))
    err = np.linalg.norm(pred_aligned[:n] - gt_aligned[:n], axis=1)
    return np.sqrt(np.mean(err**2))

def run_car_safe():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    
    imu = data['retargetted_imu']
    gt_pos = data['retargetted_pos']
    gt_quat = data['retargetted_quat']
    timestamps = data['retargetted_ts']
    
    print(f"数据总长度: {len(imu)} 帧")
    
    # 1. 静态去偏 (Car 专用)
    static_bias = np.mean(imu[:50], axis=0)
    imu_net = imu - static_bias
    
    # 2. 模型推理 (分块版)
    print(f"Step 2: 模型推理 (Chunk Size = {CHUNK_SIZE})...")
    model = TartanIMUModel().to(DEVICE)
    load_checkpoint(model, CKPT_PATH, DEVICE)
    model.eval()
    
    all_xy = []
    all_z = []
    
    total_len = len(imu_net)
    
    # --- 分块循环 ---
    with torch.no_grad():
        for i in range(0, total_len, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, total_len)
            
            # 准备小块数据
            chunk_data = imu_net[i:end]
            chunk_tensor = torch.FloatTensor(chunk_data).permute(1, 0).unsqueeze(0).to(DEVICE)
            
            # 推理
            outputs = model(chunk_tensor)
            head_out = outputs['car'] # Car Head
            
            # 提取结果
            v_xy = head_out[0].cpu().numpy().squeeze()
            v_z  = head_out[2].cpu().numpy().squeeze()
            
            # 维度保险
            if v_xy.ndim == 1: v_xy = v_xy[:, np.newaxis]
            if v_z.ndim == 0: v_z = np.full((len(v_xy), 1), v_z)
            elif v_z.ndim == 1: v_z = v_z[:, np.newaxis]
            
            all_xy.append(v_xy)
            all_z.append(v_z)
            
            # 打印进度
            print(f"   -> Processed {i} to {end} / {total_len}...", end='\r')
            
    print("\n推理完成，正在拼接...")
    v_xy_all = np.concatenate(all_xy, axis=0)
    
    # 3. 车辆约束 (Vy=0, Vz=0)
    # 只取 Vx (前进速度)
    v_forward = v_xy_all[:, 0:1]
    v_zeros = np.zeros_like(v_forward)
    vel_body = np.hstack([v_forward, v_zeros, v_zeros])

    # 4. 积分
    min_len = min(len(vel_body), len(gt_pos), len(gt_quat))
    vel_body = vel_body[:min_len]
    gt_pos = gt_pos[:min_len]
    gt_quat = gt_quat[:min_len]
    timestamps = timestamps[:min_len]
    
    dt = (timestamps[-1] - timestamps[0]) / (min_len - 1)
    
    r = R.from_quat(gt_quat)
    vel_world = r.apply(vel_body)
    
    pred_pos = np.zeros_like(gt_pos)
    pred_pos[0] = gt_pos[0]
    
    for k in range(1, min_len):
        pred_pos[k] = pred_pos[k-1] + vel_world[k-1] * dt

    # 5. 评估
    ate = compute_ate(pred_pos, gt_pos)
    
    # 计算路程 Scale
    gt_dist = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    pred_dist = np.sum(np.linalg.norm(np.diff(pred_pos, axis=0), axis=1))
    scale_ratio = gt_dist / pred_dist if pred_dist > 0 else 0

    print("="*40)
    print(f"[OK] Result (Chunked):")
    print(f"   - ATE: {ate:.4f} m")
    print(f"   - Scale (GT/Pred): {scale_ratio:.4f}")
    print("="*40)

    plt.figure(figsize=(10, 8))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', label='GT')
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'r-', label='Car Pred')
    plt.title(f"Car Trajectory (ATE: {ate:.2f}m)")
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.savefig("result_car_safe.png")
    print("结果已保存为 result_car_safe.png")
    plt.show()

if __name__ == "__main__":
    run_car_safe()