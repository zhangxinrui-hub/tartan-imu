#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
import os

# 引入你的 model 定义
try:
    from model import TartanIMUModel, load_checkpoint 
except ImportError:
    print("错误: 找不到 model.py！")
    sys.exit(1)

def main():
    # ================= 1. 加载数据 =================
    print("Loading data ...")
    if not os.path.exists("imu_data.npy"):
        print("[ERROR] 错误: 找不到 imu_data.npy，请先运行 Preprocessing.py")
        return

    imu_data = np.load("imu_data.npy") # [N, 6]
    gt_pos   = np.load("gt_pos.npy")   # [N, 3]
    gt_quat  = np.load("gt_quat.npy")  # [N, 4]
    
    # 转换为 Tensor: [1, 6, N]
    # 注意：这里我们假设 imu_data 是 (N, 6)，需要转置成 (1, 6, N)
    imu_tensor = torch.from_numpy(imu_data).float()
    imu_tensor = imu_tensor.unsqueeze(0).transpose(1, 2)

    # ================= 2. 模型推理 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TartanIMUModel()
    checkpoint = "checkpoint_24.pt"
    if not os.path.exists(checkpoint):
        print(f"[ERROR] 错误: 找不到权重文件 {checkpoint}")
        return

    load_checkpoint(model, checkpoint, device=device)
    model.to(device)
    model.eval()

    print("Running inference ...")
    imu_tensor = imu_tensor.to(device)
    with torch.no_grad():
        outputs = model(imu_tensor)
        
    if "drone" in outputs:
        _, out3d, _ = outputs["drone"]
    else:
        # Fallback
        first_key = list(outputs.keys())[0]
        _, out3d, _ = outputs[first_key]

    # 转回 Numpy
    pred_vel_raw = out3d.squeeze(0).cpu().numpy()
    print(f"Raw output shape: {pred_vel_raw.shape}")

    # ================= 3. 形状自动修正 (Fix Shape) =================
    # 目标形状: (N, 3)
    # 如果是 (3, N)，就转置
    if pred_vel_raw.shape[0] == 3 and pred_vel_raw.shape[1] != 3:
        print(" -> Detected (3, N), Transposing to (N, 3)...")
        pred_vel_body = pred_vel_raw.T
    else:
        pred_vel_body = pred_vel_raw
        
    print(f"Corrected velocity shape: {pred_vel_body.shape}")

    # ================= 4. 对齐长度 =================
    N = min(len(pred_vel_body), len(gt_pos), len(gt_quat))
    print(f"Aligning to length: {N}")
    
    pred_vel_body = pred_vel_body[:N]
    gt_pos = gt_pos[:N]
    gt_quat = gt_quat[:N]
    
    # ================= 5. 坐标系旋转 (Body -> World) =================
    print("Rotating Velocity from Body to World ...")
    
    # gt_quat 是 Body(FRD) -> World(NED) 的旋转
    rots = R.from_quat(gt_quat)
    
    # 修正坐标系定义: FRD -> FLU
    # TartanIMU 输出的是 FLU 下的速度，但姿态是 FRD 下的
    # 公式: V_world = R_frd * R_frd2flu * V_flu
    r_frd2flu = R.from_matrix(np.diag([1, -1, -1]))
    rots_flu = rots * r_frd2flu
    
    # 现在输入形状肯定是 (N, 3) 了，apply 不会报错
    pred_vel_world = rots_flu.apply(pred_vel_body)
    
    # ================= 6. 积分 =================
    dt = 1.0 / 200.0
    # 累加得到位置
    pred_pos = np.cumsum(pred_vel_world, axis=0) * dt
    # 对齐起点
    pred_pos = pred_pos + (gt_pos[0] - pred_pos[0])

    # ================= 7. 画图 =================
    print("Plotting ...")
    plt.figure(figsize=(10, 8))
    
    # 2D 轨迹
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], "k--", label="Ground Truth", linewidth=2)
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], "r-", label="Stage-1 Pred (Rotated)", linewidth=2)
    
    # 标记起点
    plt.plot(gt_pos[0, 0], gt_pos[0, 1], "go", label="Start")
    
    plt.title(f"Trajectory Comparison (Duration: {N*dt:.1f}s)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    
    save_path = "final_result.png"
    plt.savefig(save_path)
    print(f"[OK] Success! Saved plot to {save_path}")
    # plt.show() 

if __name__ == "__main__":
    main()