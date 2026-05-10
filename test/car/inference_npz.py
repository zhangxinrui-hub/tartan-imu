import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from model import TartanIMUModel, load_checkpoint

def calculate_ate(gt_pos, pred_pos):
    """
    计算绝对轨迹误差 (ATE - Absolute Trajectory Error)
    这里使用 RMSE (Root Mean Square Error) 标准
    ATE = sqrt( (1/N) * sum( ||gt_i - pred_i||^2 ) )
    """
    # 计算每一帧的欧氏距离误差 (Euclidean Distance Error)
    errors = np.linalg.norm(gt_pos - pred_pos, axis=1)
    
    # 计算均方根误差 (RMSE)
    ate_rmse = np.sqrt(np.mean(errors**2))
    
    # 也可以计算最大误差作为参考
    max_error = np.max(errors)
    
    return ate_rmse, max_error

def run_inference_with_eval(ckpt_path, npz_path, device='cuda', chunk_size=1000, save_name="result_trajectory.png"):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    imu = data['retargetted_imu']       # [N, 6]
    gt_pos = data['retargetted_pos']    # [N, 3]
    gt_quat = data['retargetted_quat']  # [N, 4]
    timestamps = data['retargetted_ts'] # [N,]
    
    total_len = imu.shape[0]

    # 1. 预处理：去重力 + 去Bias + 归一化（与训练时一致）
    # retargetted_imu 包含重力（acc norm ≈ 9.8），必须去除
    acc_raw  = imu[:, :3]
    gyro_raw = imu[:, 3:]

    g_world = np.array([0.0, 0.0, 9.81])
    r_obj   = R.from_quat(gt_quat)          # scalar-last [x,y,z,w]
    g_body  = r_obj.inv().apply(g_world)    # 重力映射到机体系
    acc_net = acc_raw - g_body              # 去重力

    # 静态 Bias（取前200帧，约1秒静止段）
    acc_net  -= np.mean(acc_net[:200],  axis=0)
    gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)

    # 归一化加速度到 g 单位（与训练一致）
    acc_net /= 9.81

    processed_imu = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)
    print(f"预处理完成：acc_net 均值={acc_net.mean(axis=0)}, gyro 均值={gyro_raw.mean(axis=0)}")
    
    # 2. 模型推理
    model = TartanIMUModel().to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()
    
    all_vel_body = []

    print(f"正在推理...")
    with torch.no_grad():
        for i in range(0, total_len, chunk_size):
            end_idx = min(i + chunk_size, total_len)
            chunk_data = processed_imu[i:end_idx]
            imu_tensor = torch.FloatTensor(chunk_data).unsqueeze(0).permute(0, 2, 1).to(device)
            
            outputs, _ = model(imu_tensor)
            head_out = outputs['car']
            
            v_xy = head_out[0].cpu().numpy().squeeze()
            v_z  = head_out[2].cpu().numpy().squeeze()
            
            # 维度修正
            if v_xy.ndim == 1: v_xy = v_xy[np.newaxis, :]
            if v_z.ndim == 0: v_z = np.array([v_z])
            if v_z.ndim == 1: v_z = v_z[:, np.newaxis]
            
            chunk_vel = np.hstack([v_xy, v_z])
            all_vel_body.append(chunk_vel)
    
    vel_body = np.concatenate(all_vel_body, axis=0)
    output_len = len(vel_body)

    # 3. 约束与积分
    # 模型输出长度 ≈ 输入长度/4（ResNet 2次 stride=2），用均匀插值对齐 gt
    idxs = np.linspace(0, total_len - 1, output_len).astype(int)
    gt_quat_sub = gt_quat[idxs]
    gt_pos_sub  = gt_pos[idxs]

    # 车辆约束：无侧滑、平地
    # 诊断发现：模型前向速度在 block1_z（索引2），不在 block1[0]（索引0）
    vel_body[:, 0] = vel_body[:, 2]  # block1_z → Vx_forward
    vel_body[:, 1] = 0.0             # Vy = 0
    vel_body[:, 2] = 0.0             # Vz = 0

    # 旋转到世界系
    r = R.from_quat(gt_quat_sub)
    vel_world = r.apply(vel_body)

    # 用实际输出时间分辨率积分（约 0.02s/步，非原始 200Hz 的 0.005s）
    dt_effective = (timestamps[-1] - timestamps[0]) / output_len
    pred_pos = np.zeros((output_len, 3))
    pred_pos[0] = gt_pos_sub[0]
    for k in range(1, output_len):
        pred_pos[k] = pred_pos[k - 1] + vel_world[k - 1] * dt_effective

    # 为评估指标统一使用 gt_pos_sub
    gt_pos = gt_pos_sub

    # 4. 计算定量指标 (ATE)
    ate_rmse, max_err = calculate_ate(gt_pos, pred_pos)

    # 计算总路程长度 (用于评估漂移百分比)
    travel_dist = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    drift_percent = (ate_rmse / travel_dist) * 100
    
    print("="*40)
    print(f"定量评估结果 (Evaluation Metrics):")
    print(f"  - ATE (RMSE)  : {ate_rmse:.4f} m")
    print(f"  - Max Error   : {max_err:.4f} m")
    print(f"  - Trajectory Length: {travel_dist:.2f} m")
    print(f"  - Drift Ratio : {drift_percent:.2f}%")
    print("="*40)

    # 5. 可视化与保存
    plt.figure(figsize=(10, 8))
    
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', linewidth=2, label='Ground Truth')
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'r-', linewidth=2, label='TartanIMU (Stage 1)')
    
    # 标记
    plt.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', label='Start')
    plt.plot(gt_pos[-1, 0], gt_pos[-1, 1], 'kx', label='GT End')
    plt.plot(pred_pos[-1, 0], pred_pos[-1, 1], 'rx', label='Pred End')

    # 在图上添加 ATE 文本
    info_text = f"ATE (RMSE): {ate_rmse:.2f}m\nDrift: {drift_percent:.2f}%"
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f"Trajectory Evaluation: {save_name}")
    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid()
    
    # 保存图片
    print(f"正在保存图片到: {save_name}")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_inference_with_eval("checkpoint_28.pt", "pretrain_1.npz", save_name="eval_result_car.png")