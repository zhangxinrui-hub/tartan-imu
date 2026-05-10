import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from model import TartanIMUModel, load_checkpoint # 导入您的模型定义

def validate(data_file='processed_car_data.npy', model_file='checkpoint_24.pt'):
    # --------------------------
    # 1. 准备工作
    # --------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载数据: {data_file}")
    raw_data = np.load(data_file) # Shape: (N, 6) -> [acc, gyro]
    
    # 提取陀螺仪数据用于姿态积分 (Input is FLU frame)
    gyro_data = raw_data[:, 3:6] 
    
    # 加载模型
    print(f"加载模型: {model_file}")
    model = TartanIMUModel()
    model = load_checkpoint(model, model_file)
    model.to(device)
    model.eval()

    # --------------------------
    # 2. 推理循环 (Inference)
    # --------------------------
    print("开始推理...")
    pred_velocities = []
    
    # 滑动窗口参数
    seq_len = 80
    stride = 1  # 设为1可以获得最密集的预测
    
    # 将数据转为 Tensor
    data_tensor = torch.FloatTensor(raw_data).to(device) # (N, 6)
    
    with torch.no_grad():
        for i in range(0, len(data_tensor) - seq_len, stride):
            # 构造输入: [Batch=1, Channels=6, Seq=13]
            # 注意：模型通常要求 Input shape 为 (1, 6, 13)
            # data_tensor[i:i+seq_len] 是 (13, 6)，需要转置
            input_seq = data_tensor[i : i+seq_len].T.unsqueeze(0)
            
            outputs = model(input_seq)
            
            # 获取 'car' 的 3D 输出
            # output_3d shape: (1, seq_len, 3)
            # 我们取这一段序列的"最后一个时刻"的预测速度作为当前时刻的速度
            vel_pred = outputs['car'][1][0, -1, :].cpu().numpy()
            pred_velocities.append(vel_pred)
            
            if i % 1000 == 0:
                print(f"   处理进度: {i}/{len(data_tensor)}")

    pred_velocities = np.array(pred_velocities) # Shape: (M, 3) 速度 (Body Frame)
    
    # --------------------------
    # 3. 航位推算 (Dead Reckoning)
    # 位置 = 上一位置 + (姿态旋转矩阵 * 速度 * dt)
    # --------------------------
    print("计算轨迹 (积分)...")
    dt = 1.0 / 200.0 # 假设频率为 200Hz
    
    positions = [[0, 0, 0]] # 起始位置
    current_rot = R.from_quat([0, 0, 0, 1]) # 起始姿态 (Identity)
    
    # 注意：pred_velocities 的长度比 raw_data 少 (因为 seq_len)
    # 我们需要对齐时间，从第 seq_len 个时刻开始积分
    start_idx = seq_len 
    
    traj_points = []
    
    for i in range(len(pred_velocities)):
        # 1. 更新姿态 (使用陀螺仪积分)
        # 这里的 gyro 对应的是推理时刻的角速度
        # R_new = R_old * exp(omega * dt)
        step_gyro = gyro_data[start_idx + i]
        angle_delta = np.linalg.norm(step_gyro) * dt
        if angle_delta > 1e-6:
            axis = step_gyro / np.linalg.norm(step_gyro)
            r_delta = R.from_rotvec(axis * angle_delta)
            current_rot = current_rot * r_delta
            
        # 2. 将速度从 Body系 转到 World系
        # v_world = R * v_body
        v_body = pred_velocities[i]
        v_world = current_rot.apply(v_body)
        
        # 3. 更新位置
        last_pos = positions[-1]
        new_pos = last_pos + v_world * dt
        positions.append(new_pos)
        
    positions = np.array(positions)
    
    # --------------------------
    # 4. 可视化 & 保存
    # --------------------------
    print(f"生成轨迹图...")
    
    plt.figure(figsize=(10, 8))
    # 画 X-Y 平面图 (俯视图)
    plt.plot(positions[:, 0], positions[:, 1], label='Predicted Path', linewidth=2)
    plt.scatter(0, 0, c='red', marker='^', s=100, label='Start') # 起点
    plt.scatter(positions[-1, 0], positions[-1, 1], c='blue', marker='x', s=100, label='End') # 终点
    
    plt.title("Visual Validation: Car Trajectory (2D)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal') # 保证比例尺一致，这很重要！
    plt.grid(True)
    plt.legend()
    
    save_path = "trajectory_check.png"
    plt.savefig(save_path)
    print(f"[OK] 验证图已保存: {save_path}")
    
    # 保存轨迹数据
    np.savetxt("pred_trajectory.txt", positions)
    print("[OK] 轨迹坐标已保存: pred_trajectory.txt")

if __name__ == "__main__":
    validate()