import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.spatial.transform import Rotation as R # 引入强大的旋转处理库

from model_torch import IMUOdometryNet
from dataset import load_oxiod_dataset, load_dataset_6d_quat

def main():
    # --- 1. 设置参数 ---
    parser = argparse.ArgumentParser()
    # 给参数设置默认值，方便在 IDE 里直接跑
    parser.add_argument('model', nargs='?', default='my_torch_model_final.pth', help='Path to model file')
    # 找一个存在的数据文件做默认值
    parser.add_argument('imu_file', nargs='?', default='Oxford Inertial Odometry Dataset/handheld/data1/syn/imu1.csv', help='Path to IMU CSV file')
    parser.add_argument('gt_file', nargs='?', default='Oxford Inertial Odometry Dataset/handheld/data1/syn/vi1.csv', help='Path to GT CSV file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = 200
    stride = 10

    # --- 2. 加载模型 ---
    print(f"Loading model from {args.model}...")
    model = IMUOdometryNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except FileNotFoundError:
        print("[ERROR] 错误: 找不到模型文件，请检查路径")
        return
    model.eval()

    # --- 3. 加载测试数据 ---
    print(f"Loading data: {args.imu_file}")
    if not os.path.exists(args.imu_file):
        print("[ERROR] 错误: 找不到数据文件")
        return

    gyro, acc, pos, ori = load_oxiod_dataset(args.imu_file, args.gt_file)
    # 注意：我们需要 yp (相对位移真值) 和 yq (相对旋转真值) 来做对比
    [xg, xa], [yp, yq], _, _ = load_dataset_6d_quat(gyro, acc, pos, ori, window_size, stride)

    tensor_gyro = torch.FloatTensor(xg).to(device)
    tensor_acc = torch.FloatTensor(xa).to(device)

    # --- 4. 运行预测 ---
    print("Running inference...")
    pred_p_list = []
    pred_q_list = [] # 我们需要旋转！
    
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(tensor_gyro), batch_size):
            batch_g = tensor_gyro[i : i+batch_size]
            batch_a = tensor_acc[i : i+batch_size]
            
            p, q = model(batch_g, batch_a)
            pred_p_list.append(p.cpu().numpy())
            pred_q_list.append(q.cpu().numpy()) # 保存预测的四元数
            
    pred_p = np.concatenate(pred_p_list, axis=0)
    pred_q = np.concatenate(pred_q_list, axis=0)

    # --- 5. 真正的轨迹重建 (Trajectory Integration) ---
    print("Reconstructing trajectory (with rotation)...")
    
    # 5.1 重建 Ground Truth (为了对比公平，我们也重新积分一遍 GT)
    # 假设初始状态为原点，初始姿态为单位阵
    gt_path = [np.array([0., 0., 0.])]
    curr_gt_pos = np.array([0., 0., 0.])
    curr_gt_rot = R.from_quat([0, 0, 0, 1]) # 初始无旋转 (scipy顺序: xyzw)

    # 5.2 重建 预测轨迹
    pred_path = [np.array([0., 0., 0.])]
    curr_pred_pos = np.array([0., 0., 0.])
    curr_pred_rot = R.from_quat([0, 0, 0, 1]) 

    # 开始循环积分
    for i in range(len(pred_p)):
        # --- 处理 GT ---
        # 1. 拿到真值的相对位移 (Body Frame)
        delta_p_gt = yp[i]
        # 2. 拿到真值的相对旋转 (Body Frame) -> dataset输出的是 xyzw 还是 wxyz? 
        # 假设 dataset 输出是 [w, x, y, z] (常见 PyTorch) -> 转 scipy [x, y, z, w]
        # 如果 dataset.py 里是 xyzw，请改这里！
        # [WARN] 这里假设 dataset 也是 wxyz
        q_gt_wxyz = yq[i]
        delta_q_gt = R.from_quat([q_gt_wxyz[1], q_gt_wxyz[2], q_gt_wxyz[3], q_gt_wxyz[0]])
        
        # 3. 更新 GT 位置: Global_Pos += Global_Rot * Local_Delta
        curr_gt_pos = curr_gt_pos + curr_gt_rot.apply(delta_p_gt)
        # 4. 更新 GT 姿态
        curr_gt_rot = curr_gt_rot * delta_q_gt
        gt_path.append(curr_gt_pos)

        # --- 处理 预测值 ---
        # 逻辑同上
        delta_p_pred = pred_p[i]
        q_pred_wxyz = pred_q[i]
        delta_q_pred = R.from_quat([q_pred_wxyz[1], q_pred_wxyz[2], q_pred_wxyz[3], q_pred_wxyz[0]])
        
        curr_pred_pos = curr_pred_pos + curr_pred_rot.apply(delta_p_pred)
        curr_pred_rot = curr_pred_rot * delta_q_pred
        pred_path.append(curr_pred_pos)

    gt_trajectory = np.array(gt_path)
    pred_trajectory = np.array(pred_path)

    # --- 6. 计算误差 RMSE ---
    # 简单的对齐：只对齐起点
    # (更高级的评估通常需要用 evo 库做 Umeyama 对齐，这里先简单做)
    min_len = min(len(gt_trajectory), len(pred_trajectory))
    gt_trajectory = gt_trajectory[:min_len]
    pred_trajectory = pred_trajectory[:min_len]

    diff = gt_trajectory - pred_trajectory
    # 计算 3D RMSE
    rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    # 计算 2D 平面误差 (通常大家更看重 xy 平面)
    rmse_xy = np.sqrt(np.mean(np.sum(diff[:, :2]**2, axis=1)))

    print(f"==============================================")
    print(f"3D Trajectory RMSE: {rmse:.4f} m")
    print(f"2D (XY) RMSE      : {rmse_xy:.4f} m")
    print(f"==============================================")

    # --- 7. 画图 ---
    plt.figure(figsize=(8, 8)) # 正方形画布
    plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'k--', label='Ground Truth', linewidth=2, alpha=0.6)
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-', label='Predicted', linewidth=2)
    
    # 标出起点和终点
    plt.scatter(gt_trajectory[0,0], gt_trajectory[0,1], c='g', marker='o', s=100, label='Start')
    plt.scatter(gt_trajectory[-1,0], gt_trajectory[-1,1], c='k', marker='x', s=100, label='GT End')
    plt.scatter(pred_trajectory[-1,0], pred_trajectory[-1,1], c='r', marker='x', s=100, label='Pred End')

    plt.title(f'Trajectory Evaluation (RMSE XY: {rmse_xy:.2f}m)')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    
    save_name = 'result_reconstruction.png'
    plt.savefig(save_name)
    print(f"Plot saved to {save_name}")
    plt.show()

if __name__ == '__main__':
    main()