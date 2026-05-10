import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from model_torch import IMUOdometryNet
from dataset import load_oxiod_dataset, load_dataset_6d_quat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model file')
    parser.add_argument('imu_file', help='Path to IMU CSV file')
    parser.add_argument('gt_file', help='Path to Ground Truth CSV file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = 200
    stride = 10

    # 加载模型
    model = IMUOdometryNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 加载数据
    gyro, acc, pos, ori = load_oxiod_dataset(args.imu_file, args.gt_file)
    [xg, xa], [yp, yq], _, _ = load_dataset_6d_quat(gyro, acc, pos, ori, window_size, stride)
    
    tensor_gyro = torch.FloatTensor(xg).to(device)
    tensor_acc = torch.FloatTensor(xa).to(device)

    # 推理
    pred_p_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(tensor_gyro), batch_size):
            batch_g = tensor_gyro[i : i+batch_size]
            batch_a = tensor_acc[i : i+batch_size]
            p, q = model(batch_g, batch_a)
            pred_p_list.append(p.cpu().numpy())
    
    pred_p = np.concatenate(pred_p_list, axis=0)

    # 积分
    gt_traj = np.cumsum(yp, axis=0)
    pred_traj = np.cumsum(pred_p, axis=0)
    
    # 对齐
    gt_traj = gt_traj - gt_traj[0]
    pred_traj = pred_traj - pred_traj[0]

    # --- 画 2D 俯视图 ---
    plt.figure(figsize=(10, 10)) # 正方形画布
    
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='Ground Truth', linewidth=2)
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Prediction', linewidth=2)
    
    # 标记起点和终点
    plt.plot(gt_traj[0,0], gt_traj[0,1], 'g^', markersize=10, label='Start')
    plt.plot(gt_traj[-1,0], gt_traj[-1,1], 'ko', markersize=8, label='End (GT)')
    plt.plot(pred_traj[-1,0], pred_traj[-1,1], 'rx', markersize=8, label='End (Pred)')

    plt.title(f'2D Top-Down View: {os.path.basename(args.imu_file)}')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    
    # 关键：强制 X 和 Y 轴比例一致！
    # 这样就算走廊再细，也不会被压扁
    plt.axis('equal') 

    save_name = 'result_2d.png'
    plt.savefig(save_name)
    print(f"[OK] 图片已保存为 {save_name}，快打开看看！")
    # plt.show() 

if __name__ == '__main__':
    main()