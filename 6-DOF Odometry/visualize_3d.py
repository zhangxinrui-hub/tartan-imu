import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
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

    # 1. 准备数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = 200
    stride = 10

    print("正在计算轨迹数据...")
    model = IMUOdometryNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    if not os.path.exists(args.imu_file):
        print(f"Error: 文件 {args.imu_file} 不存在")
        return

    gyro, acc, pos, ori = load_oxiod_dataset(args.imu_file, args.gt_file)
    [xg, xa], [yp, yq], _, _ = load_dataset_6d_quat(gyro, acc, pos, ori, window_size, stride)
    
    tensor_gyro = torch.FloatTensor(xg).to(device)
    tensor_acc = torch.FloatTensor(xa).to(device)

    pred_p_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(tensor_gyro), batch_size):
            batch_g = tensor_gyro[i : i+batch_size]
            batch_a = tensor_acc[i : i+batch_size]
            p, q = model(batch_g, batch_a)
            pred_p_list.append(p.cpu().numpy())
    
    pred_p = np.concatenate(pred_p_list, axis=0)
    gt_traj = np.cumsum(yp, axis=0)
    pred_traj = np.cumsum(pred_p, axis=0)
    
    gt_traj = gt_traj - gt_traj[0]
    pred_traj = pred_traj - pred_traj[0]

    # === 关键修改 1：开启倍速模式 ===
    # 从原来的 5 改成 10 或 20，让它跑快点，能展示更长的距离
    skip = 10 
    gt_traj = gt_traj[::skip]
    pred_traj = pred_traj[::skip]
    
    print(f"数据准备完毕，全程共 {len(gt_traj)} 帧 (已倍速抽稀)")

    # 2. 设置画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('IMU Odometry: Full Trajectory')

    line_gt, = ax.plot([], [], [], 'b-', label='Ground Truth', linewidth=1.5, alpha=0.6)
    line_pred, = ax.plot([], [], [], 'r-', label='Prediction (Ours)', linewidth=2)
    point_gt, = ax.plot([], [], [], 'bo', markersize=5)
    point_pred, = ax.plot([], [], [], 'ro', markersize=5)

    ax.legend()

    def update(num):
        ax.view_init(elev=40, azim=num * 0.5) 
        
        current_idx = min(num, len(gt_traj) - 1)
        
        # 更新线条
        line_gt.set_data(gt_traj[:num+1, 0], gt_traj[:num+1, 1])
        line_gt.set_3d_properties(gt_traj[:num+1, 2])

        line_pred.set_data(pred_traj[:num+1, 0], pred_traj[:num+1, 1])
        line_pred.set_3d_properties(pred_traj[:num+1, 2])

        # 更新点
        point_gt.set_data([gt_traj[current_idx, 0]], [gt_traj[current_idx, 1]])
        point_gt.set_3d_properties([gt_traj[current_idx, 2]])

        point_pred.set_data([pred_traj[current_idx, 0]], [pred_traj[current_idx, 1]])
        point_pred.set_3d_properties([pred_traj[current_idx, 2]])
        
        # 动态视野
        current_data = gt_traj[:num+1]
        if len(current_data) > 0:
            x_min, x_max = current_data[:,0].min(), current_data[:,0].max()
            y_min, y_max = current_data[:,1].min(), current_data[:,1].max()
            
            margin = 5.0
            # 保证视野不会太小
            if x_max - x_min < 10: x_min -= 5; x_max += 5
            if y_max - y_min < 10: y_min -= 5; y_max += 5
            
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            
            z_mid = (current_data[:,2].min() + current_data[:,2].max()) / 2
            ax.set_zlim(z_mid - 8, z_mid + 8)

        return line_gt, line_pred, point_gt, point_pred

    # === 关键修改 2：取消 600 帧限制 ===
    # 限制最大 1000 帧，防止 GIF 过大，但通常够跑完全程了
    total_frames = min(len(gt_traj), 1200)
    print(f"正在渲染全程动画 ({total_frames} 帧)... 请耐心等待...")
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=30, blit=False)
    
    save_name = 'trajectory_full_loop.gif'
    writer = animation.PillowWriter(fps=30)
    ani.save(save_name, writer=writer)
    
    print(f"成功！完整绕圈动画已保存为 {save_name}")

if __name__ == '__main__':
    main()