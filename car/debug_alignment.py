import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

def debug_alignment(imu_csv, gt_csv):
    print(f"正在诊断数据对齐问题...")
    
    # 1. 加载数据
    df_imu = pd.read_csv(imu_csv)
    df_gt = pd.read_csv(gt_csv)
    
    # 降采样 IMU (400 -> 200)
    df_imu = df_imu.iloc[::2].reset_index(drop=True)
    
    # 提取数据
    t_imu = df_imu['timestamp'].values
    acc_imu = df_imu[['ax', 'ay', 'az']].values
    gyro_imu = df_imu[['gx', 'gy', 'gz']].values
    
    t_gt = df_gt['timestamp'].values
    pos_gt = df_gt[['x_gt', 'y_gt', 'z_gt']].values
    
    # 转换 GT 欧拉角 -> 旋转矩阵
    euler_gt = df_gt[['roll_gt', 'pitch_gt', 'yaw_gt']].values
    r_gt_obj = R.from_euler('xyz', euler_gt, degrees=True)
    rot_gt = r_gt_obj.as_matrix() # [N, 3, 3]
    
    # 2. 计算 GT 的 "Body Frame Velocity" (真值速度)
    # v_world = (p2 - p1) / dt
    dt_gt = np.diff(t_gt)
    v_world = np.diff(pos_gt, axis=0) / dt_gt[:, None]
    # 补齐长度
    v_world = np.vstack([v_world, v_world[-1]])
    
    # 将世界速度投影回车身坐标系: v_body = R^T * v_world
    # 使用爱因斯坦求和约定进行批量矩阵乘法
    v_body_gt = np.einsum('nji,nj->ni', rot_gt, v_world)
    
    # 3. 对齐时间轴 (为了画图)
    # 我们只取前 60秒 (或者中间一段有运动的数据) 来观察细节
    # 找一段速度不为0的区域
    speed = np.linalg.norm(v_body_gt, axis=1)
    moving_idxs = np.where(speed > 2.0)[0] # 速度大于 2m/s 的时刻
    
    if len(moving_idxs) == 0:
        print("[WARN] 警告：GT 显示车辆似乎全程静止？")
        start_idx, end_idx = 0, 2000
    else:
        start_idx = moving_idxs[0]
        end_idx = min(start_idx + 2000, len(t_gt)) # 看 2000 帧
    
    # 截取 GT 片段
    t_slice = t_gt[start_idx:end_idx]
    v_body_slice = v_body_gt[start_idx:end_idx]
    
    # 截取对应的 IMU 片段 (简单通过时间戳查找)
    imu_start_mask = t_imu >= t_slice[0]
    imu_slice_data = df_imu[imu_start_mask]
    if len(imu_slice_data) > 2000:
        imu_slice_data = imu_slice_data.iloc[:2000]
    
    t_imu_slice = imu_slice_data['timestamp'].values
    acc_imu_slice = imu_slice_data[['ax', 'ay', 'az']].values
    gyro_imu_slice = imu_slice_data[['gx', 'gy', 'gz']].values
    
    # 4. 可视化对比
    plt.figure(figsize=(12, 10))
    
    # 子图 1: GT 速度 (基准)
    plt.subplot(3, 1, 1)
    plt.plot(t_slice, v_body_slice[:, 0], 'r-', label='GT Vel X (Forward)', linewidth=2)
    plt.plot(t_slice, v_body_slice[:, 1], 'g--', label='GT Vel Y (Lateral)')
    plt.title("1. Ground Truth Body Velocity (Confirm Forward Axis)")
    plt.legend()
    plt.grid()
    
    # 子图 2: IMU 加速度 (寻找相关性)
    plt.subplot(3, 1, 2)
    plt.plot(t_imu_slice, acc_imu_slice[:, 0], 'r-', label='IMU Acc X', alpha=0.7)
    plt.plot(t_imu_slice, acc_imu_slice[:, 1], 'g-', label='IMU Acc Y', alpha=0.7)
    plt.plot(t_imu_slice, acc_imu_slice[:, 2]-9.8, 'b-', label='IMU Acc Z (-9.8)', alpha=0.3)
    plt.title("2. IMU Acceleration (Which axis correlates with Velocity change?)")
    plt.legend()
    plt.grid()
    
    # 子图 3: IMU 角速度 (检查单位)
    plt.subplot(3, 1, 3)
    plt.plot(t_imu_slice, gyro_imu_slice[:, 0], 'r-', label='Gyro X')
    plt.plot(t_imu_slice, gyro_imu_slice[:, 1], 'g-', label='Gyro Y')
    plt.plot(t_imu_slice, gyro_imu_slice[:, 2], 'b-', label='Gyro Z')
    plt.title("3. IMU Gyroscope (Check Range: rad/s < 1.0, deg/s > 20.0)")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("debug_alignment.png")
    plt.show()

if __name__ == "__main__":
    imu_f = "car_imu_data_full.csv"
    gt_f = "car_ground_truth.csv"
    debug_alignment(imu_f, gt_f)