import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def run_sanity_check():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_paths = {
        'gt_pos': os.path.join(current_dir, 'gt_pos.npy'),
        'gt_quat': os.path.join(current_dir, 'gt_quat.npy'),
        'imu': os.path.join(current_dir, 'imu_data.npy')
    }

    # ==========================================
    # 关键参数区
    # ==========================================
    GRAVITY = 0.0        # 数据已去重力，设为0
    FIX_QUAT_ORDER = False # 根据你之前的图，False(默认)似乎比True(飞出天际)更好
    
    # [新] 是否将数据视为 Delta Velocity (m/s)
    # 如果你的数据是 m/s^2，设为 False
    # 如果你的数据是 m/s (看起来像)，设为 True
    IS_DELTA_VELOCITY = True 
    
    DT = 0.02 

    # ==========================================
    # 2. 加载与处理
    # ==========================================
    print("⏳ 正在加载数据...")
    try:
        gt_pos = np.load(file_paths['gt_pos'])
        gt_quat = np.load(file_paths['gt_quat'])
        imu_data = np.load(file_paths['imu'])
    except Exception as e:
        print(e); return

    n_samples = min(len(gt_pos), len(gt_quat), len(imu_data))
    gt_pos = gt_pos[:n_samples]
    gt_quat = gt_quat[:n_samples]
    
    # 假设前3列是加速度/速度增量
    acc_body = imu_data[:n_samples, :3]
    
    # 轴向修正 (如果跑出来圆圈是竖着的，这里可能要改)
    # 基于FLU坐标系：X前, Y左, Z上
    # 如果你的数据是 NED (X前, Y右, Z下)，可能需要 acc_body[:, 1] = -acc_body[:, 1]
    
    if FIX_QUAT_ORDER:
        gt_quat = np.roll(gt_quat, -1, axis=1) # wxyz -> xyzw

    # ==========================================
    # 3. 物理积分
    # ==========================================
    pred_pos = np.zeros((n_samples, 3))
    pred_vel = np.zeros((n_samples, 3))
    pred_pos[0] = gt_pos[0]

    print(f"计算中... (模式: {'Delta Velocity' if IS_DELTA_VELOCITY else 'Acceleration'})")
    
    r = R.from_quat(gt_quat)
    acc_world_all = r.apply(acc_body) # 旋转到世界系
    
    for i in range(1, n_samples):
        curr_input = acc_world_all[i-1]
        
        # --- 核心修改 ---
        if IS_DELTA_VELOCITY:
            # 假设输入已经是 v * dt，所以直接加
            pred_vel[i] = pred_vel[i-1] + curr_input
        else:
            # 标准加速度积分 a * dt
            pred_vel[i] = pred_vel[i-1] + curr_input * DT
            
        pred_pos[i] = pred_pos[i-1] + pred_vel[i] * DT

    # ==========================================
    # 4. 绘图 (带比例尺修正)
    # ==========================================
    print("正在绘图...")
    fig = plt.figure(figsize=(14, 6))

    # 3D 视图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gt_pos[:,0], gt_pos[:,1], gt_pos[:,2], 'k--', label='GT')
    ax1.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2], 'r-', label='Pred')
    ax1.set_title('Trajectory Check')
    ax1.legend()
    
    # 强制等比例 (关键！否则圆圈会被压扁)
    all_pos = np.concatenate([gt_pos, pred_pos])
    max_range = np.ptp(all_pos, axis=0).max() / 2.0
    mid_vals = np.mean(all_pos, axis=0)
    ax1.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
    ax1.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
    ax1.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)

    # XY 平面视图 (最容易看清圆圈)
    ax2 = fig.add_subplot(122)
    ax2.plot(gt_pos[:,0], gt_pos[:,1], 'k--', label='GT XY')
    ax2.plot(pred_pos[:,0], pred_pos[:,1], 'r-', label='Pred XY')
    ax2.set_title('Top-Down View (XY Plane)')
    ax2.set_aspect('equal', adjustable='datalim') # 强制等比例
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sanity_check()