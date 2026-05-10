import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline, interp1d

# =======================
# CONFIG
# =======================

CONFIG = {
    'imu_path': 'imu_flight-01p-ellipse.csv',
    'gt_path':  'mocap_flight-01p-ellipse.csv',
    'target_freq': 200.0,
    'g_world_frd': np.array([0, 0, 9.81]),   # World(FRD) gravity
    'static_duration': 1.0,
    'save_dir': '.'
}

TIME_DIVISOR = 1e6

# =======================
# TIME SYNC
# =======================

def load_and_sync_time(imu_path, gt_path, target_freq):

    imu_df = pd.read_csv(imu_path)
    gt_df  = pd.read_csv(gt_path)

    def parse_time(df):
        return df['timestamp'].values / TIME_DIVISOR

    t_imu = parse_time(imu_df)
    t_gt  = parse_time(gt_df)

    t_start = max(t_imu[0], t_gt[0]) + 0.1
    t_end   = min(t_imu[-1], t_gt[-1]) - 0.1

    t_target = np.arange(t_start, t_end, 1.0 / target_freq)

    print(f"Synced duration: {t_target[-1] - t_target[0]:.2f}s")

    return imu_df, gt_df, t_imu, t_gt, t_target

# =======================
# MAIN PIPELINE
# =======================

def process_data():

    imu_df, gt_df, t_imu, t_gt, t_target = load_and_sync_time(
        CONFIG['imu_path'],
        CONFIG['gt_path'],
        CONFIG['target_freq']
    )

    # ===========================
    # 1. IMU Interpolate
    # ===========================
    
    acc_raw  = imu_df[['accel_x','accel_y','accel_z']].values
    gyro_raw = imu_df[['gyro_x','gyro_y','gyro_z']].values

    f_acc  = interp1d(t_imu, acc_raw,  axis=0, fill_value="extrapolate")
    f_gyro = interp1d(t_imu, gyro_raw, axis=0, fill_value="extrapolate")

    acc_frd  = f_acc(t_target)
    gyro_frd = f_gyro(t_target)

    # ===========================
    # 2. GT Pose Interpolate
    # ===========================

    # ---- Quaternion ----
    try:
        rot_cols = [f'drone_rot[{i}]' for i in range(9)]
        rot_raw  = gt_df[rot_cols].values.reshape(-1,3,3)
        quat_gt  = R.from_matrix(rot_raw).as_quat()

    except KeyError:
        quat_gt = gt_df[['qx','qy','qz','qw']].values

    # ---- Slerp to IMU timeline ----
    _, idx = np.unique(t_gt, return_index=True)

    slerp = Slerp(t_gt[idx], R.from_quat(quat_gt[idx]))
    rots_interp = slerp(t_target)

    # ===========================
    # 3. Position & World Velocity
    # ===========================

    pos_world_raw = gt_df[['drone_x','drone_y','drone_z']].values

    cs_pos = CubicSpline(t_gt, pos_world_raw, axis=0)
    
    pos_world = cs_pos(t_target)
    vel_world = cs_pos(t_target, 1)

    # ===========================
    # 4. Gravity in BODY frame
    # ===========================

    # g_body = R_BW * g_world
    g_body = rots_interp.inv().apply(CONFIG['g_world_frd'])

    # ===========================
    # 5. Bias Removal
    # ===========================

    static_N = int(CONFIG['static_duration'] * CONFIG['target_freq'])

    gyro_bias = np.mean(gyro_frd[:static_N], axis=0)
    gyro_frd -= gyro_bias

    acc_bias = np.mean(acc_frd[:static_N] - g_body[:static_N], axis=0)
    acc_frd -= acc_bias

    # ===========================
    # 6. Gravity Removal
    # ===========================

    acc_pure_frd = acc_frd - g_body

    # ===========================
    # 7. World Vel → Body Vel
    # ===========================

    vel_body_frd = rots_interp.inv().apply(vel_world)

    # ===========================
    # 8. FRD → FLU
    # ===========================

    def frd_to_flu(x):
        y = x.copy()
        y[:,1] *= -1
        y[:,2] *= -1
        return y

    acc_flu  = frd_to_flu(acc_pure_frd)
    gyro_flu = frd_to_flu(gyro_frd)
    vel_flu  = frd_to_flu(vel_body_frd)
    pos_flu  = frd_to_flu(pos_world)

    # ===========================
    # 9. Save for Stage-1
    # ===========================

    imu_data = np.hstack([acc_flu, gyro_flu])

    np.save("imu_data.npy", imu_data.astype(np.float32))
    np.save("gt_vel.npy",   vel_flu.astype(np.float32))
    np.save("gt_pos.npy",   pos_flu.astype(np.float32))
    np.save("gt_quat.npy",  rots_interp.as_quat().astype(np.float32))

    print("="*50)
    print("[OK] Preprocess done and files saved:")
    print("   imu_data.npy  (acc_nog + gyro)  in BODY FLU")
    print("   gt_vel.npy    (velocity GT)    in BODY FLU")
    print("   gt_pos.npy    (trajectory)     in WORLD FLU")
    print("   gt_quat.npy   (pose)           world ⟵ body")
    print("="*50)


if __name__ == "__main__":
    process_data()
