import numpy as np
import pandas as pd
import os

# ============================================================
# 参数区（一般不需要改）
# ============================================================

WAIST_FILE = "4d91_long_waist_imu.xlsx"

TIME_COL = "time after start [s]"
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]
POS_COLS = ["gt_pos_body_x", "gt_pos_body_y", "gt_pos_body_z"]

FS_TARGET = 200.0        # TartanIMU Stage 1 要求
GT_WINDOW_SEC = 1.0      # 1 秒监督窗口
STATIC_SEC = 2.0         # 用前 2 秒估计重力方向
G = 9.81                 # 重力常数

# ============================================================
# 工具函数
# ============================================================

def drop_nan_rows(t, *arrays):
    stacked = np.column_stack(arrays)
    mask = ~np.isnan(stacked).any(axis=1)
    return (t[mask],) + tuple(a[mask] for a in arrays)


def resample_to_fs(t, data, fs_new):
    t0, t1 = t[0], t[-1]
    t_new = np.arange(t0, t1, 1.0 / fs_new)
    data_new = np.vstack([
        np.interp(t_new, t, data[:, k])
        for k in range(data.shape[1])
    ]).T
    return t_new, data_new


def compute_stage1_gt_velocity(pos_body, fs, window_sec):
    window = int(fs * window_sec)
    if len(pos_body) <= window:
        raise ValueError("Sequence too short for Stage 1 supervision")
    dp = pos_body[window:] - pos_body[:-window]
    return dp / window_sec


def remove_gravity_by_projection(acc, g_dir, g=9.81):
    """
    acc: (N,3) in body frame (FLU)
    g_dir: unit gravity direction in body frame
    """
    return acc - g * g_dir


# ============================================================
# 主流程
# ============================================================

def preprocess_waist_stage1_v2():
    if not os.path.exists(WAIST_FILE):
        raise FileNotFoundError(WAIST_FILE)

    print("Loading waist IMU data...")
    df = pd.read_excel(WAIST_FILE)

    # -------- 1. 读原始数据 --------
    t_raw = df[TIME_COL].to_numpy()
    acc_raw = df[ACC_COLS].to_numpy()
    gyr_raw = df[GYR_COLS].to_numpy()
    pos_body_raw = df[POS_COLS].to_numpy()

    # -------- 2. 清理 NaN（修 GT NaN）--------
    t_raw, acc_raw, gyr_raw, pos_body_raw = drop_nan_rows(
        t_raw, acc_raw, gyr_raw, pos_body_raw
    )

    # -------- 3. gyro: deg/s → rad/s --------
    gyr_raw = np.deg2rad(gyr_raw)

    # -------- 4. 重采样到 200 Hz（IMU + GT 同步）--------
    print("Resampling to 200 Hz...")
    t, acc = resample_to_fs(t_raw, acc_raw, FS_TARGET)
    _, gyr = resample_to_fs(t_raw, gyr_raw, FS_TARGET)
    _, pos_body = resample_to_fs(t_raw, pos_body_raw, FS_TARGET)

    # -------- 5. waist 的 FLU 轴对齐 --------
    # 结论来自你前 2 秒统计：
    #   Z_up     ← acc_x
    #   Y_left   ← acc_y
    #   X_forward← -acc_z
    print("Applying waist FLU axis alignment...")
    acc_flu = np.stack([
        -acc[:, 2],   # X forward
         acc[:, 1],   # Y left
         acc[:, 0],   # Z up
    ], axis=1)

    gyr_flu = np.stack([
        -gyr[:, 2],
         gyr[:, 1],
         gyr[:, 0],
    ], axis=1)

    # -------- 6. 用静止段估计重力方向 --------
    static_len = int(STATIC_SEC * FS_TARGET)
    g_vec = acc_flu[:static_len].mean(axis=0)
    g_dir = g_vec / np.linalg.norm(g_vec)

    print(f"Estimated gravity direction (body frame): {g_dir}")

    # -------- 7. 投影法去重力（关键修复）--------
    print("Removing gravity by projection...")
    acc_lin = remove_gravity_by_projection(acc_flu, g_dir, g=G)

    # -------- 8. 组装 Stage 1 IMU 输入 --------
    imu_stage1 = np.hstack([acc_lin, gyr_flu])  # (N,6)

    # -------- 9. Stage 1 GT velocity（1 秒窗）--------
    print("Computing Stage-1 GT velocity (1s window)...")
    gt_vel_stage1 = compute_stage1_gt_velocity(
        pos_body, fs=FS_TARGET, window_sec=GT_WINDOW_SEC
    )

    # -------- 10. 保存 --------
    np.save("imu_stage1.npy", imu_stage1)
    np.save("gt_vel_stage1.npy", gt_vel_stage1)

    print("\n[OK] Preprocessing v2 finished.")
    print(f"IMU Stage1 shape : {imu_stage1.shape}")
    print(f"GT  Stage1 shape : {gt_vel_stage1.shape}")
    print("Saved files:")
    print("  - imu_stage1.npy")
    print("  - gt_vel_stage1.npy")


if __name__ == "__main__":
    preprocess_waist_stage1_v2()
