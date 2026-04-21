"""
evaluate_real.py
----------------
使用真实采集数据的三平台基线评估（替代 evaluate_all.py 的作者测试数据）

数据来源
────────
  Car   : car/car_imu_data_full.csv  +  car/car_ground_truth.csv
            - 480Hz IMU (acc+gyro+四元数),  100Hz GPS (xyz位置)
            - 真实道路驾驶: 4662m,  均速 7.63 m/s (≈27 km/h)
  Human : human/4d91_long_waist_imu.xlsx  +  human/4d91_long_ground_truth.xlsx
          human/4d91_long_instep_imu.xlsx +  human/4d91_long_ground_truth.xlsx
            - 约 306s，IMU 204.8Hz，GT 100Hz
            - 使用 human/ 下现有脚本同口径：轴映射 + 简单去重力 + gyro yaw 积分
  Drone : Drone_Dataset1/piloted/*_500hz_freq_sync.csv
            - 500Hz 已同步: IMU + mocap GT (位置 + 旋转矩阵)
            - 多条序列: 椭圆 (flight-01p~06p) + 双扭线 (flight-07p~12p)

运行方式
────────
  conda run -n tartan_gpu python evaluate_real.py
"""

import sys
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'test', 'car'))
from model import TartanIMUModel, load_checkpoint


# ─────────────────────────────────────────────────────────────
# 通用工具函数（与 evaluate_all.py 相同）
# ─────────────────────────────────────────────────────────────

def compute_ate(gt_pos, pred_pos):
    return float(np.sqrt(np.mean(np.linalg.norm(gt_pos - pred_pos, axis=1) ** 2)))


def run_model_chunked(imu_np, model, device, head, chunk_size=1000, min_chunk=10):
    all_b1, all_b2, all_bz = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imu_np), chunk_size):
            seg = imu_np[i:i + chunk_size]
            if len(seg) < min_chunk:   # skip tail if too short
                continue
            chunk = torch.FloatTensor(seg)
            chunk = chunk.unsqueeze(0).permute(0, 2, 1).to(device)
            result = model(chunk)
            outputs = result[0] if isinstance(result, tuple) else result
            b1 = outputs[head][0].cpu().numpy().squeeze()
            b2 = outputs[head][1].cpu().numpy().squeeze()
            bz = outputs[head][2].cpu().numpy().squeeze()
            # guarantee 2-D shape [T, C]
            if b1.ndim < 2: b1 = b1.reshape(-1, 2)
            if b2.ndim < 2: b2 = b2.reshape(-1, 3)
            if bz.ndim < 2: bz = bz.reshape(-1, 1)
            all_b1.append(b1); all_b2.append(b2); all_bz.append(bz)
    return (np.concatenate(all_b1, 0),
            np.concatenate(all_b2, 0),
            np.concatenate(all_bz, 0))


def integrate_trajectory(vel_world, dt, start_pos):
    M = len(vel_world)
    pos = np.zeros((M, 3))
    pos[0] = start_pos
    for k in range(1, M):
        pos[k] = pos[k - 1] + vel_world[k - 1] * dt
    return pos


def preprocess_imu_with_gravity(imu_raw, gt_quat, n_static=200, frd_to_flu=False):
    """
    标准预处理：去重力 + 去偏置 + /9.81

    imu_raw    : [N, 6]  (acc含重力, gyro)
    gt_quat    : [N, 4]  (x,y,z,w)  body→world 旋转 (FRD 或各平台约定)
    frd_to_flu : True 时在最后执行 FRD→FLU 转换（negate Y,Z）
                 这与 Dataset_drone/check_shapes.py 的 drone 预处理一致。

    流程（严格对应 check_shapes.py）：
      1. 计算 g_body = R.inv().apply([0,0,9.81])
      2. acc_bias  = mean(acc_raw[:n_static] - g_body[:n_static])
      3. acc_frd   = acc_raw - acc_bias
      4. acc_pure  = acc_frd - g_body
      5. gyro_bias = mean(gyro[:n_static]);  gyro -= gyro_bias
      6. [可选] FRD→FLU: negate Y,Z of both acc and gyro
      7. acc /= 9.81
    """
    acc_raw  = imu_raw[:, :3].copy()
    gyro_raw = imu_raw[:, 3:].copy()

    g_body    = R.from_quat(gt_quat).inv().apply(np.array([0., 0., 9.81]))
    acc_bias  = np.mean(acc_raw[:n_static] - g_body[:n_static], axis=0)
    acc_net   = acc_raw - acc_bias - g_body

    gyro_raw -= np.mean(gyro_raw[:n_static], axis=0)

    if frd_to_flu:
        acc_net[:, 1]  *= -1
        acc_net[:, 2]  *= -1
        gyro_raw[:, 1] *= -1
        gyro_raw[:, 2] *= -1

    acc_net /= 9.81
    return np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)


def align_to_model_output(gt_pos, gt_quat, N, M):
    """从 N 帧 GT 均匀采样到模型输出 M 帧，返回 gt_pos_s, gt_quat_s。"""
    idxs = np.linspace(0, N - 1, M).astype(int)
    return gt_pos[idxs], gt_quat[idxs]


# ─────────────────────────────────────────────────────────────
# 数据加载：Car（真实道路数据）
# ─────────────────────────────────────────────────────────────

def load_car_data(imu_csv, gt_csv, target_hz=200.0):
    """
    加载并同步车辆真实数据到 target_hz。

    IMU CSV : 480Hz，含 ax,ay,az,gx,gy,gz,qx,qy,qz,qw
    GT  CSV : 100Hz，含 x_gt,y_gt,z_gt
    返回 : (imu_200hz [N,6], gt_pos [N,3], gt_quat [N,4], ts [N])
    """
    print("  读取 Car IMU CSV …")
    df_imu = pd.read_csv(imu_csv)
    ts_imu = df_imu['timestamp'].values.astype(np.float64)
    acc    = df_imu[['ax', 'ay', 'az']].values
    gyro   = df_imu[['gx', 'gy', 'gz']].values
    quat   = df_imu[['qx', 'qy', 'qz', 'qw']].values
    # 归一化四元数（防止 slerp 出问题）
    norms  = np.linalg.norm(quat, axis=1, keepdims=True)
    quat  /= norms

    print("  读取 Car GT CSV …")
    df_gt  = pd.read_csv(gt_csv)
    ts_gt  = df_gt['timestamp'].values.astype(np.float64)
    pos_gt = df_gt[['x_gt', 'y_gt', 'z_gt']].values

    # 有效重叠时间范围（各留 1 秒缓冲）
    t_start = max(ts_imu[0], ts_gt[0]) + 1.0
    t_end   = min(ts_imu[-1], ts_gt[-1]) - 1.0
    target_ts = np.arange(t_start, t_end, 1.0 / target_hz)
    N = len(target_ts)
    print(f"  重叠时长: {t_end - t_start:.1f}s  →  {N} 帧 @ {target_hz:.0f}Hz")

    # 插值 IMU (linear)
    acc_200  = interp1d(ts_imu, acc,  axis=0, kind='linear',
                        fill_value='extrapolate')(target_ts)
    gyro_200 = interp1d(ts_imu, gyro, axis=0, kind='linear',
                        fill_value='extrapolate')(target_ts)

    # 四元数 slerp
    rots  = R.from_quat(quat)
    slerp = Slerp(ts_imu, rots)
    quat_200 = slerp(target_ts).as_quat()

    # 插值 GT 位置
    pos_200 = interp1d(ts_gt, pos_gt, axis=0, kind='linear',
                       fill_value='extrapolate')(target_ts)

    imu_200 = np.concatenate([acc_200, gyro_200], axis=1).astype(np.float32)
    return imu_200, pos_200, quat_200, target_ts


# ─────────────────────────────────────────────────────────────
# 数据加载：Drone（Drone_Dataset1 单条飞行）
# ─────────────────────────────────────────────────────────────

def load_drone_flight(csv_path, target_hz=200.0):
    """
    加载 Drone_Dataset1 一条飞行的同步 CSV (500Hz)。
    CSV 含: timestamp(μs), accel_x/y/z, gyro_x/y/z,
            drone_x/y/z, drone_rot[0]~[8] (旋转矩阵, body→world)
    返回 : (imu_200hz [N,6], gt_pos [N,3], gt_quat [N,4], ts [N])
    """
    df = pd.read_csv(csv_path)
    ts  = df['timestamp'].values.astype(np.float64) / 1e6  # μs → s

    acc  = df[['accel_x', 'accel_y', 'accel_z']].values
    gyro = df[['gyro_x',  'gyro_y',  'gyro_z' ]].values
    pos  = df[['drone_x', 'drone_y', 'drone_z' ]].values

    # 旋转矩阵 → 四元数
    rot_cols = [f'drone_rot[{i}]' for i in range(9)]
    rot_mat  = df[rot_cols].values.reshape(-1, 3, 3)
    # 保证正交（数值噪声可能破坏正交性）
    rots_raw = R.from_matrix(rot_mat)
    quat     = rots_raw.as_quat()   # [N, 4] x,y,z,w

    # 重采样 500Hz → target_hz（200Hz）
    # 留一帧缓冲避免浮点误差导致 Slerp 越界
    target_ts = np.arange(ts[0], ts[-1] - 1.0 / target_hz, 1.0 / target_hz)
    N = len(target_ts)

    acc_r  = interp1d(ts, acc,  axis=0, kind='linear')(target_ts)
    gyro_r = interp1d(ts, gyro, axis=0, kind='linear')(target_ts)
    pos_r  = interp1d(ts, pos,  axis=0, kind='linear')(target_ts)

    slerp    = Slerp(ts, rots_raw)
    quat_r   = slerp(target_ts).as_quat()

    imu_r = np.concatenate([acc_r, gyro_r], axis=1).astype(np.float32)
    return imu_r, pos_r, quat_r, target_ts


# ─────────────────────────────────────────────────────────────
# 平台 1：Car（真实道路）
# ─────────────────────────────────────────────────────────────

def eval_car_real(model, device):
    imu_csv = os.path.join(ROOT, 'car', 'car_imu_data_full.csv')
    gt_csv  = os.path.join(ROOT, 'car', 'car_ground_truth.csv')

    imu_raw, gt_pos, gt_quat, ts = load_car_data(imu_csv, gt_csv)
    N      = len(imu_raw)
    dt_eff_imu = float((ts[-1] - ts[0]) / (N - 1))

    # 预处理（车辆数据 NED/ENU 约定，不需要 FRD→FLU）
    imu_pp = preprocess_imu_with_gravity(imu_raw, gt_quat, frd_to_flu=False)

    # 推理
    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='car')
    M = len(bz)

    gt_pos_s, gt_quat_s = align_to_model_output(gt_pos, gt_quat, N, M)
    dt_eff = (ts[-1] - ts[0]) / M

    # 车辆约束：仅前向速度，侧向/垂直=0
    vel_body = np.zeros((M, 3))
    vel_body[:, 0] = bz[:, 0]
    vel_world = R.from_quat(gt_quat_s).apply(vel_body)

    # 起点归零
    gt_pos_s -= gt_pos_s[0]
    pred_pos  = integrate_trajectory(vel_world, dt_eff, np.zeros(3))

    path_len = float(np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1)))
    ate      = compute_ate(gt_pos_s, pred_pos)

    print(f"  Car  ATE={ate:.2f}m  drift={ate/path_len*100:.2f}%  path={path_len:.0f}m")
    return {
        'ate': ate, 'drift_pct': ate / path_len * 100, 'path_len': path_len,
        'gt_pos': gt_pos_s, 'pred_pos': pred_pos,
    }


# ─────────────────────────────────────────────────────────────
# 平台 2：Human（你自己的真实数据）
# ─────────────────────────────────────────────────────────────

def load_human_sequence(imu_xlsx, gt_xlsx, target_hz=200.0):
    df_imu = pd.read_excel(imu_xlsx)
    df_gt  = pd.read_excel(gt_xlsx)
    df_imu.columns = [str(c).strip() for c in df_imu.columns]
    df_gt.columns  = [str(c).strip() for c in df_gt.columns]

    ts_imu = df_imu['time after start [s]'].to_numpy(dtype=np.float64)
    ts_gt  = (df_gt['time_s'] if 'time_s' in df_gt.columns
              else df_gt['time after start [s]']).to_numpy(dtype=np.float64)

    acc = df_imu[['acc_x', 'acc_y', 'acc_z']].to_numpy(dtype=np.float64)
    gyr = df_imu[['gyr_x', 'gyr_y', 'gyr_z']].to_numpy(dtype=np.float64)
    gt_pos = df_gt[['gt_pos_global_x', 'gt_pos_global_y', 'gt_pos_global_z']].to_numpy(dtype=np.float64)

    # GT 文件以 km 存储，需要恢复到 m
    if np.max(np.abs(gt_pos)) < 10.0:
        gt_pos *= 1000.0

    if np.max(np.abs(gyr)) > 15.0:
        gyr = np.deg2rad(gyr)

    t_start = max(ts_imu[0], ts_gt[0])
    t_end   = min(ts_imu[-1], ts_gt[-1])
    target_ts = np.arange(t_start, t_end, 1.0 / target_hz)

    acc_r = interp1d(ts_imu, acc, axis=0, kind='linear')(target_ts)
    gyr_r = interp1d(ts_imu, gyr, axis=0, kind='linear')(target_ts)
    gt_r  = interp1d(ts_gt, gt_pos, axis=0, kind='linear')(target_ts)
    return acc_r, gyr_r, gt_r, target_ts


def preprocess_human_real(acc_raw, gyr_raw):
    """
    对齐 human/ 下 inference.py 的预处理：
      user Z -> model X (forward)
      user Y -> model Y (left)
      user X -> model Z (up)
    并采用简化去重力：仅从 model Z 减去 9.81。
    """
    acc_in = np.zeros_like(acc_raw)
    gyr_in = np.zeros_like(gyr_raw)

    acc_in[:, 0] = acc_raw[:, 2]
    acc_in[:, 1] = acc_raw[:, 1]
    acc_in[:, 2] = acc_raw[:, 0]

    gyr_in[:, 0] = gyr_raw[:, 2]
    gyr_in[:, 1] = gyr_raw[:, 1]
    gyr_in[:, 2] = gyr_raw[:, 0]

    acc_net = acc_in.copy()
    acc_net[:, 2] -= 9.81
    imu_pp = np.concatenate([acc_net, gyr_in], axis=1).astype(np.float32)
    return imu_pp, gyr_in[:, 2].copy()


def eval_human_sequence(model, device, imu_xlsx, gt_xlsx, name):
    acc_raw, gyr_raw, gt_pos, ts = load_human_sequence(imu_xlsx, gt_xlsx)
    imu_pp, yaw_rate = preprocess_human_real(acc_raw, gyr_raw)

    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='human', chunk_size=2000, min_chunk=20)
    vel_body = np.hstack([b1, bz])
    M = len(vel_body)

    gt_pos_s = interp1d(np.linspace(0.0, 1.0, len(gt_pos)), gt_pos, axis=0)(np.linspace(0.0, 1.0, M))
    yaw = np.cumsum(yaw_rate[:M] * (1.0 / 200.0))

    vel_world = np.zeros_like(vel_body)
    vel_world[:, 0] = vel_body[:, 0] * np.cos(yaw) - vel_body[:, 1] * np.sin(yaw)
    vel_world[:, 1] = vel_body[:, 0] * np.sin(yaw) + vel_body[:, 1] * np.cos(yaw)
    vel_world[:, 2] = vel_body[:, 2]

    gt_pos_s -= gt_pos_s[0]
    pred_pos = integrate_trajectory(vel_world, 1.0 / 200.0, np.zeros(3))

    path_len = float(np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1)))
    ate = compute_ate(gt_pos_s, pred_pos)
    print(f"    {name:<12} ATE={ate:.2f}m  drift={ate/path_len*100:.2f}%  path={path_len:.0f}m")
    return {
        'ate': ate, 'drift_pct': ate / path_len * 100, 'path_len': path_len,
        'gt_pos': gt_pos_s, 'pred_pos': pred_pos, 'sequence': name,
    }


def eval_human_real(model, device):
    human_dir = os.path.join(ROOT, 'human')
    gt_xlsx = os.path.join(human_dir, '4d91_long_ground_truth.xlsx')
    sequences = [
        ('waist',  os.path.join(human_dir, '4d91_long_waist_imu.xlsx')),
        ('instep', os.path.join(human_dir, '4d91_long_instep_imu.xlsx')),
    ]

    results = {}
    for name, imu_xlsx in sequences:
        print(f"  → {name}")
        results[name] = eval_human_sequence(model, device, imu_xlsx, gt_xlsx, name)
    return results


# ─────────────────────────────────────────────────────────────
# 平台 3：Drone（多条序列）
# ─────────────────────────────────────────────────────────────

def eval_drone_flight(model, device, csv_path, flight_name):
    imu_raw, gt_pos, gt_quat, ts = load_drone_flight(csv_path)
    N      = len(imu_raw)
    duration = float(ts[-1] - ts[0])

    # Drone 数据为 FRD 体系，需要转换到 FLU（与 Dataset_drone/check_shapes.py 一致）
    imu_pp = preprocess_imu_with_gravity(imu_raw, gt_quat, frd_to_flu=True)

    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='drone')
    M = len(bz)

    gt_pos_s, gt_quat_s = align_to_model_output(gt_pos, gt_quat, N, M)
    dt_eff = duration / M

    vel_body  = np.hstack([b1, bz])
    vel_world = R.from_quat(gt_quat_s).apply(vel_body)

    gt_pos_s -= gt_pos_s[0]
    pred_pos  = integrate_trajectory(vel_world, dt_eff, np.zeros(3))

    path_len = float(np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1)))
    ate      = compute_ate(gt_pos_s, pred_pos)
    print(f"    {flight_name:<28} ATE={ate:.2f}m  drift={ate/path_len*100:.2f}%  path={path_len:.0f}m")
    return {
        'ate': ate, 'drift_pct': ate / path_len * 100, 'path_len': path_len,
        'gt_pos': gt_pos_s, 'pred_pos': pred_pos,
        'flight': flight_name,
    }


def eval_drone_all(model, device):
    """评估 Drone_Dataset1 中选定的飞行序列，按飞行类型分组报告。"""
    piloted_dir = os.path.join(ROOT, 'Drone_Dataset1', 'piloted')
    autono_dir  = os.path.join(ROOT, 'Drone_Dataset1', 'autonomous')

    # 选取代表性序列：各类型全部 piloted + 全部 autonomous
    sequences = []
    for base, mode in [(piloted_dir, 'p'), (autono_dir, 'a')]:
        for flight_type in ['ellipse', 'lemniscate']:
            pattern = os.path.join(base, f'flight-*{mode}-{flight_type}',
                                   f'flight-*{mode}-{flight_type}_500hz_freq_sync.csv')
            for csv_path in sorted(glob.glob(pattern)):
                name = os.path.basename(os.path.dirname(csv_path))
                sequences.append((name, csv_path, flight_type, mode))

    results_by_type = {}
    for name, csv_path, ftype, mode in sequences:
        print(f"  → {name}")
        try:
            r = eval_drone_flight(model, device, csv_path, name)
            key = f"{ftype}-{mode}"
            if key not in results_by_type:
                results_by_type[key] = []
            results_by_type[key].append(r)
        except Exception as e:
            print(f"    [跳过] {e}")

    return results_by_type


# ─────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────

def plot_results(car_r, human_r, drone_results, save_path):
    # 收集 drone 的第一条序列用于可视化示例
    drone_sample = None
    for key, rlist in drone_results.items():
        if rlist:
            drone_sample = rlist[0]
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, r) in zip(axes, [('Car (real)', car_r),
                                     ('Human (waist)', human_r),
                                     ('Drone (sample)', drone_sample)]):
        if r is None:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title(name); continue
        gp = r['gt_pos'];  pp = r['pred_pos']
        ax.plot(gp[:, 0], gp[:, 1], 'k--', lw=2, label='GT')
        ax.plot(pp[:, 0], pp[:, 1], 'r-',  lw=2,
                label=f"Pred  ATE={r['ate']:.1f}m")
        ax.plot(gp[0, 0], gp[0, 1], 'go', ms=8)
        ax.set_title(f"{name}  drift={r['drift_pct']:.1f}%", fontsize=11)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.axis('equal'); ax.grid(True); ax.legend(fontsize=8)
    plt.suptitle('TartanIMU Baseline — Real Data  (checkpoint_28, no scale correction)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = os.path.join(ROOT, 'checkpoint_28.pt')

    model = TartanIMUModel().to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    print('\n' + '='*65)
    print('  平台 1：Car（你自己的真实道路数据）')
    print('='*65)
    car_r = eval_car_real(model, device)

    print('\n' + '='*65)
    print('  平台 2：Human（你自己的真实数据）')
    print('='*65)
    human_results = eval_human_real(model, device)
    human_r = human_results['waist']

    print('\n' + '='*65)
    print('  平台 3：Drone（Drone_Dataset1，多条序列）')
    print('='*65)
    drone_results = eval_drone_all(model, device)

    # ── 打印汇总表 ──────────────────────────────────────────
    print('\n\n' + '=' * 72)
    print('  TartanIMU Baseline  (checkpoint_28, no scale correction)')
    print('  真实数据版本')
    print('=' * 72)
    print(f"{'Platform':<10} {'Sequence':<26} {'ATE (m)':>9} "
          f"{'Drift (%)':>11} {'Path (m)':>10}")
    print('-' * 72)

    print(f"{'Car':<10} {'real road (1 seq)':<26} {car_r['ate']:9.2f} "
          f"{car_r['drift_pct']:11.2f} {car_r['path_len']:10.0f}")

    for seq_name in ['waist', 'instep']:
        r = human_results[seq_name]
        print(f"{'Human':<10} {f'{seq_name} (real)':<26} {r['ate']:9.2f} "
              f"{r['drift_pct']:11.2f} {r['path_len']:10.0f}")

    for ftype_mode, rlist in sorted(drone_results.items()):
        ates   = [r['ate']        for r in rlist]
        drifts = [r['drift_pct']  for r in rlist]
        paths  = [r['path_len']   for r in rlist]
        label  = f"drone-{ftype_mode} ({len(rlist)} seq)"
        print(f"{'Drone':<10} {label:<26} "
              f"{np.mean(ates):9.2f}±{np.std(ates):.1f}  "
              f"{np.mean(drifts):9.2f}±{np.std(drifts):.1f}  "
              f"{np.mean(paths):10.0f}")

    print('=' * 72)

    # ── 绘图 ───────────────────────────────────────────────
    save_path = os.path.join(ROOT, 'results', 'baseline_real.png')
    plot_results(car_r, human_r, drone_results, save_path)


if __name__ == '__main__':
    main()
