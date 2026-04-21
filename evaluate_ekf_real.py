"""
evaluate_ekf_real.py
--------------------
真实数据版本的 EKF 后端评估。

对比模式
────────
Car   : GT-quat / EKF / EKF+CalibBias
Human : Gyro-yaw baseline / EKF
Drone : GT-quat / EKF (gyro-only)

运行方式
────────
  conda run -n tartan_gpu python evaluate_ekf_real.py
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'test', 'car'))

from model import TartanIMUModel, load_checkpoint
from ekf_backend import run_eskf_sequence
from evaluate_real import (
    compute_ate,
    run_model_chunked,
    integrate_trajectory,
    preprocess_imu_with_gravity,
    align_to_model_output,
    load_car_data,
    load_human_sequence,
    preprocess_human_real,
    load_drone_flight,
)


def compute_scale_factor(gt_pos: np.ndarray, vel_body: np.ndarray, dt: float) -> float:
    gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    pred_dist = float(np.sum(np.linalg.norm(vel_body[:-1], axis=1)) * dt)
    if pred_dist < 1e-6:
        return float('inf')
    return gt_dist / pred_dist


def subsample_quats(quats_all: np.ndarray, M: int) -> R:
    idxs = np.linspace(0, len(quats_all) - 1, M).astype(int)
    return R.from_quat(quats_all[idxs])


def estimate_q0_from_gravity(acc_body: np.ndarray) -> R:
    """Use the initial mean acceleration to align body gravity with world +Z."""
    g_world = np.array([[0.0, 0.0, 9.81]])
    g_body = np.mean(acc_body, axis=0)
    if np.linalg.norm(g_body) < 1e-6:
        return R.identity()
    rot, _ = R.align_vectors(g_world, g_body.reshape(1, 3))
    return rot


def human_align_raw(acc_raw: np.ndarray, gyr_raw: np.ndarray):
    """Match human/inference.py axis mapping, but keep gravity for EKF."""
    acc_in = np.zeros_like(acc_raw)
    gyr_in = np.zeros_like(gyr_raw)

    acc_in[:, 0] = acc_raw[:, 2]
    acc_in[:, 1] = acc_raw[:, 1]
    acc_in[:, 2] = acc_raw[:, 0]

    gyr_in[:, 0] = gyr_raw[:, 2]
    gyr_in[:, 1] = gyr_raw[:, 1]
    gyr_in[:, 2] = gyr_raw[:, 0]
    return acc_in, gyr_in


def add_common_metrics(results: dict, gt_pos: np.ndarray, vel_body: np.ndarray, dt: float):
    path_len = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    scale = compute_scale_factor(gt_pos, vel_body, dt)
    for value in results.values():
        value['drift_pct'] = value['ate'] / max(path_len, 1e-6) * 100.0
        value['path_len'] = path_len
        value['scale'] = scale
    return results


def eval_car_real_ekf(model, device):
    imu_csv = os.path.join(ROOT, 'car', 'car_imu_data_full.csv')
    gt_csv = os.path.join(ROOT, 'car', 'car_ground_truth.csv')
    imu_raw, gt_pos, gt_quat, ts = load_car_data(imu_csv, gt_csv)
    N = len(imu_raw)
    dt = float((ts[-1] - ts[0]) / (N - 1))

    acc_raw = imu_raw[:, :3].copy()
    gyro_raw = imu_raw[:, 3:].copy()
    imu_pp = preprocess_imu_with_gravity(imu_raw, gt_quat, frd_to_flu=False)

    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='car')
    M = len(bz)
    gt_pos_s, gt_quat_s = align_to_model_output(gt_pos, gt_quat, N, M)
    gt_pos_s = gt_pos_s - gt_pos_s[0]
    dt_eff = (ts[-1] - ts[0]) / M

    vel_body = np.zeros((M, 3))
    vel_body[:, 0] = bz[:, 0]

    results = {}

    vel_w = R.from_quat(gt_quat_s).apply(vel_body)
    pred = integrate_trajectory(vel_w, dt_eff, np.zeros(3))
    results['GT-quat'] = {'ate': compute_ate(gt_pos_s, pred), 'gt_pos': gt_pos_s, 'pred_pos': pred}

    q0 = R.from_quat(gt_quat[0])
    bg0_local = np.mean(gyro_raw[:200], axis=0)
    bg0_global = np.mean(gyro_raw, axis=0)

    quats_ekf = run_eskf_sequence(
        acc_raw, gyro_raw, q0, bg0_local, dt,
        use_acc=True, std_gyro=2e-3, std_bg=1e-5, std_acc=0.8, acc_gate=0.25,
    )
    vel_w = subsample_quats(quats_ekf, M).apply(vel_body)
    pred = integrate_trajectory(vel_w, dt_eff, np.zeros(3))
    results['EKF'] = {'ate': compute_ate(gt_pos_s, pred), 'gt_pos': gt_pos_s, 'pred_pos': pred}

    quats_cal = run_eskf_sequence(
        acc_raw, gyro_raw, q0, bg0_global, dt,
        use_acc=True, std_gyro=2e-3, std_bg=1e-5, std_acc=0.8, acc_gate=0.25,
    )
    vel_w = subsample_quats(quats_cal, M).apply(vel_body)
    pred = integrate_trajectory(vel_w, dt_eff, np.zeros(3))
    results['EKF+CalibBias'] = {
        'ate': compute_ate(gt_pos_s, pred),
        'gt_pos': gt_pos_s,
        'pred_pos': pred,
        'bias_improve_deg_s': float((bg0_global[2] - bg0_local[2]) * 180.0 / np.pi),
    }
    return add_common_metrics(results, gt_pos_s, vel_body, dt_eff)


def eval_human_real_ekf(model, device):
    human_dir = os.path.join(ROOT, 'human')
    imu_xlsx = os.path.join(human_dir, '4d91_long_waist_imu.xlsx')
    gt_xlsx = os.path.join(human_dir, '4d91_long_ground_truth.xlsx')

    acc_raw, gyr_raw, gt_pos, ts = load_human_sequence(imu_xlsx, gt_xlsx)
    acc_align, gyr_align = human_align_raw(acc_raw, gyr_raw)
    imu_pp, yaw_rate = preprocess_human_real(acc_raw, gyr_raw)

    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='human', chunk_size=2000, min_chunk=20)
    vel_body = np.hstack([b1, bz])
    M = len(vel_body)
    dt = 1.0 / 200.0

    gt_pos_s = interp1d(np.linspace(0.0, 1.0, len(gt_pos)), gt_pos, axis=0)(
        np.linspace(0.0, 1.0, M)
    )
    gt_pos_s = gt_pos_s - gt_pos_s[0]

    results = {}

    yaw = np.cumsum(yaw_rate[:M] * dt)
    vel_w = np.zeros_like(vel_body)
    vel_w[:, 0] = vel_body[:, 0] * np.cos(yaw) - vel_body[:, 1] * np.sin(yaw)
    vel_w[:, 1] = vel_body[:, 0] * np.sin(yaw) + vel_body[:, 1] * np.cos(yaw)
    vel_w[:, 2] = vel_body[:, 2]
    pred = integrate_trajectory(vel_w, dt, np.zeros(3))
    results['Gyro-yaw'] = {'ate': compute_ate(gt_pos_s, pred), 'gt_pos': gt_pos_s, 'pred_pos': pred}

    q0 = estimate_q0_from_gravity(acc_align[:400])
    bg0 = np.mean(gyr_align[:200], axis=0)
    quats_ekf = run_eskf_sequence(
        acc_align, gyr_align, q0, bg0, dt,
        use_acc=True, std_gyro=2e-3, std_bg=1e-5, std_acc=0.8, acc_gate=0.35,
    )
    vel_w = subsample_quats(quats_ekf, M).apply(vel_body)
    pred = integrate_trajectory(vel_w, dt, np.zeros(3))
    results['EKF'] = {'ate': compute_ate(gt_pos_s, pred), 'gt_pos': gt_pos_s, 'pred_pos': pred}
    return add_common_metrics(results, gt_pos_s, vel_body, dt)


def eval_drone_real_flight(model, device, csv_path, flight_name):
    imu_raw, gt_pos, gt_quat, ts = load_drone_flight(csv_path)
    N = len(imu_raw)
    dt = 1.0 / 200.0
    duration = float(ts[-1] - ts[0])

    acc_raw = imu_raw[:, :3].copy()
    gyro_raw = imu_raw[:, 3:].copy()
    imu_pp = preprocess_imu_with_gravity(imu_raw, gt_quat, frd_to_flu=True)

    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='drone')
    M = len(bz)
    gt_pos_s, gt_quat_s = align_to_model_output(gt_pos, gt_quat, N, M)
    gt_pos_s = gt_pos_s - gt_pos_s[0]
    dt_eff = duration / M

    vel_body = np.hstack([b1, bz])
    results = {}

    vel_w = R.from_quat(gt_quat_s).apply(vel_body)
    pred = integrate_trajectory(vel_w, dt_eff, np.zeros(3))
    results['GT-quat'] = {'ate': compute_ate(gt_pos_s, pred), 'gt_pos': gt_pos_s, 'pred_pos': pred}

    q0 = R.from_quat(gt_quat[0])
    bg0 = np.mean(gyro_raw[:200], axis=0)
    quats_ekf = run_eskf_sequence(
        acc_raw, gyro_raw, q0, bg0, dt,
        use_acc=False, std_gyro=1e-3, std_bg=1e-5,
    )
    vel_w = subsample_quats(quats_ekf, M).apply(vel_body)
    pred = integrate_trajectory(vel_w, dt_eff, np.zeros(3))
    results['EKF (gyro)'] = {'ate': compute_ate(gt_pos_s, pred), 'gt_pos': gt_pos_s, 'pred_pos': pred}

    add_common_metrics(results, gt_pos_s, vel_body, dt_eff)
    for value in results.values():
        value['flight'] = flight_name
    return results


def eval_drone_real_all(model, device):
    piloted_dir = os.path.join(ROOT, 'Drone_Dataset1', 'piloted')
    autono_dir = os.path.join(ROOT, 'Drone_Dataset1', 'autonomous')

    sequences = []
    for base, mode in [(piloted_dir, 'p'), (autono_dir, 'a')]:
        for flight_type in ['ellipse', 'lemniscate']:
            pattern = os.path.join(
                base,
                f'flight-*{mode}-{flight_type}',
                f'flight-*{mode}-{flight_type}_500hz_freq_sync.csv',
            )
            for csv_path in sorted(glob.glob(pattern)):
                name = os.path.basename(os.path.dirname(csv_path))
                sequences.append((name, csv_path, flight_type, mode))

    grouped = {}
    for name, csv_path, flight_type, mode in sequences:
        print(f"  → {name}")
        result = eval_drone_real_flight(model, device, csv_path, name)
        key = f"{flight_type}-{mode}"
        grouped.setdefault(key, []).append(result)
    return grouped


COLORS = {
    'GT-quat': ('#1a1a1a', '--', 2.2),
    'EKF': ('#e63946', '-', 2.0),
    'EKF+CalibBias': ('#457b9d', '-', 2.0),
    'Gyro-yaw': ('#6a4c93', '-', 2.0),
    'EKF (gyro)': ('#e76f51', '-', 2.0),
}


def plot_ekf_real(car_results, human_results, drone_grouped, save_path):
    drone_sample = drone_grouped['ellipse-p'][0] if 'ellipse-p' in drone_grouped else next(iter(drone_grouped.values()))[0]
    panels = [
        ('Car (real)', car_results),
        ('Human (waist)', human_results),
        ('Drone sample', drone_sample),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, results) in zip(axes, panels):
        gt_pos = next(iter(results.values()))['gt_pos']
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='#2d6a4f', lw=2.5, linestyle='--', label='GT')
        ax.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', ms=8)
        for mode, item in results.items():
            color, ls, lw = COLORS.get(mode, ('#888888', '-', 1.5))
            label = f"{mode}  ATE={item['ate']:.1f}m ({item['drift_pct']:.1f}%)"
            ax.plot(item['pred_pos'][:, 0], item['pred_pos'][:, 1], color=color, ls=ls, lw=lw, label=label)
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)

    plt.suptitle('TartanIMU – Real Data EKF Comparison (checkpoint_28)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"图已保存: {save_path}")


def print_grouped_drone_table(drone_grouped: dict):
    print(f"{'DroneGroup':<22} {'Mode':<12} {'ATE (m)':>12} {'Drift (%)':>14} {'Path (m)':>10} {'Scale':>10}")
    print('-' * 86)
    for group_name, results_list in sorted(drone_grouped.items()):
        for mode in ['GT-quat', 'EKF (gyro)']:
            ates = [r[mode]['ate'] for r in results_list]
            drifts = [r[mode]['drift_pct'] for r in results_list]
            paths = [r[mode]['path_len'] for r in results_list]
            scales = [r[mode]['scale'] for r in results_list]
            print(
                f"{group_name:<22} {mode:<12} "
                f"{np.mean(ates):8.2f}±{np.std(ates):.1f} "
                f"{np.mean(drifts):10.2f}±{np.std(drifts):.1f} "
                f"{np.mean(paths):10.0f} {np.mean(scales):10.2f}"
            )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = os.path.join(ROOT, 'checkpoint_28.pt')

    model = TartanIMUModel().to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    print('\n' + '=' * 68)
    print('  平台 1：Car（真实道路数据，GT / EKF / EKF+CalibBias）')
    print('=' * 68)
    car_results = eval_car_real_ekf(model, device)

    print('\n' + '=' * 68)
    print('  平台 2：Human（waist，Gyro-yaw / EKF）')
    print('=' * 68)
    human_results = eval_human_real_ekf(model, device)

    print('\n' + '=' * 68)
    print('  平台 3：Drone（真实多序列，GT / gyro-only EKF）')
    print('=' * 68)
    drone_grouped = eval_drone_real_all(model, device)

    print('\n\n' + '=' * 92)
    print('  TartanIMU – Real Data EKF Ablation (checkpoint_28)')
    print('=' * 92)
    print(f"{'Platform':<10} {'Mode':<16} {'ATE (m)':>9} {'Drift (%)':>11} {'Path (m)':>10} {'Scale':>10}  Note")
    print('-' * 92)
    for mode in ['GT-quat', 'EKF', 'EKF+CalibBias']:
        if mode not in car_results:
            continue
        note = ''
        if mode == 'EKF+CalibBias':
            note = f"yaw-bias Δ≈{car_results[mode].get('bias_improve_deg_s', 0.0):.3f} deg/s"
        item = car_results[mode]
        print(f"{'Car':<10} {mode:<16} {item['ate']:9.2f} {item['drift_pct']:11.2f} {item['path_len']:10.0f} {item['scale']:10.2f}  {note}")

    for mode in ['Gyro-yaw', 'EKF']:
        item = human_results[mode]
        print(f"{'Human':<10} {mode:<16} {item['ate']:9.2f} {item['drift_pct']:11.2f} {item['path_len']:10.0f} {item['scale']:10.2f}")

    print('-' * 92)
    print_grouped_drone_table(drone_grouped)
    print('=' * 92)

    save_path = os.path.join(ROOT, 'results', 'ekf_real.png')
    plot_ekf_real(car_results, human_results, drone_grouped, save_path)


if __name__ == '__main__':
    main()
