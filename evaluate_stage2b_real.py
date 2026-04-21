"""
evaluate_stage2b_real.py
------------------------
Stage 2B: uncertainty-aware velocity EKF on real data.

Online path
───────────
1. Raw IMU -> orientation EKF (no GT quaternion in the online loop)
2. Estimated quaternion -> gravity removal for TartanIMU input
3. TartanIMU -> velocity v_hat + uncertainty proxy sigma_hat
4. Velocity-aided EKF -> position / velocity / attitude estimate

This script keeps GT only for offline evaluation (ATE/drift), not for the
online state estimation path itself.
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
from ekf_backend import (
    run_eskf_sequence,
    run_velocity_ekf_sequence,
    estimate_orientation_from_gravity,
)
from evaluate_real import (
    compute_ate,
    run_model_chunked,
    preprocess_imu_with_gravity,
    load_car_data,
    load_human_sequence,
    preprocess_human_real,
    load_drone_flight,
)


def resample_series(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr.copy()
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, target_len)
    return interp1d(x_old, arr, axis=0, kind='linear')(x_new)


def compute_path_len(gt_pos: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))


def compute_scale_factor(gt_pos: np.ndarray, vel_body: np.ndarray, dt: float) -> float:
    gt_dist = compute_path_len(gt_pos)
    pred_dist = float(np.sum(np.linalg.norm(vel_body[:-1], axis=1)) * dt)
    return gt_dist / max(pred_dist, 1e-6)


def human_align_raw(acc_raw: np.ndarray, gyr_raw: np.ndarray):
    acc_aligned = np.zeros_like(acc_raw)
    gyr_aligned = np.zeros_like(gyr_raw)

    acc_aligned[:, 0] = acc_raw[:, 2]
    acc_aligned[:, 1] = acc_raw[:, 1]
    acc_aligned[:, 2] = acc_raw[:, 0]

    gyr_aligned[:, 0] = gyr_raw[:, 2]
    gyr_aligned[:, 1] = gyr_raw[:, 1]
    gyr_aligned[:, 2] = gyr_raw[:, 0]
    return acc_aligned, gyr_aligned


def frd_to_flu(acc_raw: np.ndarray, gyr_raw: np.ndarray):
    acc = acc_raw.copy()
    gyr = gyr_raw.copy()
    acc[:, 1] *= -1
    acc[:, 2] *= -1
    gyr[:, 1] *= -1
    gyr[:, 2] *= -1
    return acc, gyr


def add_metrics(results: dict, gt_pos: np.ndarray, vel_body: np.ndarray, dt: float):
    path_len = compute_path_len(gt_pos)
    scale = compute_scale_factor(gt_pos, vel_body, dt)
    for item in results.values():
        item['drift_pct'] = item['ate'] / max(path_len, 1e-6) * 100.0
        item['path_len'] = path_len
        item['scale'] = scale
    return results


def predict_with_velocity_ekf(
    acc_raw: np.ndarray,
    gyro_raw: np.ndarray,
    vel_body: np.ndarray,
    sigma_proxy: np.ndarray,
    dt: float,
    axes,
    use_sigma: bool,
    q0=None,
    bg0=None,
):
    return run_velocity_ekf_sequence(
        acc_raw=acc_raw,
        gyro_raw=gyro_raw,
        vel_body_meas=vel_body,
        vel_sigma_proxy=sigma_proxy if use_sigma else None,
        dt=dt,
        q0=q0,
        bg0=bg0,
        axes=axes,
        fixed_std=None if use_sigma else 0.20,
        sigma_scale=0.15,
        sigma_floor=0.05,
        std_gyro=2e-3,
        std_acc=2e-2,
        std_bg=1e-5,
        std_ba=1e-4,
    )


def eval_car_stage2b(model, device):
    imu_csv = os.path.join(ROOT, 'car', 'car_imu_data_full.csv')
    gt_csv = os.path.join(ROOT, 'car', 'car_ground_truth.csv')
    imu_raw, gt_pos, gt_quat, ts = load_car_data(imu_csv, gt_csv)
    N = len(imu_raw)
    dt = float((ts[-1] - ts[0]) / (N - 1))

    acc_raw = imu_raw[:, :3].copy()
    gyro_raw = imu_raw[:, 3:].copy()

    # Stage 2A orientation estimate to remove gravity for model input
    q0 = estimate_orientation_from_gravity(acc_raw[:200])
    bg0 = np.mean(gyro_raw[:200], axis=0)
    quats_orient = run_eskf_sequence(
        acc_raw, gyro_raw, q0, bg0, dt,
        use_acc=True, std_gyro=2e-3, std_bg=1e-5, std_acc=0.8, acc_gate=0.25,
    )

    imu_pp = preprocess_imu_with_gravity(imu_raw, quats_orient, frd_to_flu=False)
    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='car')
    M = len(bz)
    dt_eff = float((ts[-1] - ts[0]) / M)

    acc_m = resample_series(acc_raw, M)
    gyro_m = resample_series(gyro_raw, M)
    gt_pos = resample_series(gt_pos, M)
    gt_pos = gt_pos - gt_pos[0]
    gt_quat_m = resample_series(gt_quat, M)
    gt_quat_m /= np.linalg.norm(gt_quat_m, axis=1, keepdims=True)

    vel_body = np.zeros((M, 3), dtype=np.float64)
    vel_body[:, 0] = bz[:, 0]
    sigma_proxy = b2

    results = {}

    vel_w = R.from_quat(gt_quat_m).apply(vel_body)
    pred_gt = np.zeros_like(gt_pos)
    for k in range(1, M):
        pred_gt[k] = pred_gt[k - 1] + vel_w[k - 1] * dt_eff
    results['GT-quat'] = {'ate': compute_ate(gt_pos, pred_gt), 'gt_pos': gt_pos, 'pred_pos': pred_gt}

    q0_m = estimate_orientation_from_gravity(acc_m[: min(200, M)])
    bg0_m = np.mean(gyro_m[: min(200, M)], axis=0)
    fixed = predict_with_velocity_ekf(acc_m, gyro_m, vel_body, sigma_proxy, dt_eff, axes=[0], use_sigma=False, q0=q0_m, bg0=bg0_m)
    sigma = predict_with_velocity_ekf(acc_m, gyro_m, vel_body, sigma_proxy, dt_eff, axes=[0], use_sigma=True, q0=q0_m, bg0=bg0_m)
    results['VelEKF-fixedR'] = {'ate': compute_ate(gt_pos, fixed['pos']), 'gt_pos': gt_pos, 'pred_pos': fixed['pos']}
    results['VelEKF+Sigma'] = {'ate': compute_ate(gt_pos, sigma['pos']), 'gt_pos': gt_pos, 'pred_pos': sigma['pos']}
    return add_metrics(results, gt_pos, vel_body, dt_eff)


def eval_human_stage2b(model, device):
    imu_xlsx = os.path.join(ROOT, 'human', '4d91_long_waist_imu.xlsx')
    gt_xlsx = os.path.join(ROOT, 'human', '4d91_long_ground_truth.xlsx')
    acc_raw_u, gyr_raw_u, gt_pos, ts = load_human_sequence(imu_xlsx, gt_xlsx)
    N = len(acc_raw_u)
    dt = 1.0 / 200.0

    acc_raw, gyro_raw = human_align_raw(acc_raw_u, gyr_raw_u)
    imu_pp, yaw_rate = preprocess_human_real(acc_raw_u, gyr_raw_u)
    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='human', chunk_size=2000, min_chunk=20)
    M = len(bz)
    dt_eff = float((ts[-1] - ts[0]) / M)

    acc_m = resample_series(acc_raw, M)
    gyro_m = resample_series(gyro_raw, M)
    yaw_rate_m = resample_series(yaw_rate[:, None], M)[:, 0]
    gt_pos = resample_series(gt_pos, M)
    gt_pos = gt_pos - gt_pos[0]

    vel_body = np.hstack([b1, bz])
    sigma_proxy = b2

    results = {}
    yaw = np.cumsum(yaw_rate_m * dt_eff)
    vel_w = np.zeros_like(vel_body)
    vel_w[:, 0] = vel_body[:, 0] * np.cos(yaw) - vel_body[:, 1] * np.sin(yaw)
    vel_w[:, 1] = vel_body[:, 0] * np.sin(yaw) + vel_body[:, 1] * np.cos(yaw)
    vel_w[:, 2] = vel_body[:, 2]
    pred = np.zeros_like(gt_pos)
    for k in range(1, M):
        pred[k] = pred[k - 1] + vel_w[k - 1] * dt_eff
    results['Gyro-yaw'] = {'ate': compute_ate(gt_pos, pred), 'gt_pos': gt_pos, 'pred_pos': pred}

    q0_m = estimate_orientation_from_gravity(acc_m[: min(200, M)])
    bg0_m = np.mean(gyro_m[: min(200, M)], axis=0)
    fixed = predict_with_velocity_ekf(acc_m, gyro_m, vel_body, sigma_proxy, dt_eff, axes=[0, 1, 2], use_sigma=False, q0=q0_m, bg0=bg0_m)
    sigma = predict_with_velocity_ekf(acc_m, gyro_m, vel_body, sigma_proxy, dt_eff, axes=[0, 1, 2], use_sigma=True, q0=q0_m, bg0=bg0_m)
    results['VelEKF-fixedR'] = {'ate': compute_ate(gt_pos, fixed['pos']), 'gt_pos': gt_pos, 'pred_pos': fixed['pos']}
    results['VelEKF+Sigma'] = {'ate': compute_ate(gt_pos, sigma['pos']), 'gt_pos': gt_pos, 'pred_pos': sigma['pos']}
    return add_metrics(results, gt_pos, vel_body, dt_eff)


def eval_drone_stage2b_flight(model, device, csv_path, flight_name):
    imu_raw, gt_pos, gt_quat, ts = load_drone_flight(csv_path)
    N = len(imu_raw)
    dt = 1.0 / 200.0

    acc_raw_frd = imu_raw[:, :3].copy()
    gyro_raw_frd = imu_raw[:, 3:].copy()
    acc_raw, gyro_raw = frd_to_flu(acc_raw_frd, gyro_raw_frd)
    imu_flu = np.concatenate([acc_raw, gyro_raw], axis=1).astype(np.float32)

    q0 = estimate_orientation_from_gravity(acc_raw[:200])
    bg0 = np.mean(gyro_raw[:200], axis=0)
    quats_orient = run_eskf_sequence(
        acc_raw, gyro_raw, q0, bg0, dt,
        use_acc=True, std_gyro=1e-3, std_bg=1e-5, std_acc=0.8, acc_gate=0.25,
    )

    imu_pp = preprocess_imu_with_gravity(imu_flu, quats_orient, frd_to_flu=False)
    b1, b2, bz = run_model_chunked(imu_pp, model, device, head='drone')
    M = len(bz)
    dt_eff = float((ts[-1] - ts[0]) / M)

    acc_m = resample_series(acc_raw, M)
    gyro_m = resample_series(gyro_raw, M)
    gt_pos = resample_series(gt_pos, M)
    gt_pos = gt_pos - gt_pos[0]
    gt_quat_m = resample_series(gt_quat, M)
    gt_quat_m /= np.linalg.norm(gt_quat_m, axis=1, keepdims=True)

    vel_body = np.hstack([b1, bz])
    sigma_proxy = b2

    results = {}
    vel_w = R.from_quat(gt_quat_m).apply(vel_body)
    pred_gt = np.zeros_like(gt_pos)
    for k in range(1, M):
        pred_gt[k] = pred_gt[k - 1] + vel_w[k - 1] * dt_eff
    results['GT-quat'] = {'ate': compute_ate(gt_pos, pred_gt), 'gt_pos': gt_pos, 'pred_pos': pred_gt}

    q0_m = estimate_orientation_from_gravity(acc_m[: min(200, M)])
    bg0_m = np.mean(gyro_m[: min(200, M)], axis=0)
    fixed = predict_with_velocity_ekf(acc_m, gyro_m, vel_body, sigma_proxy, dt_eff, axes=[0, 1, 2], use_sigma=False, q0=q0_m, bg0=bg0_m)
    sigma = predict_with_velocity_ekf(acc_m, gyro_m, vel_body, sigma_proxy, dt_eff, axes=[0, 1, 2], use_sigma=True, q0=q0_m, bg0=bg0_m)
    results['VelEKF-fixedR'] = {'ate': compute_ate(gt_pos, fixed['pos']), 'gt_pos': gt_pos, 'pred_pos': fixed['pos']}
    results['VelEKF+Sigma'] = {'ate': compute_ate(gt_pos, sigma['pos']), 'gt_pos': gt_pos, 'pred_pos': sigma['pos']}

    add_metrics(results, gt_pos, vel_body, dt_eff)
    for item in results.values():
        item['flight'] = flight_name
    return results


def eval_drone_stage2b_all(model, device):
    piloted_dir = os.path.join(ROOT, 'Drone_Dataset1', 'piloted')
    autono_dir = os.path.join(ROOT, 'Drone_Dataset1', 'autonomous')
    sequences = []
    for base, mode in [(piloted_dir, 'p'), (autono_dir, 'a')]:
        for flight_type in ['ellipse', 'lemniscate']:
            pattern = os.path.join(base, f'flight-*{mode}-{flight_type}', f'flight-*{mode}-{flight_type}_500hz_freq_sync.csv')
            for csv_path in sorted(glob.glob(pattern)):
                name = os.path.basename(os.path.dirname(csv_path))
                sequences.append((name, csv_path, flight_type, mode))

    grouped = {}
    for name, csv_path, flight_type, mode in sequences:
        print(f"  → {name}")
        result = eval_drone_stage2b_flight(model, device, csv_path, name)
        grouped.setdefault(f"{flight_type}-{mode}", []).append(result)
    return grouped


COLORS = {
    'GT-quat': ('#1a1a1a', '--', 2.2),
    'Gyro-yaw': ('#6a4c93', '-', 2.0),
    'VelEKF-fixedR': ('#e76f51', '-', 2.0),
    'VelEKF+Sigma': ('#2a9d8f', '-', 2.0),
}


def plot_results(car_results, human_results, drone_grouped, save_path):
    drone_sample = drone_grouped['ellipse-p'][0] if 'ellipse-p' in drone_grouped else next(iter(drone_grouped.values()))[0]
    panels = [('Car', car_results), ('Human waist', human_results), ('Drone sample', drone_sample)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, results) in zip(axes, panels):
        gt_pos = next(iter(results.values()))['gt_pos']
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='#2d6a4f', lw=2.5, linestyle='--', label='GT')
        ax.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', ms=8)
        for mode, item in results.items():
            color, ls, lw = COLORS.get(mode, ('#888888', '-', 1.5))
            ax.plot(item['pred_pos'][:, 0], item['pred_pos'][:, 1], color=color, ls=ls, lw=lw, label=f"{mode}  ATE={item['ate']:.1f}m")
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    plt.suptitle('Stage 2B – Velocity EKF with output_block2 uncertainty', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"图已保存: {save_path}")


def print_drone_grouped_table(grouped: dict):
    print(f"{'DroneGroup':<18} {'Mode':<16} {'ATE (m)':>12} {'Drift (%)':>14} {'Path (m)':>10}")
    print('-' * 78)
    for group_name, results_list in sorted(grouped.items()):
        for mode in ['GT-quat', 'VelEKF-fixedR', 'VelEKF+Sigma']:
            ates = [r[mode]['ate'] for r in results_list]
            drifts = [r[mode]['drift_pct'] for r in results_list]
            paths = [r[mode]['path_len'] for r in results_list]
            print(f"{group_name:<18} {mode:<16} {np.mean(ates):8.2f}±{np.std(ates):.1f} {np.mean(drifts):10.2f}±{np.std(drifts):.1f} {np.mean(paths):10.0f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = os.path.join(ROOT, 'checkpoint_28.pt')
    model = TartanIMUModel().to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    print('\n' + '=' * 70)
    print('  Stage 2B – Car real')
    print('=' * 70)
    car_results = eval_car_stage2b(model, device)

    print('\n' + '=' * 70)
    print('  Stage 2B – Human waist real')
    print('=' * 70)
    human_results = eval_human_stage2b(model, device)

    print('\n' + '=' * 70)
    print('  Stage 2B – Drone real multi-sequence')
    print('=' * 70)
    drone_grouped = eval_drone_stage2b_all(model, device)

    print('\n\n' + '=' * 92)
    print('  Stage 2B – uncertainty-aware velocity EKF (real data)')
    print('=' * 92)
    print(f"{'Platform':<10} {'Mode':<16} {'ATE (m)':>9} {'Drift (%)':>11} {'Path (m)':>10} {'Scale':>10}")
    print('-' * 92)
    for mode in ['GT-quat', 'VelEKF-fixedR', 'VelEKF+Sigma']:
        if mode in car_results:
            item = car_results[mode]
            print(f"{'Car':<10} {mode:<16} {item['ate']:9.2f} {item['drift_pct']:11.2f} {item['path_len']:10.0f} {item['scale']:10.2f}")
    for mode in ['Gyro-yaw', 'VelEKF-fixedR', 'VelEKF+Sigma']:
        item = human_results[mode]
        print(f"{'Human':<10} {mode:<16} {item['ate']:9.2f} {item['drift_pct']:11.2f} {item['path_len']:10.0f} {item['scale']:10.2f}")
    print('-' * 92)
    print_drone_grouped_table(drone_grouped)
    print('=' * 92)

    save_path = os.path.join(ROOT, 'results', 'stage2b_real.png')
    plot_results(car_results, human_results, drone_grouped, save_path)


if __name__ == '__main__':
    main()
