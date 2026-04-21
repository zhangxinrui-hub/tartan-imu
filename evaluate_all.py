"""
evaluate_all.py
---------------
三平台统一基线评估：Car / Human / Drone
使用 checkpoint_28.pt（Base model），不依赖 GT 四元数以外的信息。

ATE 计算采用标准 RMSE，不使用任何尺度校正。
同时报告速度尺度比（gt_speed / pred_speed），用于诊断模型速度估计偏差。

运行方式：
  conda run -n tartan_gpu python evaluate_all.py
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# 使用 test/car/model.py（含 hidden state 支持，有 load_checkpoint 函数）
# 注意：只使用 TartanIMUModel（不启用 LoRA）和 load_checkpoint
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'test', 'car'))
from model import TartanIMUModel, load_checkpoint


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def compute_ate(gt_pos, pred_pos):
    """RMSE ATE（3D）。"""
    return float(np.sqrt(np.mean(np.linalg.norm(gt_pos - pred_pos, axis=1) ** 2)))


def compute_scale_factor(gt_pos, pred_vel, dt):
    """
    计算速度尺度比：GT 路程 / 预测路程。
    >1 表示模型低估了速度；<1 表示高估。
    仅用于诊断，评估表中不用此值修正 ATE。
    """
    gt_dist   = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    pred_dist = float(np.sum(np.linalg.norm(pred_vel[:-1], axis=1)) * dt)
    if pred_dist < 1e-6:
        return float('inf')
    return gt_dist / pred_dist


def run_model(imu_np, model, device, chunk_size=1000):
    """
    分块推理，返回模型原始输出字典（每个 head 是三元组）。
    imu_np: [T, 6] float32，已预处理
    返回: dict{head: (block1[T,2], block2[T,3], block1_z[T,1])}
    """
    T = len(imu_np)
    all_b1, all_b2, all_bz = [], [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, T, chunk_size):
            chunk = torch.FloatTensor(imu_np[i:i + chunk_size])  # [t, 6]
            chunk = chunk.unsqueeze(0).permute(0, 2, 1).to(device)  # [1, 6, t]
            outputs, _ = model(chunk)
            return outputs  # 只要第一次就能拿到结构，之后分块
    return None  # unreachable


def run_model_chunked(imu_np, model, device, head, chunk_size=1000):
    """
    分块推理并拼接指定 head 的输出。
    返回: block1 [M,2], block2 [M,3], block1_z [M,1]
    """
    T = len(imu_np)
    all_b1, all_b2, all_bz = [], [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, T, chunk_size):
            chunk = torch.FloatTensor(imu_np[i:i + chunk_size])
            chunk = chunk.unsqueeze(0).permute(0, 2, 1).to(device)
            result = model(chunk)
            # 兼容 forward 返回 (outputs, hidden) 或只返回 outputs
            outputs = result[0] if isinstance(result, tuple) else result
            b1 = outputs[head][0].cpu().numpy().squeeze()
            b2 = outputs[head][1].cpu().numpy().squeeze()
            bz = outputs[head][2].cpu().numpy().squeeze()
            if b1.ndim == 1: b1 = b1[:, None]
            if b2.ndim == 1: b2 = b2[:, None]
            if bz.ndim == 1: bz = bz[:, None]
            all_b1.append(b1)
            all_b2.append(b2)
            all_bz.append(bz)

    return (np.concatenate(all_b1, axis=0),
            np.concatenate(all_b2, axis=0),
            np.concatenate(all_bz, axis=0))


def integrate_trajectory(vel_world, dt, start_pos):
    M = len(vel_world)
    pos = np.zeros((M, 3))
    pos[0] = start_pos
    for k in range(1, M):
        pos[k] = pos[k - 1] + vel_world[k - 1] * dt
    return pos


def align_gt(gt_pos, gt_quat, output_len, total_len):
    """将 GT 均匀采样到模型输出长度。"""
    idxs = np.linspace(0, total_len - 1, output_len).astype(int)
    return gt_pos[idxs], gt_quat[idxs], idxs


# ─────────────────────────────────────────────────────────────
# 平台 1：Car
# ─────────────────────────────────────────────────────────────

def eval_car(model, device, data_path, ckpt_path):
    """
    前向速度来自 block1_z（经诊断确认的正确分量）。
    施加 Vy=0, Vz=0 的车辆约束。
    """
    data      = np.load(data_path)
    imu_raw   = data['retargetted_imu']
    gt_pos    = data['retargetted_pos']
    gt_quat   = data['retargetted_quat']
    ts        = data['retargetted_ts']
    N         = len(imu_raw)

    # 预处理：去重力 + 去 Bias + /9.81
    acc_raw, gyro_raw = imu_raw[:, :3], imu_raw[:, 3:]
    g_body = R.from_quat(gt_quat).inv().apply(np.array([0., 0., 9.81]))
    acc_net = acc_raw - g_body
    acc_net  -= np.mean(acc_net[:200], axis=0)
    gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    acc_net  /= 9.81
    imu = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

    b1, b2, bz = run_model_chunked(imu, model, device, head='car')
    M = len(bz)

    gt_pos_s, gt_quat_s, _ = align_gt(gt_pos, gt_quat, M, N)
    dt_eff = (ts[-1] - ts[0]) / M

    # 车辆约束：前向 = block1_z，侧向/垂直 = 0
    vel_body = np.zeros((M, 3))
    vel_body[:, 0] = bz[:, 0]
    vel_world = R.from_quat(gt_quat_s).apply(vel_body)

    pred_pos = integrate_trajectory(vel_world, dt_eff, gt_pos_s[0])
    ate      = compute_ate(gt_pos_s, pred_pos)
    gt_dist  = float(np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1)))
    scale    = compute_scale_factor(gt_pos_s, vel_body, dt_eff)

    return {
        'ate': ate, 'drift_pct': ate / gt_dist * 100,
        'scale': scale, 'path_len': gt_dist,
        'gt_pos': gt_pos_s, 'pred_pos': pred_pos,
    }


# ─────────────────────────────────────────────────────────────
# 平台 2：Human
# ─────────────────────────────────────────────────────────────

def eval_human(model, device, data_path, ckpt_path):
    """
    使用全 3D 速度 [block1[0], block1[1], block1_z]，无尺度校正。
    数据若非 200Hz 则先重采样。
    """
    data    = np.load(data_path)
    ts_raw  = np.squeeze(data['retargetted_ts'])
    imu_raw = data['retargetted_imu']
    gt_pos  = data['retargetted_pos']
    gt_quat = data['retargetted_quat']

    # 重采样至 200Hz（human 数据约 100Hz）
    duration = ts_raw[-1] - ts_raw[0]
    new_len  = int(duration * 200.0)
    new_ts   = np.linspace(ts_raw[0], ts_raw[-1], new_len)
    def resamp(arr):
        return interp1d(ts_raw, arr, axis=0, kind='linear',
                        fill_value='extrapolate')(new_ts)
    imu_raw  = resamp(imu_raw).astype(np.float32)
    gt_pos   = resamp(gt_pos)
    gt_quat  = resamp(gt_quat)
    gt_quat /= np.linalg.norm(gt_quat, axis=1, keepdims=True)
    N = len(imu_raw)

    # 预处理
    acc_raw, gyro_raw = imu_raw[:, :3], imu_raw[:, 3:]
    g_body = R.from_quat(gt_quat).inv().apply(np.array([0., 0., 9.81]))
    acc_net = acc_raw - g_body
    acc_net  -= np.mean(acc_net[:200], axis=0)
    gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    acc_net  /= 9.81
    imu = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

    b1, b2, bz = run_model_chunked(imu, model, device, head='human')
    M = len(bz)

    gt_pos_s, gt_quat_s, _ = align_gt(gt_pos, gt_quat, M, N)
    dt_eff = duration / M

    # 全 3D 速度
    vel_body  = np.hstack([b1, bz])          # [M, 3]
    vel_world = R.from_quat(gt_quat_s).apply(vel_body)
    # Z 方向去均值，防止高度漂移（与原 inference.py 一致）
    vel_world[:, 2] -= np.mean(vel_world[:, 2])

    pred_pos = integrate_trajectory(vel_world, dt_eff, gt_pos_s[0])
    ate      = compute_ate(gt_pos_s, pred_pos)
    gt_dist  = float(np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1)))
    scale    = compute_scale_factor(gt_pos_s, vel_body, dt_eff)

    return {
        'ate': ate, 'drift_pct': ate / gt_dist * 100,
        'scale': scale, 'path_len': gt_dist,
        'gt_pos': gt_pos_s, 'pred_pos': pred_pos,
    }


# ─────────────────────────────────────────────────────────────
# 平台 3：Drone
# ─────────────────────────────────────────────────────────────

def eval_drone(model, device, data_dir, ckpt_path):
    """
    数据已去重力（acc norm ≈ 5.2），仍需去 Bias + /9.81。
    使用全 3D 速度。
    """
    imu_raw  = np.load(os.path.join(data_dir, 'imu_data.npy')).astype(np.float32)
    gt_pos   = np.load(os.path.join(data_dir, 'gt_pos.npy'))
    gt_quat  = np.load(os.path.join(data_dir, 'gt_quat.npy'))
    N        = len(imu_raw)
    duration = N / 200.0   # 200Hz

    # 预处理（重力已去，只需去 Bias + /9.81）
    acc_raw, gyro_raw = imu_raw[:, :3].copy(), imu_raw[:, 3:].copy()
    acc_raw  -= np.mean(acc_raw[:200],  axis=0)
    gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    acc_raw  /= 9.81
    imu = np.concatenate([acc_raw, gyro_raw], axis=1).astype(np.float32)

    b1, b2, bz = run_model_chunked(imu, model, device, head='drone')
    M = len(bz)

    gt_pos_s, gt_quat_s, _ = align_gt(gt_pos, gt_quat, M, N)
    dt_eff = duration / M

    # 全 3D 速度
    vel_body  = np.hstack([b1, bz])          # [M, 3]
    vel_world = R.from_quat(gt_quat_s).apply(vel_body)

    pred_pos = integrate_trajectory(vel_world, dt_eff, gt_pos_s[0])
    ate      = compute_ate(gt_pos_s, pred_pos)
    gt_dist  = float(np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1)))
    scale    = compute_scale_factor(gt_pos_s, vel_body, dt_eff)

    return {
        'ate': ate, 'drift_pct': ate / gt_dist * 100,
        'scale': scale, 'path_len': gt_dist,
        'gt_pos': gt_pos_s, 'pred_pos': pred_pos,
    }


# ─────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────

def plot_results(results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, r) in zip(axes, results.items()):
        gp, pp = r['gt_pos'], r['pred_pos']
        ax.plot(gp[:, 0], gp[:, 1], 'k--', lw=2, label='Ground Truth')
        ax.plot(pp[:, 0], pp[:, 1], 'r-',  lw=2,
                label=f"Pred  ATE={r['ate']:.1f}m")
        ax.plot(gp[0, 0], gp[0, 1], 'go', ms=8, label='Start')
        ax.set_title(f"{name.upper()}  |  drift={r['drift_pct']:.1f}%"
                     f"  scale={r['scale']:.2f}",
                     fontsize=11)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.axis('equal'); ax.grid(True); ax.legend(fontsize=8)
    plt.suptitle('TartanIMU Baseline Evaluation (checkpoint_28, no scale correction)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = os.path.join(ROOT, 'checkpoint_28.pt')

    # 加载一次模型，三个平台共用
    model = TartanIMUModel().to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    platforms = {
        'car': lambda: eval_car(
            model, device,
            os.path.join(ROOT, 'test', 'car', 'pretrain_1.npz'),
            ckpt_path),
        'human': lambda: eval_human(
            model, device,
            os.path.join(ROOT, 'test', 'human', 'pretrain_1.npz'),
            ckpt_path),
        'drone': lambda: eval_drone(
            model, device,
            os.path.join(ROOT, 'Dataset_drone'),
            ckpt_path),
    }

    results = {}
    for name, fn in platforms.items():
        print(f"\n{'='*50}")
        print(f"  评估平台: {name.upper()}")
        print(f"{'='*50}")
        results[name] = fn()

    # ── 结果表格 ──────────────────────────────────────────────
    print('\n\n' + '=' * 65)
    print('  TartanIMU Baseline (checkpoint_28, no scale correction)')
    print('=' * 65)
    print(f"{'Platform':<10} {'ATE (m)':>10} {'Drift (%)':>12} "
          f"{'Path (m)':>10} {'Speed Scale':>12}")
    print('-' * 65)
    for name, r in results.items():
        print(f"{name:<10} {r['ate']:10.2f} {r['drift_pct']:12.2f} "
              f"{r['path_len']:10.1f} {r['scale']:12.2f}")
    print('=' * 65)
    print("Speed Scale > 1 → model under-predicts speed")
    print("Speed Scale < 1 → model over-predicts speed")

    # ── 绘图 ───────────────────────────────────────────────────
    save_path = os.path.join(ROOT, 'results', 'baseline_eval.png')
    plot_results(results, save_path)


if __name__ == '__main__':
    main()
