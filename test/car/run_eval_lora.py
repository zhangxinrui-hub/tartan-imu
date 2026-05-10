"""
LoRA 评估：Base + LoRA 双 checkpoint，积分算 ATE。

说明
----
- 「test/car」只放 **模型代码**（model.py）和 **可选** 的作者 NPZ（pretrain_1.npz），
  **不是你的车采数据**。你的 CSV 在上一级目录的 **car/** 里。
- 默认在 **car/car_imu_data_full.csv + car_ground_truth.csv** 上评估（与 car/train_lora.py 一致）。
- 可在仓库根目录运行，无需 cd 到 test/car：

    python test/car/run_eval_lora.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAR = ROOT / "car"
DEFAULT_EVAL_OUT = str(CAR / "eval_result_lora.png")
DEFAULT_COMPARE_OUT = str(CAR / "eval_base_vs_lora.png")

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import TartanIMUModel, load_checkpoint, load_dual_checkpoints


def load_car_csv(imu_csv: str, gt_csv: str, target_hz: float = 200.0):
    """与 evaluate_real / car/train_lora 一致。"""
    df_imu = pd.read_csv(imu_csv)
    ts_imu = df_imu["timestamp"].values.astype(np.float64)
    acc = df_imu[["ax", "ay", "az"]].values
    gyro = df_imu[["gx", "gy", "gz"]].values
    quat = df_imu[["qx", "qy", "qz", "qw"]].values
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    df_gt = pd.read_csv(gt_csv)
    ts_gt = df_gt["timestamp"].values.astype(np.float64)
    pos_gt = df_gt[["x_gt", "y_gt", "z_gt"]].values

    t_start = max(ts_imu[0], ts_gt[0]) + 1.0
    t_end = min(ts_imu[-1], ts_gt[-1]) - 1.0
    target_ts = np.arange(t_start, t_end, 1.0 / target_hz)

    acc_200 = interp1d(ts_imu, acc, axis=0, kind="linear", fill_value="extrapolate")(target_ts)
    gyro_200 = interp1d(ts_imu, gyro, axis=0, kind="linear", fill_value="extrapolate")(target_ts)
    rots = R.from_quat(quat)
    slerp = Slerp(ts_imu, rots)
    quat_200 = slerp(target_ts).as_quat()
    pos_200 = interp1d(ts_gt, pos_gt, axis=0, kind="linear", fill_value="extrapolate")(target_ts)

    imu_200 = np.concatenate([acc_200, gyro_200], axis=1).astype(np.float32)
    return imu_200, pos_200, quat_200, target_ts


def preprocess_imu_car(imu_raw: np.ndarray, gt_quat: np.ndarray, n_static: int = 200) -> np.ndarray:
    acc_raw, gyro_raw = imu_raw[:, :3].copy(), imu_raw[:, 3:].copy()
    g_body = R.from_quat(gt_quat).inv().apply(np.array([0.0, 0.0, 9.81]))
    acc_net = acc_raw - g_body
    acc_net -= np.mean(acc_net[:n_static], axis=0)
    gyro_raw = gyro_raw - np.mean(gyro_raw[:n_static], axis=0)
    acc_net /= 9.81
    return np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)


def calculate_ate(gt_pos, pred_pos):
    errors = np.linalg.norm(gt_pos - pred_pos, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def integrate_trajectory(vel_world: np.ndarray, dt: float, start_pos: np.ndarray) -> np.ndarray:
    """与 evaluate_real.py 一致。"""
    m = len(vel_world)
    pos = np.zeros((m, 3), dtype=np.float64)
    pos[0] = start_pos
    for k in range(1, m):
        pos[k] = pos[k - 1] + vel_world[k - 1] * dt
    return pos


def infer_car_block1_z(
    model: torch.nn.Module,
    imu: np.ndarray,
    device: str,
    chunk_size: int,
    min_chunk: int,
) -> np.ndarray:
    """分块前向，拼接 car head 的 block1_z，形状 [M, 1]。"""
    all_vz = []
    with torch.no_grad():
        for i in range(0, len(imu), chunk_size):
            input_seq = imu[i : i + chunk_size]
            if len(input_seq) < min_chunk:
                continue
            inp = torch.FloatTensor(input_seq).unsqueeze(0).permute(0, 2, 1).to(device)
            outputs, _ = model(inp)
            out_z = outputs["car"][2]
            v_z_np = out_z.cpu().numpy()
            if v_z_np.ndim == 3:
                v_z_np = v_z_np[0]
            if v_z_np.ndim == 1:
                v_z_np = v_z_np[:, None]
            all_vz.append(v_z_np)
    return np.concatenate(all_vz, axis=0)


def metrics_from_vz(
    v_z_cat: np.ndarray,
    total_imu_len: int,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    timestamps: np.ndarray,
):
    """由 block1_z 得到轨迹指标（与 evaluate_real.eval_car_real 一致：起点对齐）。"""
    timestamps = np.squeeze(np.asarray(timestamps))
    output_len = len(v_z_cat)
    idxs = np.linspace(0, total_imu_len - 1, output_len).astype(int)
    gt_pos_s = gt_pos[idxs]
    gt_quat_s = gt_quat[idxs]

    vel_body = np.zeros((output_len, 3), dtype=np.float64)
    vel_body[:, 0] = v_z_cat[:, 0]
    vel_world = R.from_quat(gt_quat_s).apply(vel_body)
    dt_effective = (timestamps[-1] - timestamps[0]) / output_len

    gt_center = gt_pos_s - gt_pos_s[0]
    pred_pos = integrate_trajectory(vel_world, dt_effective, np.zeros(3))

    ate = calculate_ate(gt_center, pred_pos)
    traj_len = float(np.sum(np.linalg.norm(np.diff(gt_center, axis=0), axis=1)))
    drift_ratio = (ate / traj_len) * 100 if traj_len > 0 else 0.0
    pred_path = float(np.sum(np.linalg.norm(np.diff(pred_pos, axis=0), axis=1)))
    scale_path = traj_len / pred_path if pred_path > 1e-6 else float("inf")

    return {
        "ate": ate,
        "drift_ratio": drift_ratio,
        "scale_path": scale_path,
        "output_len": output_len,
        "dt_effective": dt_effective,
        "gt_center": gt_center,
        "pred_pos": pred_pos,
    }


def run_eval_dual(
    base_ckpt_path: str,
    lora_ckpt_path: str,
    imu: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    timestamps: np.ndarray,
    device="cuda",
    chunk_size=1000,
    min_chunk: int = 10,
    save_name="eval_result_lora.png",
    base_only: bool = False,
):
    if not os.path.exists(base_ckpt_path):
        print(f"ERROR: Base checkpoint not found: {base_ckpt_path}")
        return
    if not base_only and not os.path.exists(lora_ckpt_path):
        print(f"ERROR: LoRA checkpoint not found: {lora_ckpt_path}")
        return

    total_len = len(imu)
    timestamps = np.squeeze(np.asarray(timestamps))
    print(f"Sequence length: {total_len} frames")

    if base_only:
        print("Mode: Base only (no LoRA)")
        model = TartanIMUModel(fine_tune_mode=None).to(device)
        load_checkpoint(model, base_ckpt_path, device)
        label = "Base"
        title_extra = base_ckpt_path
    else:
        print("Mode: Base + LoRA")
        model = TartanIMUModel(fine_tune_mode="lora").to(device)
        model = load_dual_checkpoints(model, base_ckpt_path, lora_ckpt_path, device)
        label = "Base+LoRA"
        title_extra = lora_ckpt_path

    model.eval()

    # 车辆前向速度必须用 output_block1_z；分块与 evaluate_real 一致。
    print(f"Running inference... chunk_size={chunk_size}")
    v_z_cat = infer_car_block1_z(model, imu, device, chunk_size, min_chunk)
    stats = metrics_from_vz(v_z_cat, total_len, gt_pos, gt_quat, timestamps)

    ate = stats["ate"]
    drift_ratio = stats["drift_ratio"]
    scale_path = stats["scale_path"]
    output_len = stats["output_len"]
    dt_effective = stats["dt_effective"]
    gt_center = stats["gt_center"]
    pred_pos = stats["pred_pos"]

    print("=" * 40)
    print(f"{label}")
    print(f"  Output frames M={output_len}  dt={dt_effective*1000:.3f} ms")
    print(f"  ATE (RMSE): {ate:.4f} m")
    print(f"  Drift:      {drift_ratio:.2f}%")
    print(f"  Path scale (GT/Pred): {scale_path:.3f}")
    print("=" * 40)

    plt.figure(figsize=(10, 8))
    plt.plot(gt_center[:, 0], gt_center[:, 1], "k--", label="Ground Truth")
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], "r-", lw=2, label=f"{label} (ATE={ate:.2f}m)")
    plt.scatter(0, 0, c="g", marker="^", label="Start")
    plt.scatter(gt_center[-1, 0], gt_center[-1, 1], c="k", marker="x", label="GT End")
    plt.scatter(pred_pos[-1, 0], pred_pos[-1, 1], c="r", marker="x", label="Pred End")
    plt.title(f"TartanIMU {label}\n{title_extra}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.savefig(save_name, dpi=300)
    print(f"Saved: {save_name}")


def _finetune_label(ckpt_path: str) -> str:
    """从 car/train_lora.py 保存的 meta 推断第二路曲线名称。"""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    meta = ck.get("meta") or {}
    if meta.get("train_head"):
        return "Base + heads.car"
    return "Base + LoRA"


def run_compare(
    base_ckpt_path: str,
    lora_ckpt_path: str,
    imu: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    timestamps: np.ndarray,
    device: str,
    chunk_size: int,
    min_chunk: int,
    save_name: str,
):
    """同一条序列上对比 Base 与微调 checkpoint，避免「凭感觉」判断。"""
    if not os.path.exists(base_ckpt_path) or not os.path.exists(lora_ckpt_path):
        print("ERROR: need valid base and lora checkpoints")
        return

    ft_label = _finetune_label(lora_ckpt_path)
    total_len = len(imu)
    print(f"Sequence length: {total_len} frames")
    print(f"Running comparison... chunk_size={chunk_size}  fine-tune: {ft_label}")

    model_b = TartanIMUModel(fine_tune_mode=None).to(device)
    load_checkpoint(model_b, base_ckpt_path, device)
    model_b.eval()
    vz_b = infer_car_block1_z(model_b, imu, device, chunk_size, min_chunk)
    st_b = metrics_from_vz(vz_b, total_len, gt_pos, gt_quat, timestamps)
    del model_b
    if device == "cuda":
        torch.cuda.empty_cache()

    model_l = TartanIMUModel(fine_tune_mode="lora").to(device)
    load_dual_checkpoints(model_l, base_ckpt_path, lora_ckpt_path, device)
    model_l.eval()
    vz_l = infer_car_block1_z(model_l, imu, device, chunk_size, min_chunk)
    st_l = metrics_from_vz(vz_l, total_len, gt_pos, gt_quat, timestamps)

    print("=" * 44)
    print("Comparison (same pipeline, lower is better)")
    print(f"   Base only : ATE={st_b['ate']:.2f} m  drift={st_b['drift_ratio']:.2f}%  scale={st_b['scale_path']:.3f}")
    print(f"   {ft_label} : ATE={st_l['ate']:.2f} m  drift={st_l['drift_ratio']:.2f}%  scale={st_l['scale_path']:.3f}")
    if st_l["ate"] > st_b["ate"]:
        print(f"   → 微调后 ATE 高于 Base（{ft_label} 未降低 RMSE 轨迹误差）。")
    else:
        print("   → 微调相对 Base 有改善。")
    print("=" * 44)

    gc = st_b["gt_center"]
    plt.figure(figsize=(10, 8))
    plt.plot(gc[:, 0], gc[:, 1], "k--", lw=1.5, label="Ground Truth")
    plt.plot(st_b["pred_pos"][:, 0], st_b["pred_pos"][:, 1], "b-", lw=2, label=f"Base (ATE={st_b['ate']:.1f}m)")
    plt.plot(st_l["pred_pos"][:, 0], st_l["pred_pos"][:, 1], "r-", lw=2, label=f"{ft_label} (ATE={st_l['ate']:.1f}m)")
    plt.scatter(0, 0, c="g", marker="^", s=80, label="Start")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title(f"Base vs {ft_label} (same pipeline)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(save_name, dpi=300)
    print(f"Comparison plot saved: {save_name}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate Base+LoRA (default: your car CSV in car/)")
    ap.add_argument(
        "--source",
        choices=("car", "npz"),
        default="car",
        help="car=自采 CSV（默认）；npz=test/car 下作者 retarget 数据",
    )
    ap.add_argument("--base", type=str, default=str(ROOT / "checkpoint_28.pt"))
    ap.add_argument("--lora", type=str, default=str(CAR / "lora_trained.pt"))
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="只加载 Base checkpoint（无 LoRA），用于和微调结果公平对比",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="同一条数据上依次跑 Base 与 Base+LoRA，打印 ATE 并画在同一张图",
    )
    ap.add_argument("--imu-csv", type=str, default=str(CAR / "car_imu_data_full.csv"))
    ap.add_argument("--gt-csv", type=str, default=str(CAR / "car_ground_truth.csv"))
    ap.add_argument(
        "--npz",
        type=str,
        default=str(HERE / "pretrain_1.npz"),
        help="--source npz 时使用",
    )
    ap.add_argument("--out", type=str, default=DEFAULT_EVAL_OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="与 evaluate_real.py 一致；不同取值会改变总输出长度 M，结果不可与基线直接比",
    )
    args = ap.parse_args()

    if args.compare and args.base_only:
        raise SystemExit("不要同时使用 --compare 与 --base-only")

    if args.source == "car":
        print("数据: 你的车辆 CSV（car/），不是 test 里的 NPZ")
        imu_raw, gt_pos, gt_quat, ts = load_car_csv(args.imu_csv, args.gt_csv)
        imu = preprocess_imu_car(imu_raw, gt_quat)
    else:
        print(f"数据: 作者 NPZ — {args.npz}")
        data = np.load(args.npz)
        imu_raw = data["retargetted_imu"]
        gt_pos = data["retargetted_pos"]
        gt_quat = data["retargetted_quat"]
        ts = data["retargetted_ts"]
        acc_raw = imu_raw[:, :3]
        gyro_raw = imu_raw[:, 3:]
        g_body = R.from_quat(gt_quat).inv().apply(np.array([0.0, 0.0, 9.81]))
        acc_net = acc_raw - g_body
        acc_net -= np.mean(acc_net[:200], axis=0)
        gyro_raw = gyro_raw - np.mean(gyro_raw[:200], axis=0)
        acc_net /= 9.81
        imu = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

    if args.compare:
        out_path = (
            DEFAULT_COMPARE_OUT
            if os.path.abspath(args.out) == os.path.abspath(DEFAULT_EVAL_OUT)
            else args.out
        )
        run_compare(
            args.base,
            args.lora,
            imu,
            gt_pos,
            gt_quat,
            ts,
            device=args.device,
            chunk_size=args.chunk_size,
            min_chunk=10,
            save_name=out_path,
        )
    else:
        run_eval_dual(
            args.base,
            args.lora,
            imu,
            gt_pos,
            gt_quat,
            ts,
            device=args.device,
            chunk_size=args.chunk_size,
            save_name=args.out,
            base_only=args.base_only,
        )


if __name__ == "__main__":
    main()
