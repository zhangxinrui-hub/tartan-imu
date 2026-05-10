"""
train_lora.py
-------------
LoRA 微调 TartanIMU（车辆 head）：放在 car/ 目录，与自采数据同处，避免误以为在「test 数据」上训练。

依赖模型定义：test/car/model.py（LoRA 版 TartanIMUModel）。

数据与预处理（重要）
--------------------
- **应用原始 CSV + 与 evaluate_real / evaluate_all 相同的在线预处理**（load_car_csv → preprocess_imu_car）：
  去重力、前 200 帧 bias、acc÷9.81。这与 TartanIMU 推理输入分布一致。
- **不要**直接把 `processed_car_data_final.npy` 当训练输入：该脚本只做了 200Hz+去重力，
  **没有** bias 扣除和 **acc÷9.81**，与模型预训练/评估口径不一致。
  若要用 .npy，需先按上述规则补全预处理，或改脚本输出与 preprocess_imu_car 一致的 6 通道。

- 默认读本目录 car_imu_data_full.csv + car_ground_truth.csv。
- 作者 NPZ：`--source npz --npz ../test/car/pretrain_1.npz`

用法（推荐在项目根目录执行）:
  cd /path/to/tartan
  python car/train_lora.py --epochs 30 --device cuda

或:
  cd car && python train_lora.py

注意
----
- 训练目标是在 **随机短窗口** 上对 block1_z 做 MSE，与 **整条序列积分后的 ATE** 不是同一指标；
  有时 loss 下降但长轨迹略变差，属于正常现象。
- 请在同一管线下对比 Base / LoRA：`python test/car/run_eval_lora.py --compare`
- `--train-head`：**只更新** `heads.car`（车辆输出头），**不更新 LoRA**，适合「保骨干、只调车头」。
  默认（不加该标志）：**只训练 LoRA**，车头保持 checkpoint_28 初值。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

# 本文件在 tartan/car/；模型在 tartan/test/car/model.py
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
_MODEL_DIR = os.path.join(ROOT, "test", "car")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from model import TartanIMUModel


def load_car_csv(imu_csv: str, gt_csv: str, target_hz: float = 200.0):
    """与 evaluate_real.load_car_data 一致：自采 IMU+GPS → 200Hz。"""
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


def load_base_into_lora(model: nn.Module, base_ckpt_path: str, device: str) -> None:
    """将 Stage1 checkpoint 载入 LoRA 结构（预训练 Conv1d.weight → LoRA 包装层里的 conv.weight）。"""
    ckpt = torch.load(base_ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}

    mapped = {}
    for k, v in state.items():
        if ("input_block" in k or "residual_groups" in k) and "weight" in k and "bn" not in k and "lora" not in k:
            mapped[k.replace(".weight", ".conv.weight")] = v
        else:
            mapped[k] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    n_lora_missing = sum(1 for k in missing if "lora" in k)
    print(f"Base loaded. Missing keys: {len(missing)} (expect LoRA-only ~{n_lora_missing})")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def preprocess_imu_car(imu_raw: np.ndarray, gt_quat: np.ndarray, n_static: int = 200) -> np.ndarray:
    acc_raw, gyro_raw = imu_raw[:, :3].copy(), imu_raw[:, 3:].copy()
    g_body = R.from_quat(gt_quat).inv().apply(np.array([0.0, 0.0, 9.81]))
    acc_net = acc_raw - g_body
    acc_net -= np.mean(acc_net[:n_static], axis=0)
    gyro_raw = gyro_raw - np.mean(gyro_raw[:n_static], axis=0)
    acc_net /= 9.81
    return np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)


def compute_gt_forward_vx(gt_pos: np.ndarray, gt_quat: np.ndarray, ts: np.ndarray) -> np.ndarray:
    N = len(gt_pos)
    if N < 2:
        return np.zeros(N, dtype=np.float32)

    ts = np.asarray(ts).reshape(-1)
    v_w = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        if i == 0:
            dt = max(ts[1] - ts[0], 1e-6)
            v_w[i] = (gt_pos[1] - gt_pos[0]) / dt
        elif i == N - 1:
            dt = max(ts[-1] - ts[-2], 1e-6)
            v_w[i] = (gt_pos[-1] - gt_pos[-2]) / dt
        else:
            dt = max(ts[i + 1] - ts[i - 1], 1e-6)
            v_w[i] = (gt_pos[i + 1] - gt_pos[i - 1]) / dt

    rot = R.from_quat(gt_quat)
    v_b = rot.inv().apply(v_w)
    return v_b[:, 0].astype(np.float32)


@torch.no_grad()
def infer_output_length(model: nn.Module, L_in: int, device: torch.device) -> int:
    x = torch.zeros(1, 6, L_in, device=device)
    out, _ = model(x)
    return int(out["car"][0].shape[1])


def window_targets(gt_vx: np.ndarray, s: int, L_in: int, L_out: int, N: int) -> np.ndarray:
    tgt = np.empty(L_out, dtype=np.float32)
    for j in range(L_out):
        pos = s + (j + 0.5) * L_in / max(L_out, 1)
        idx = int(np.clip(pos, 0, N - 1))
        tgt[j] = gt_vx[idx]
    return tgt


def train_one_epoch(
    model: nn.Module,
    imu: np.ndarray,
    gt_vx: np.ndarray,
    N: int,
    L_in: int,
    L_out: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    supervise: str,
    train_head: bool,
    steps_per_epoch: int,
    batch_size: int,
) -> float:
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    total_loss = 0.0
    for _ in range(steps_per_epoch):
        batch_x = []
        batch_t = []
        for _ in range(batch_size):
            s = np.random.randint(0, max(1, N - L_in))
            w = imu[s : s + L_in]
            if w.shape[0] < L_in:
                continue
            batch_x.append(w)
            batch_t.append(window_targets(gt_vx, s, L_in, L_out, N))

        if len(batch_x) < batch_size:
            continue

        x = torch.from_numpy(np.stack(batch_x, axis=0)).float().to(device)
        x = x.transpose(1, 2)
        tgt = torch.from_numpy(np.stack(batch_t, axis=0)).float().to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs, _ = model(x)
        b1, _, bz = outputs["car"]

        if supervise == "block1_first":
            pred = b1[:, :, 0]
        elif supervise == "block1_z":
            pred = bz[:, :, 0]
        else:
            raise ValueError(supervise)

        loss = nn.functional.mse_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(steps_per_epoch, 1)


def save_lora_checkpoint(model: nn.Module, path: str, train_head: bool) -> None:
    sd = model.state_dict()
    out = {}
    if train_head:
        for k, v in sd.items():
            if k.startswith("heads.car."):
                out[k] = v
    else:
        for k, v in sd.items():
            if "lora_" in k:
                out[k] = v
    torch.save({"model_state_dict": out, "meta": {"train_head": train_head}}, path)
    mode = "heads.car only" if train_head else "LoRA only"
    print(f"Saved checkpoint ({len(out)} tensors, {mode}) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=str,
        choices=("car", "npz"),
        default="car",
        help="car=本目录自采 CSV；npz=作者 retarget（调试用）",
    )
    ap.add_argument(
        "--imu-csv",
        type=str,
        default=os.path.join(HERE, "car_imu_data_full.csv"),
        help="--source car 时的 IMU CSV",
    )
    ap.add_argument(
        "--gt-csv",
        type=str,
        default=os.path.join(HERE, "car_ground_truth.csv"),
        help="--source car 时的 GT CSV",
    )
    ap.add_argument(
        "--npz",
        type=str,
        default=os.path.join(ROOT, "test", "car", "pretrain_1.npz"),
        help="--source npz 时的路径（默认 test/car/pretrain_1.npz）",
    )
    ap.add_argument("--base-ckpt", type=str, default=os.path.join(ROOT, "checkpoint_28.pt"))
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "lora_trained.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument(
        "--supervise",
        type=str,
        default="block1_z",
        choices=("block1_first", "block1_z"),
    )
    ap.add_argument(
        "--train-head",
        action="store_true",
        help="只训练 heads.car；不加此标志则只训练 LoRA（二者互斥，勿同时训）",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Base checkpoint: {args.base_ckpt}")

    if args.source == "car":
        print("Training data: 本目录自采车辆 CSV (source=car)")
        print(f"  IMU: {args.imu_csv}")
        print(f"  GT:  {args.gt_csv}")
        if not os.path.isfile(args.imu_csv) or not os.path.isfile(args.gt_csv):
            raise SystemExit(
                "找不到 CSV。请确认文件在 car/ 下，或指定 --imu-csv / --gt-csv；"
                "调试可用: --source npz"
            )
        imu_raw, gt_pos, gt_quat, ts = load_car_csv(args.imu_csv, args.gt_csv)
    else:
        print(f"Training data: author NPZ — {args.npz}")
        if not os.path.isfile(args.npz):
            raise SystemExit(f"找不到 NPZ: {args.npz}")
        data = np.load(args.npz)
        imu_raw = data["retargetted_imu"]
        gt_pos = data["retargetted_pos"]
        gt_quat = data["retargetted_quat"]
        ts = np.squeeze(data["retargetted_ts"]).astype(np.float64)

    N = len(imu_raw)
    imu = preprocess_imu_car(imu_raw, gt_quat)
    gt_vx = compute_gt_forward_vx(gt_pos, gt_quat, ts)
    print(f"Sequence length N={N}, gt_vx range [{gt_vx.min():.3f}, {gt_vx.max():.3f}] m/s")

    model = TartanIMUModel(fine_tune_mode="lora").to(device)
    load_base_into_lora(model, args.base_ckpt, str(device))

    for _, p in model.named_parameters():
        p.requires_grad = False
    if args.train_head:
        for name, p in model.named_parameters():
            if name.startswith("heads.car."):
                p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "heads.car only" if args.train_head else "LoRA adapters only"
    print(f"Trainable parameters ({mode}): {n_train:,}")

    L_in = min(args.window, N - 2)
    if L_in < 64:
        raise SystemExit("Sequence too short for training window.")

    L_out = infer_output_length(model, L_in, device)
    print(f"Window L_in={L_in} -> output length L_out={L_out}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            imu,
            gt_vx,
            N,
            L_in,
            L_out,
            device,
            optimizer,
            args.supervise,
            args.train_head,
            args.steps_per_epoch,
            args.batch_size,
        )
        print(f"Epoch {ep:3d}/{args.epochs}  loss={loss:.6f}")

    save_lora_checkpoint(model, args.out, args.train_head)
    print(f"Done in {time.time() - t0:.1f}s")
    print("评估：在 test/car 下运行 run_eval_lora.py，将 LORA_CKPT 指向:")
    print(f"  {args.out}")


if __name__ == "__main__":
    main()
