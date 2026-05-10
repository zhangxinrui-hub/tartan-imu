"""
train_torch.py
--------------
PyTorch training script for the 6-DOF inertial odometry baseline (Lima et al., Sensors 2019).
Mirrors the Keras pipeline in train.py using model_torch.py definitions.

Usage:
    python train_torch.py oxiod my_torch_model
    python train_torch.py euroc my_torch_model --epochs 200
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model_torch import IMUOdometryNet, Lima2019MultiTaskLoss

# Re-use data loading utilities from dataset.py (numpy-quaternion based)
import quaternion  # noqa: F401  — needed by dataset.py internals
from dataset import load_oxiod_dataset, load_euroc_mav_dataset, load_dataset_6d_quat


def main():
    parser = argparse.ArgumentParser(description='Train 6-DOF IMU Odometry (PyTorch)')
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Training dataset')
    parser.add_argument('output', help='Output model name (without extension)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)

    window_size = 200
    stride = 10

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        pairs = [
            ('data5', 'imu3', 'vi3'), ('data2', 'imu1', 'vi1'), ('data2', 'imu2', 'vi2'),
            ('data5', 'imu2', 'vi2'), ('data3', 'imu4', 'vi4'), ('data4', 'imu4', 'vi4'),
            ('data4', 'imu2', 'vi2'), ('data1', 'imu7', 'vi7'), ('data5', 'imu4', 'vi4'),
            ('data4', 'imu5', 'vi5'), ('data1', 'imu3', 'vi3'), ('data3', 'imu2', 'vi2'),
            ('data2', 'imu3', 'vi3'), ('data1', 'imu1', 'vi1'), ('data3', 'imu3', 'vi3'),
            ('data3', 'imu5', 'vi5'), ('data1', 'imu4', 'vi4'),
        ]
        base = 'Oxford Inertial Odometry Dataset/handheld'
        for d, imu, vi in pairs:
            imu_data_filenames.append(f'{base}/{d}/syn/{imu}.csv')
            gt_data_filenames.append(f'{base}/{d}/syn/{vi}.csv')

    elif args.dataset == 'euroc':
        seqs = [
            ('MH_01_easy', ''), ('MH_03_medium', ''), ('MH_05_difficult', ''),
            ('V1_02_medium', ''), ('V2_01_easy', ''), ('V2_03_difficult', ''),
        ]
        for seq, _ in seqs:
            imu_data_filenames.append(f'{seq}/mav0/imu0/data.csv')
            gt_data_filenames.append(f'{seq}/mav0/state_groundtruth_estimate0/data.csv')

    # --- Load data ---
    x_gyro_all, x_acc_all = [], []
    y_dp_all, y_dq_all = [], []

    print("Loading data...")
    for imu_f, gt_f in zip(imu_data_filenames, gt_data_filenames):
        if args.dataset == 'oxiod':
            gyro, acc, pos, ori = load_oxiod_dataset(imu_f, gt_f)
        else:
            gyro, acc, pos, ori = load_euroc_mav_dataset(imu_f, gt_f)

        [xg, xa], [ydp, ydq], _, _ = load_dataset_6d_quat(gyro, acc, pos, ori, window_size, stride)
        x_gyro_all.append(xg)
        x_acc_all.append(xa)
        y_dp_all.append(ydp)
        y_dq_all.append(ydq)

    x_gyro = np.vstack(x_gyro_all)
    x_acc = np.vstack(x_acc_all)
    y_dp = np.vstack(y_dp_all)
    y_dq = np.vstack(y_dq_all)

    # Shuffle
    idx = np.random.permutation(len(x_gyro))
    x_gyro, x_acc, y_dp, y_dq = x_gyro[idx], x_acc[idx], y_dp[idx], y_dq[idx]
    print(f"Data loaded. Training samples: {len(x_gyro)}")

    # Train/val split (90/10)
    split = int(0.9 * len(x_gyro))
    train_ds = TensorDataset(
        torch.from_numpy(x_gyro[:split]).float(),
        torch.from_numpy(x_acc[:split]).float(),
        torch.from_numpy(y_dp[:split]).float(),
        torch.from_numpy(y_dq[:split]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_gyro[split:]).float(),
        torch.from_numpy(x_acc[split:]).float(),
        torch.from_numpy(y_dp[split:]).float(),
        torch.from_numpy(y_dq[split:]).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = IMUOdometryNet(sequence_length=window_size).to(device)
    criterion = Lima2019MultiTaskLoss().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print(f"Start training on {device} for {args.epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for bg, ba, bp, bq in train_loader:
            bg, ba = bg.to(device), ba.to(device)
            bp, bq = bp.to(device), bq.to(device)

            optimizer.zero_grad()
            pred_p, pred_q = model(bg, ba)
            loss, _, _ = criterion(pred_p, pred_q, bp, bq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bg, ba, bp, bq in val_loader:
                bg, ba = bg.to(device), ba.to(device)
                bp, bq = bp.to(device), bq.to(device)
                pred_p, pred_q = model(bg, ba)
                loss, _, _ = criterion(pred_p, pred_q, bp, bq)
                val_loss += loss.item()

        avg_val = val_loss / max(len(val_loader), 1)
        history['val_loss'].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'my_torch_model_checkpoint.pth')

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  best_val={best_val_loss:.4f}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save final model
    torch.save(model.state_dict(), f'{args.output}.pth')
    print(f"Final model saved to {args.output}.pth")
    print(f"Best checkpoint saved to my_torch_model_checkpoint.pth")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('6-DOF IMU Odometry Training Loss (PyTorch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('train_loss_torch.png', dpi=150)
    print("Loss curve saved to train_loss_torch.png")


if __name__ == '__main__':
    main()
