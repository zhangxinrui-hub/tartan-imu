#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TartanIMUModel, load_checkpoint


def main():

    # ======================
    # 1. Load data
    # ======================
    print("\n=== Loading data ===")
    imu_data = np.load("imu_data.npy")   # (N,6)   preprocessed IMU
    gt_pos   = np.load("gt_pos.npy")     # (N,3)   ground truth positions

    print("IMU shape:", imu_data.shape)
    print("GT  shape:", gt_pos.shape)

    # prepare input tensor [B, C, T]
    imu_tensor = torch.from_numpy(imu_data).float()
    imu_tensor = imu_tensor.unsqueeze(0)     # (1, N, 6)
    imu_tensor = imu_tensor.transpose(1, 2) # (1, 6, N)


    # ======================
    # 2. Load model
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = TartanIMUModel()
    load_checkpoint(model, "checkpoint_24.pt", device=device)
    model.to(device)
    model.eval()


    # ======================
    # 3. Inference
    # ======================
    print("\n=== Running inference ===")
    imu_tensor = imu_tensor.to(device)

    with torch.no_grad():
        outputs = model(imu_tensor)


    # ======================
    # 4. Get velocity output
    # ======================
    # pick drone head
    if not isinstance(outputs, dict):
        raise RuntimeError("Model outputs not dict type")

    if "drone" not in outputs:
        key = list(outputs.keys())[0]
        print(f"[WARN] Using first head instead of 'drone': {key}")
        out2d, out3d, out_z = outputs[key]
    else:
        out2d, out3d, out_z = outputs["drone"]

    vel_raw = out3d.squeeze(0).cpu().numpy()

    print("Raw velocity shape:", vel_raw.shape)

    # ensure (N,3)
    if vel_raw.ndim != 2:
        raise RuntimeError("Unexpected velocity tensor ndim")

    # If channel-first -> transpose
    if vel_raw.shape[0] == 3:
        pred_vel = vel_raw.T
    else:
        pred_vel = vel_raw

    print("Final pred_vel shape:", pred_vel.shape)


    # ======================
    # 5. Align lengths
    # ======================
    N = min(len(pred_vel), len(gt_pos))
    pred_vel = pred_vel[:N]
    gt_pos   = gt_pos[:N]


    # ======================
    # 6. Correct dt calculation
    # ======================
    fs_imu  = 200.0
    imu_len = len(imu_data)

    total_time = imu_len / fs_imu
    dt_pred = total_time / len(pred_vel)

    print("\n=== Time alignment ===")
    print("IMU frames      :", imu_len)
    print("Pred frames     :", len(pred_vel))
    print("Total duration  :", total_time, "s")
    print("Predicted dt    :", dt_pred, "s")


    # ======================
    # 7. Integrate velocity
    # ======================
    pred_pos = np.cumsum(pred_vel, axis=0) * dt_pred

    # align starting point
    pred_pos += (gt_pos[0] - pred_pos[0])


    # ======================
    # 8. Plot
    # ======================
    print("\n=== Plotting ===")

    plt.figure(figsize=(10,5))

    plt.plot(gt_pos[:,0], gt_pos[:,1],
             "k--", linewidth=2, label="Ground Truth")

    plt.plot(pred_pos[:,0], pred_pos[:,1],
             "r-", linewidth=2, label="Stage-1 Pred")

    plt.axis("equal")
    plt.grid(True)

    plt.title("Top View Trajectory (Stage-1 Zero-shot)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("stage1_trajectory.png", dpi=200)
    plt.show()

    print("\nSaved: stage1_trajectory.png")


if __name__ == "__main__":
    main()
