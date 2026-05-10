import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from model import TartanIMUModel, load_checkpoint

# ============================================================
# Trajectory evaluation metrics
# ============================================================

def compute_ate(gt, pred):
    return np.mean(np.linalg.norm(gt - pred, axis=1))

def compute_trte(gt, pred, segment_len=5.0, dt=0.05):
    K = int(segment_len / dt)
    errors = []
    for i in range(len(gt) - K):
        dp_gt   = gt[i+K]   - gt[i]
        dp_pred = pred[i+K] - pred[i]
        errors.append(np.linalg.norm(dp_gt - dp_pred))
    return np.mean(errors)

# ============================================================
# Validation script
# ============================================================

def validate(model_file="checkpoint_24.pt"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n============================================")
    print("Running TartanIMU Stage-1 Evaluation")
    print("============================================")

    # --------------------------------------------------------
    # 1. Load preprocessed data
    # --------------------------------------------------------

    imu_data = np.load("imu_data.npy")     # [N,6] Body-FLU IMU
    gt_quat  = np.load("gt_quat.npy")      # [N,4] R_WB quaternion
    gt_pos   = np.load("gt_pos.npy")       # [N,3] World-FLU position

    print(f"Loaded IMU:      {imu_data.shape}")
    print(f"Loaded GT pos:   {gt_pos.shape}")
    print(f"Loaded GT quat:  {gt_quat.shape}")

    # --------------------------------------------------------
    # 2. Model inference
    # --------------------------------------------------------

    model = TartanIMUModel()
    load_checkpoint(model, model_file, device=device)
    model.to(device)
    model.eval()

    imu_tensor = torch.from_numpy(imu_data).float()
    imu_tensor = imu_tensor.unsqueeze(0).transpose(1,2).to(device)

    with torch.no_grad():
        outputs = model(imu_tensor)
        key = "drone" if "drone" in outputs else list(outputs.keys())[0]
        _, out3d, _ = outputs[key]

        vel_body = out3d.squeeze(0).cpu().numpy()

    if vel_body.shape[0] == 3:
        vel_body = vel_body.T

    print(f"Pred velocity shape: {vel_body.shape}")

    # --------------------------------------------------------
    # 3. Dummy covariance estimation Σ
    # --------------------------------------------------------

    sigma_sq = np.var(vel_body, axis=0)
    covariances = np.repeat(np.diag(sigma_sq)[None,:,:],
                             vel_body.shape[0],
                             axis=0)

    np.save("pred_vel_body.npy", vel_body)
    np.save("pred_covariance.npy", covariances)

    # --------------------------------------------------------
    # 4. Time alignment (MOST IMPORTANT FIX)
    # --------------------------------------------------------

    T = vel_body.shape[0]

    idx = np.linspace(
            0,
            len(gt_quat) - 1,
            T
          ).astype(int)

    gt_quat_down = gt_quat[idx]
    gt_pos_down  = gt_pos[idx]

    rots_wb = R.from_quat(gt_quat_down)

    # --------------------------------------------------------
    # 5. Rotate velocity to world frame
    # --------------------------------------------------------

    vel_world = rots_wb.apply(vel_body)

    # --------------------------------------------------------
    # 6. Integrate predicted velocity
    # --------------------------------------------------------

    imu_freq = 200.0
    ratio = imu_data.shape[0] / T
    dt = (1.0 / imu_freq) * ratio

    print(f"Velocity dt = {dt:.4f} s")

    pred_pos = np.cumsum(vel_world, axis=0) * dt

    pred_pos += gt_pos_down[0] - pred_pos[0]

    # --------------------------------------------------------
    # 7. Compute metrics
    # --------------------------------------------------------

    ate = compute_ate(gt_pos_down, pred_pos)

    trte = compute_trte(
                gt_pos_down,
                pred_pos,
                dt=dt)

    print("\n============================================")
    print("Evaluation Metrics")
    print("--------------------------------------------")
    print(f"ATE           : {ate:8.3f} m")
    print(f"T-RTE (5 sec) : {trte:8.3f} m")
    print("============================================")

    # --------------------------------------------------------
    # 8. Visualization
    # --------------------------------------------------------

    t_axis = np.arange(T) * dt

    fig = plt.figure(figsize=(14,6))

    # 3D Trajectory
    ax = fig.add_subplot(121, projection="3d")
    
    ax.plot(gt_pos_down[:,0], gt_pos_down[:,1], gt_pos_down[:,2],
            'k--', linewidth=2, label="GT")

    ax.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2],
            'r-',  linewidth=2, label="Pred")

    ax.set_title("Trajectory Comparison (World FLU)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    
    ax.legend()

    # Height plot
    ax2 = fig.add_subplot(122)

    ax2.plot(t_axis, gt_pos_down[:,2], 'k--', label="GT height")
    ax2.plot(t_axis, pred_pos[:,2], 'r-', label="Pred height")

    ax2.set_title("Height Profile Z(t)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Height [m]")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig("stage1_result.png", dpi=250)
    plt.close()

    # --------------------------------------------------------
    # 9. Final report
    # --------------------------------------------------------

    print("\n[OK] Finished successfully")
    print("\nSaved files:")
    print("  pred_vel_body.npy   -> predicted velocities v̂(t)")
    print("  pred_covariance.npy -> covariance Σ(t)")
    print("  stage1_result.png   -> trajectory comparison plot")
    print("============================================\n")


# ============================================================
if __name__ == "__main__":
    validate()
