import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 配置
# =========================
DATA_PATH = "pretrain_1.npz"   # 改成你的 npz
SAVE_FIG  = "gt_only.png"

def visualize_gt_only():
    print(f"Loading GT from: {DATA_PATH}")
    data = np.load(DATA_PATH)

    # -------- 必须只用 GT --------
    gt_pos = data["retargetted_pos"]      # (N, 3)
    gt_ts  = data["retargetted_ts"]    
    gt_ts = np.asarray(gt_ts).squeeze()
    assert gt_ts.ndim == 1, f"gt_ts should be 1D, but got shape {gt_ts.shape}"

    print("=" * 40)
    print("GT sanity info:")
    print(f"  Frames: {len(gt_pos)}")
    print(f"  Time span: {gt_ts[0]:.3f} → {gt_ts[-1]:.3f}  (Δt ≈ {np.mean(np.diff(gt_ts)):.4f}s)")
    print(f"  XYZ range:")
    print(f"    X: {gt_pos[:,0].min():.2f} ~ {gt_pos[:,0].max():.2f}")
    print(f"    Y: {gt_pos[:,1].min():.2f} ~ {gt_pos[:,1].max():.2f}")
    print(f"    Z: {gt_pos[:,2].min():.2f} ~ {gt_pos[:,2].max():.2f}")
    print("=" * 40)

    # =========================
    # 1⃣ XY 平面轨迹（最重要）
    # =========================
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k-', linewidth=1)
    plt.scatter(gt_pos[0, 0], gt_pos[0, 1], c='g', s=60, label="Start")
    plt.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='r', s=60, label="End")
    plt.title("GT Trajectory (XY Plane)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # =========================
    # 2⃣ Z 随时间（看上下是否合理）
    # =========================
    plt.subplot(1, 2, 2)
    plt.plot(gt_ts - gt_ts[0], gt_pos[:, 2], 'k-')
    plt.title("GT Height (Z vs Time)")
    plt.xlabel("Time (s)")
    plt.ylabel("Z (m)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(SAVE_FIG, dpi=150)
    print(f"[OK] GT-only figure saved to: {SAVE_FIG}")
    plt.show()

    # =========================
    # 3⃣ 可选：3D 轨迹（辅助理解）
    # =========================
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_pos[:,0], gt_pos[:,1], gt_pos[:,2], 'k-')
    ax.scatter(gt_pos[0,0], gt_pos[0,1], gt_pos[0,2], c='g', s=50)
    ax.scatter(gt_pos[-1,0], gt_pos[-1,1], gt_pos[-1,2], c='r', s=50)
    ax.set_title("GT Trajectory (3D)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1,1,0.5])  # 防止 Z 被拉太夸张
    plt.show()


if __name__ == "__main__":
    visualize_gt_only()
