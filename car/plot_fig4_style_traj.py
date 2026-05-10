import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from model import TartanIMUModel, load_checkpoint

# ===================== 配置 =====================
IMU_FILE = "processed_car_data.npy"
GT_FILE  = "gt_large_loop.txt"
CKPT     = "checkpoint_28.pt"

FS = 200.0
DT = 1.0 / FS
WINDOW = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 模型 =====================
model = TartanIMUModel().to(device)
model = load_checkpoint(model, CKPT, device=device)
model.eval()

# ===================== 数据 =====================
imu = np.load(IMU_FILE)

col_names = [
    "UTCTime","Week","GPSTime","Latitude","Longitude","H_Ell",
    "X_ECEF","Y_ECEF","Z_ECEF",
    "VX_ECEF","VY_ECEF","VZ_ECEF",
    "VEast","VNorth","VUp",
    "VelBdyX","VelBdyY","VelBdyZ",
    "AccBdyX","AccBdyY","AccBdyZ",
    "Roll","Pitch","Heading","Q",
    "SDX_ECEF","SDY_ECEF","SDZ_ECEF",
    "SDEast","SDNorth","SDHeight",
    "HdngSD","SD_VN","SD_VE","SD_VH",
    "RollSD","PitchSD","AzStDev"
]
gt = np.genfromtxt(GT_FILE, skip_header=2, names=col_names)

# ===================== 姿态 =====================
roll  = np.deg2rad(gt["Roll"])
pitch = np.deg2rad(gt["Pitch"])
yaw   = np.deg2rad(gt["Heading"])
rot_wb = R.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))
quat_wb = rot_wb.as_quat()

# ===================== 连续积分 =====================
pos = np.zeros(3)
traj = [pos.copy()]

T = min(len(imu), len(quat_wb))

for t in range(0, T - WINDOW, WINDOW):
    imu_win = imu[t:t+WINDOW]

    imu_tensor = (
        torch.from_numpy(imu_win)
        .float()
        .T.unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out = model(imu_tensor)

    # 取整个 velocity 序列
    v_body_seq = out["car"][1][0].cpu().numpy()   # (L,3)

    for k in range(len(v_body_seq)):
        Rwb = R.from_quat(quat_wb[t+k]).as_matrix()
        v_world = Rwb @ v_body_seq[k]
        pos = pos + v_world * DT
        traj.append(pos.copy())

traj = np.array(traj)
x_pred = traj[:,0] - traj[0,0]
y_pred = traj[:,1] - traj[0,1]

# ===================== GT =====================
x_gt = gt["X_ECEF"] - gt["X_ECEF"][0]
y_gt = gt["Y_ECEF"] - gt["Y_ECEF"][0]

# ===================== 画图 =====================
plt.figure(figsize=(7,7))
plt.plot(x_gt, y_gt, "k--", linewidth=2, label="GT (large loop)")
plt.plot(x_pred, y_pred, "r-", linewidth=2, label="TartanIMU Stage 1 (continuous)")

plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Zero-shot trajectory (Fig.4-style, continuous)")
plt.legend()
plt.grid(True)
plt.show()
