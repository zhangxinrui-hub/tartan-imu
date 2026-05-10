import numpy as np

imu = np.load("imu_stage1.npy")
gt  = np.load("gt_vel_stage1.npy")

print("IMU shape:", imu.shape)
print("GT  shape:", gt.shape)

print("\n[Check 2] acc_lin first 2s mean/std")
print("mean:", imu[:400, :3].mean(axis=0))
print("std :", imu[:400, :3].std(axis=0))

print("\n[Check 3] gyro norm (rad/s) first 2s")
print(np.linalg.norm(imu[:400, 3:], axis=1).mean())

print("\n[Check 4] acc / gyro norm stats")
acc_norm = np.linalg.norm(imu[:, :3], axis=1)
gyro_norm = np.linalg.norm(imu[:, 3:], axis=1)
print("acc mean/std:", acc_norm.mean(), acc_norm.std())
print("gyro mean/std:", gyro_norm.mean(), gyro_norm.std())

print("\n[Check 5] GT velocity norm stats")
gt_norm = np.linalg.norm(gt, axis=1)
print("mean:", gt_norm.mean())
print("percentile 50/90/99:", np.percentile(gt_norm, [50,90,99]))
