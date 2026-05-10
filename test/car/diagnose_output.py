import torch, numpy as np, sys
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '.')
from model import TartanIMUModel, load_checkpoint

data    = np.load('pretrain_1.npz')
imu_raw = data['retargetted_imu']
gt_quat = data['retargetted_quat']
gt_pos  = data['retargetted_pos']
ts      = data['retargetted_ts']

# 预处理
acc_raw  = imu_raw[:, :3];  gyro_raw = imu_raw[:, 3:]
g_world  = np.array([0., 0., 9.81])
r_obj    = R.from_quat(gt_quat)
g_body   = r_obj.inv().apply(g_world)
acc_net  = acc_raw - g_body
acc_net  -= np.mean(acc_net[:200], axis=0)
gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)
acc_net  /= 9.81
imu = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

device = 'cuda'
model = TartanIMUModel().to(device)
load_checkpoint(model, 'checkpoint_28.pt', device)
model.eval()

N = len(imu)
all_out = []
with torch.no_grad():
    for i in range(0, N, 1000):
        chunk = torch.FloatTensor(imu[i:i+1000]).unsqueeze(0).permute(0,2,1).to(device)
        out, _ = model(chunk)
        c = out['car']
        v0 = c[0].cpu().numpy().squeeze()  # [T, 2]
        v1 = c[1].cpu().numpy().squeeze()  # [T, 3]
        v2 = c[2].cpu().numpy().squeeze()  # [T, 1]
        if v0.ndim == 1: v0 = v0[:, None]
        if v1.ndim == 1: v1 = v1[:, None]
        if v2.ndim == 1: v2 = v2[:, None]
        all_out.append(np.hstack([v0, v1, v2]))  # [T, 6]
pred = np.concatenate(all_out, axis=0)  # [M, 6]

idxs = np.linspace(0, N-1, len(pred)).astype(int)
dt_gt = 0.005
gt_vel_w = np.diff(gt_pos, axis=0) / dt_gt
gt_vel_w = np.vstack([gt_vel_w, gt_vel_w[-1:]])
gt_vel_b = R.from_quat(gt_quat).inv().apply(gt_vel_w)
gt_vb = gt_vel_b[idxs]  # [M, 3]

cols = ['block1[0]', 'block1[1]', 'block2[0]', 'block2[1]', 'block2[2]', 'block1_z']
gt_names = ['GT_Vx(fwd)', 'GT_Vy(lat)', 'GT_Vz(up)']

print("相关性矩阵（模型各输出 vs GT body速度各分量）:")
print("{:12}  {:>12}  {:>12}  {:>12}".format("", *gt_names))
for ci, cname in enumerate(cols):
    corrs = [np.corrcoef(pred[:, ci], gt_vb[:, gi])[0, 1] for gi in range(3)]
    best = max(abs(c) for c in corrs)
    flag = "  <<<<" if best > 0.3 else ""
    print("{:12}  {:12.4f}  {:12.4f}  {:12.4f}{}".format(cname, *corrs, flag))

print()
print("各输出 mean / std:")
for ci, cname in enumerate(cols):
    print("  {:12}  mean={:8.4f}  std={:8.4f}".format(cname, pred[:, ci].mean(), pred[:, ci].std()))
print()
print("GT body速度 mean / std:")
for gi, gname in enumerate(gt_names):
    print("  {:12}  mean={:8.4f}  std={:8.4f}".format(gname, gt_vb[:, gi].mean(), gt_vb[:, gi].std()))

# ------- 也尝试无 /9.81 归一化的结果 -------
print()
print("=" * 60)
print("尝试：不做 /9.81 归一化")
acc_raw2  = imu_raw[:, :3]
gyro_raw2 = imu_raw[:, 3:]
g_body2   = R.from_quat(gt_quat).inv().apply(g_world)
acc_net2  = acc_raw2 - g_body2
acc_net2  -= np.mean(acc_net2[:200], axis=0)
gyro_raw2  = gyro_raw2 - np.mean(gyro_raw2[:200], axis=0)
# 不归一化
imu2 = np.concatenate([acc_net2, gyro_raw2], axis=1).astype(np.float32)

all_out2 = []
with torch.no_grad():
    for i in range(0, N, 1000):
        chunk = torch.FloatTensor(imu2[i:i+1000]).unsqueeze(0).permute(0,2,1).to(device)
        out, _ = model(chunk)
        c = out['car']
        v0 = c[0].cpu().numpy().squeeze()
        v1 = c[1].cpu().numpy().squeeze()
        v2 = c[2].cpu().numpy().squeeze()
        if v0.ndim == 1: v0 = v0[:, None]
        if v1.ndim == 1: v1 = v1[:, None]
        if v2.ndim == 1: v2 = v2[:, None]
        all_out2.append(np.hstack([v0, v1, v2]))
pred2 = np.concatenate(all_out2, axis=0)

print("{:12}  {:>12}  {:>12}  {:>12}".format("", *gt_names))
for ci, cname in enumerate(cols):
    corrs = [np.corrcoef(pred2[:, ci], gt_vb[:, gi])[0, 1] for gi in range(3)]
    best = max(abs(c) for c in corrs)
    flag = "  <<<<" if best > 0.3 else ""
    print("{:12}  {:12.4f}  {:12.4f}  {:12.4f}{}".format(cname, *corrs, flag))

print()
print("各输出 mean / std（无归一化）:")
for ci, cname in enumerate(cols):
    print("  {:12}  mean={:8.4f}  std={:8.4f}".format(cname, pred2[:, ci].mean(), pred2[:, ci].std()))
