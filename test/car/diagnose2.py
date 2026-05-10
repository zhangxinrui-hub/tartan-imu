"""
深度诊断：
1. GT速度在1Hz平滑后与模型输出的相关性
2. 用不同输出分量积分，看哪条轨迹最接近 GT
3. 测试 /9.81 归一化 vs 不归一化
"""
import torch, numpy as np, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '.')
from model import TartanIMUModel, load_checkpoint

data    = np.load('pretrain_1.npz')
imu_raw = data['retargetted_imu']
gt_quat = data['retargetted_quat']
gt_pos  = data['retargetted_pos']
ts      = data['retargetted_ts']
N = len(imu_raw)

def preprocess(imu_raw, gt_quat, normalize=True):
    acc_raw  = imu_raw[:, :3]
    gyro_raw = imu_raw[:, 3:]
    g_world  = np.array([0., 0., 9.81])
    g_body   = R.from_quat(gt_quat).inv().apply(g_world)
    acc_net  = acc_raw - g_body
    acc_net  -= np.mean(acc_net[:200], axis=0)
    gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    if normalize:
        acc_net /= 9.81
    return np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

def run_model(imu, device, ckpt='checkpoint_28.pt'):
    model = TartanIMUModel().to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()
    all_out = []
    with torch.no_grad():
        for i in range(0, len(imu), 1000):
            chunk = torch.FloatTensor(imu[i:i+1000]).unsqueeze(0).permute(0,2,1).to(device)
            out, _ = model(chunk)
            c = out['car']
            v0 = c[0].cpu().numpy().squeeze()
            v2 = c[2].cpu().numpy().squeeze()
            if v0.ndim == 1: v0 = v0[:, None]
            if v2.ndim == 1: v2 = v2[:, None]
            all_out.append(np.hstack([v0, v2]))   # [T, 3]: block1[0], block1[1], block1_z
    return np.concatenate(all_out, axis=0)

device = 'cuda'
# 对齐 GT
def get_gt_body_vel(idxs):
    dt_gt = 0.005
    gt_vel_w = np.diff(gt_pos, axis=0) / dt_gt
    gt_vel_w = np.vstack([gt_vel_w, gt_vel_w[-1:]])
    gt_vel_b = R.from_quat(gt_quat).inv().apply(gt_vel_w)
    return gt_vel_b[idxs]

# ======= 测试1: /9.81 归一化 =======
print("=== 测试1: 有 /9.81 归一化 ===")
imu_norm = preprocess(imu_raw, gt_quat, normalize=True)
pred_norm = run_model(imu_norm, device)
M = len(pred_norm)
idxs = np.linspace(0, N-1, M).astype(int)
gt_vb = get_gt_body_vel(idxs)

# 1Hz GT (每50步平均)
k50 = max(1, M // 50)
# 平滑GT
def smooth(x, w): return np.convolve(x, np.ones(w)/w, mode='same')
gt_vx_smooth = smooth(gt_vb[:, 0], 50)

for ci, cname in enumerate(['block1[0]', 'block1[1]', 'block1_z']):
    r_raw = np.corrcoef(pred_norm[:, ci], gt_vb[:, 0])[0, 1]
    r_sm  = np.corrcoef(pred_norm[:, ci], gt_vx_smooth)[0, 1]
    print(f"  {cname:12} corr_vs_GT_Vx: {r_raw:.4f}   corr_vs_GT_Vx_1Hz: {r_sm:.4f}   mean={pred_norm[:, ci].mean():.4f}")

# ======= 测试2: 无归一化 =======
print("\n=== 测试2: 无 /9.81 归一化 ===")
imu_no = preprocess(imu_raw, gt_quat, normalize=False)
pred_no = run_model(imu_no, device)
for ci, cname in enumerate(['block1[0]', 'block1[1]', 'block1_z']):
    r_raw = np.corrcoef(pred_no[:, ci], gt_vb[:, 0])[0, 1]
    r_sm  = np.corrcoef(pred_no[:, ci], gt_vx_smooth)[0, 1]
    print(f"  {cname:12} corr_vs_GT_Vx: {r_raw:.4f}   corr_vs_GT_Vx_1Hz: {r_sm:.4f}   mean={pred_no[:, ci].mean():.4f}")

# ======= 测试3: 各种积分方案 =======
print("\n=== 测试3: 积分对比（无归一化）===")
dt_eff = (ts[-1] - ts[0]) / M
gt_pos_s = gt_pos[idxs]
gt_quat_s = gt_quat[idxs]
rot = R.from_quat(gt_quat_s)

def integrate(fwd_vel, rot, dt):
    vel_b = np.zeros((len(fwd_vel), 3))
    vel_b[:, 0] = fwd_vel
    vel_w = rot.apply(vel_b)
    pos = np.zeros((len(fwd_vel), 3))
    pos[0] = gt_pos_s[0]
    for k in range(1, len(fwd_vel)):
        pos[k] = pos[k-1] + vel_w[k-1] * dt
    return pos

def ate(pred, gt):
    return np.sqrt(np.mean(np.linalg.norm(pred - gt, axis=1)**2))

schemes = [
    ('norm: block1[0]',   pred_norm[:, 0]),
    ('norm: block1[1]',   pred_norm[:, 1]),
    ('norm: block1_z',    pred_norm[:, 2]),
    ('nonorm: block1[0]', pred_no[:, 0]),
    ('nonorm: block1[1]', pred_no[:, 1]),
    ('nonorm: block1_z',  pred_no[:, 2]),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

gt_path = np.sum(np.linalg.norm(np.diff(gt_pos_s, axis=0), axis=1))
print(f"GT path length: {gt_path:.1f} m")

for idx, (name, fwd) in enumerate(schemes):
    pos = integrate(fwd, rot, dt_eff)
    a = ate(pos, gt_pos_s)
    print(f"  {name:25}  ATE={a:.2f}m  fwd_vel mean={fwd.mean():.4f} std={fwd.std():.4f}")
    axes[idx].plot(gt_pos_s[:, 0], gt_pos_s[:, 1], 'k--', lw=1.5, label='GT')
    axes[idx].plot(pos[:, 0], pos[:, 1], 'r-', lw=1.5, label=f'Pred ATE={a:.1f}m')
    axes[idx].set_title(name)
    axes[idx].axis('equal'); axes[idx].grid(True); axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.savefig('diagnose_traj.png', dpi=150)
print("\n图已保存: diagnose_traj.png")
