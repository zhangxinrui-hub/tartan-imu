"""
精细测试：平滑 + 正确输出分量 + 找最佳超参
"""
import torch, numpy as np, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, '.')
from model import TartanIMUModel, load_checkpoint

data    = np.load('pretrain_1.npz')
imu_raw = data['retargetted_imu']
gt_quat = data['retargetted_quat']
gt_pos  = data['retargetted_pos']
ts      = data['retargetted_ts']
N = len(imu_raw)

device = 'cuda'
model = TartanIMUModel().to(device)
load_checkpoint(model, 'checkpoint_28.pt', device)
model.eval()

def preprocess(normalize=True):
    acc_raw  = imu_raw[:, :3];  gyro_raw = imu_raw[:, 3:]
    g_world  = np.array([0., 0., 9.81])
    g_body   = R.from_quat(gt_quat).inv().apply(g_world)
    acc_net  = acc_raw - g_body
    acc_net  -= np.mean(acc_net[:200], axis=0)
    gr = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    if normalize: acc_net /= 9.81
    return np.concatenate([acc_net, gr], axis=1).astype(np.float32)

def run_model(imu):
    all_out = []
    with torch.no_grad():
        for i in range(0, len(imu), 1000):
            chunk = torch.FloatTensor(imu[i:i+1000]).unsqueeze(0).permute(0,2,1).to(device)
            out, _ = model(chunk)
            c = out['car']
            v0 = c[0].cpu().numpy().squeeze()  # block1: [T,2]
            v2 = c[2].cpu().numpy().squeeze()  # block1_z: [T,1]
            if v0.ndim==1: v0=v0[:,None]
            if v2.ndim==1: v2=v2[:,None]
            all_out.append(np.hstack([v0, v2]))
    return np.concatenate(all_out, axis=0)

imu_norm   = preprocess(normalize=True)
imu_nonorm = preprocess(normalize=False)
pred_n  = run_model(imu_norm)
pred_nn = run_model(imu_nonorm)

M = len(pred_n)
idxs = np.linspace(0, N-1, M).astype(int)
dt_eff = (ts[-1] - ts[0]) / M
gt_pos_s  = gt_pos[idxs]
gt_quat_s = gt_quat[idxs]
rot = R.from_quat(gt_quat_s)

def ate(pred):
    return np.sqrt(np.mean(np.linalg.norm(pred - gt_pos_s, axis=1)**2))

def integrate(fwd_vel, rot, dt):
    vb = np.zeros((M, 3)); vb[:, 0] = fwd_vel
    vw = rot.apply(vb)
    pos = np.zeros((M, 3)); pos[0] = gt_pos_s[0]
    for k in range(1, M):
        pos[k] = pos[k-1] + vw[k-1] * dt
    return pos

# 候选分量
candidates = {
    'norm_b1z':    pred_n[:, 2],
    'nonorm_b1[1]': pred_nn[:, 1],
}

# 平滑窗口从 1 到 200
windows = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]

print("平滑窗口 vs ATE (m):")
print(f"{'window':>8}  {'norm_b1z':>12}  {'nonorm_b1[1]':>14}")
best_ever = 1e9; best_cfg = None

rows = []
for w in windows:
    row = [w]
    for cname, v_raw in candidates.items():
        v_sm = uniform_filter1d(v_raw, size=w)
        pos = integrate(v_sm, rot, dt_eff)
        a = ate(pos)
        row.append(a)
        if a < best_ever:
            best_ever = a; best_cfg = (cname, w, v_sm, pos)
    rows.append(row)
    print(f"{w:8d}  {row[1]:12.2f}  {row[2]:14.2f}")

print(f"\nBest: {best_cfg[0]} window={best_cfg[1]} ATE={best_ever:.2f}m")

# 比较原始（w=1）vs 最优平滑
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
idx = 0
for cname, v_raw in candidates.items():
    for w in [1, 50, 100]:
        v_sm = uniform_filter1d(v_raw, size=w)
        pos = integrate(v_sm, rot, dt_eff)
        a = ate(pos)
        ax = axes[idx]
        ax.plot(gt_pos_s[:, 0], gt_pos_s[:, 1], 'k--', lw=1.5, label='GT')
        ax.plot(pos[:, 0], pos[:, 1], 'r-', lw=1.5, label=f'ATE={a:.1f}m')
        ax.set_title(f"{cname} win={w}", fontsize=9)
        ax.axis('equal'); ax.grid(True); ax.legend(fontsize=8)
        idx += 1

plt.tight_layout()
plt.savefig('diagnose4_smooth.png', dpi=150)
print("图已保存: diagnose4_smooth.png")

# GT 速度与最优预测的时序对比
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
gt_vel_w = np.diff(gt_pos, axis=0) / 0.005
gt_vel_w = np.vstack([gt_vel_w, gt_vel_w[-1:]])
gt_vel_b = R.from_quat(gt_quat).inv().apply(gt_vel_w)
gt_vx = gt_vel_b[idxs, 0]

t_out = np.linspace(0, ts[-1]-ts[0], M)
axes2[0].plot(t_out[:3000], gt_vx[:3000], 'k-', lw=1, label='GT Vx_body', alpha=0.7)
axes2[0].plot(t_out[:3000], pred_n[:3000, 2], 'b-', lw=0.8, label='norm_b1z raw', alpha=0.7)
v_sm = uniform_filter1d(pred_n[:, 2], size=50)
axes2[0].plot(t_out[:3000], v_sm[:3000], 'r-', lw=1.5, label='norm_b1z win50')
axes2[0].set_ylabel('Vx_fwd (m/s)'); axes2[0].legend(); axes2[0].grid(True)

axes2[1].plot(t_out[:3000], gt_vx[:3000], 'k-', lw=1, label='GT Vx_body', alpha=0.7)
axes2[1].plot(t_out[:3000], pred_nn[:3000, 1], 'b-', lw=0.8, label='nonorm_b1[1] raw', alpha=0.7)
v_sm2 = uniform_filter1d(pred_nn[:, 1], size=50)
axes2[1].plot(t_out[:3000], v_sm2[:3000], 'r-', lw=1.5, label='nonorm_b1[1] win50')
axes2[1].set_ylabel('Vfwd (m/s)'); axes2[1].legend(); axes2[1].grid(True)

axes2[2].plot(t_out, gt_vx, 'k-', lw=0.8, label='GT Vx_body', alpha=0.7)
axes2[2].plot(t_out, v_sm, 'r-', lw=1, label='norm_b1z win50')
axes2[2].set_xlabel('Time (s)'); axes2[2].set_ylabel('Full seq'); axes2[2].legend(); axes2[2].grid(True)

plt.tight_layout()
plt.savefig('diagnose4_timeseries.png', dpi=150)
print("时序图已保存: diagnose4_timeseries.png")
