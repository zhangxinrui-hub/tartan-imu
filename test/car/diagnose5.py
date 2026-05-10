"""
测试：
1. 尺度校正（scale b1_z）
2. checkpoint_24 vs checkpoint_28
3. 不同 chunk_size
"""
import torch, numpy as np, sys, os, matplotlib
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

def preprocess(normalize=True):
    acc_raw  = imu_raw[:, :3];  gyro_raw = imu_raw[:, 3:]
    g_world  = np.array([0., 0., 9.81])
    g_body   = R.from_quat(gt_quat).inv().apply(g_world)
    acc_net  = acc_raw - g_body
    acc_net  -= np.mean(acc_net[:200], axis=0)
    gr = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    if normalize: acc_net /= 9.81
    return np.concatenate([acc_net, gr], axis=1).astype(np.float32)

def run_model(imu, ckpt, chunk_size=1000):
    model = TartanIMUModel().to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()
    all_out = []
    with torch.no_grad():
        for i in range(0, len(imu), chunk_size):
            chunk = torch.FloatTensor(imu[i:i+chunk_size]).unsqueeze(0).permute(0,2,1).to(device)
            out, _ = model(chunk)
            c = out['car']
            v0 = c[0].cpu().numpy().squeeze()  # block1[0,1]
            v2 = c[2].cpu().numpy().squeeze()  # block1_z
            if v0.ndim==1: v0=v0[:,None]
            if v2.ndim==1: v2=v2[:,None]
            all_out.append(np.hstack([v0, v2]))
    return np.concatenate(all_out, axis=0)

imu_norm = preprocess(normalize=True)
imu_nonorm = preprocess(normalize=False)

def get_aligned(pred):
    M = len(pred)
    idxs = np.linspace(0, N-1, M).astype(int)
    dt_eff = (ts[-1] - ts[0]) / M
    return idxs, dt_eff, gt_pos[idxs], R.from_quat(gt_quat[idxs])

def integrate_and_ate(fwd_vel, gt_pos_s, rot, dt_eff):
    M = len(fwd_vel)
    vb = np.zeros((M, 3)); vb[:, 0] = fwd_vel
    vw = rot.apply(vb)
    pos = np.zeros((M, 3)); pos[0] = gt_pos_s[0]
    for k in range(1, M): pos[k] = pos[k-1] + vw[k-1] * dt_eff
    ate = np.sqrt(np.mean(np.linalg.norm(pos - gt_pos_s, axis=1)**2))
    return pos, ate

print("=" * 70)
print("测试1: checkpoint_24 vs checkpoint_28 (norm, b1_z)")
print("=" * 70)
ckpts = ['checkpoint_28.pt']
if os.path.exists('../../checkpoint_24.pt'):
    ckpts.append('../../checkpoint_24.pt')
elif os.path.exists('../../../checkpoint_24.pt'):
    ckpts.append('../../../checkpoint_24.pt')

results = {}
for ckpt in ckpts:
    pred = run_model(imu_norm, ckpt)
    idxs, dt_eff, gt_pos_s, rot = get_aligned(pred)
    fwd = pred[:, 2]   # b1_z as forward
    pos, ate = integrate_and_ate(fwd, gt_pos_s, rot, dt_eff)
    name = os.path.basename(ckpt)
    print(f"  {name} b1_z (norm): ATE={ate:.2f}m  mean_vel={fwd.mean():.4f}")
    results[name] = (pred, idxs, dt_eff, gt_pos_s, rot)

print()
print("=" * 70)
print("测试2: chunk_size 对比 (checkpoint_28, norm, b1_z)")
print("=" * 70)
pred28 = run_model(imu_norm, 'checkpoint_28.pt', chunk_size=1000)
for cs in [200, 500, 1000, 2000]:
    pred = run_model(imu_norm, 'checkpoint_28.pt', chunk_size=cs)
    idxs, dt_eff, gps, rot = get_aligned(pred)
    fwd = pred[:, 2]
    pos, ate = integrate_and_ate(fwd, gps, rot, dt_eff)
    print(f"  chunk_size={cs:5d}  n_out={len(pred):6d}  ATE={ate:.2f}m  mean_vel={fwd.mean():.4f}")

print()
print("=" * 70)
print("测试3: 尺度校正 (multiply b1_z by scale factor)")
print("=" * 70)
pred = run_model(imu_norm, 'checkpoint_28.pt')
idxs, dt_eff, gps, rot = get_aligned(pred)
fwd_raw = pred[:, 2]

gt_vel_w = np.diff(gt_pos, axis=0) / 0.005
gt_vel_w = np.vstack([gt_vel_w, gt_vel_w[-1:]])
gt_vx_body = R.from_quat(gt_quat).inv().apply(gt_vel_w)[:, 0]
gt_vx_body_s = gt_vx_body[idxs]

# 求最优 scale (LSQ)
scale_lsq = np.dot(fwd_raw, gt_vx_body_s) / np.dot(fwd_raw, fwd_raw)
print(f"  LSQ scale = {scale_lsq:.4f}  (mean: GT={gt_vx_body_s.mean():.4f} / pred={fwd_raw.mean():.4f} = {gt_vx_body_s.mean()/fwd_raw.mean():.4f})")

for scale in [1.0, gt_vx_body_s.mean()/fwd_raw.mean(), scale_lsq, 1.5, 2.0]:
    fwd_sc = fwd_raw * scale
    pos, ate = integrate_and_ate(fwd_sc, gps, rot, dt_eff)
    print(f"  scale={scale:.4f}  ATE={ate:.2f}m")

# 画最佳结果
print()
print("=" * 70)
print("绘制最佳方案对比图")
print("=" * 70)
best_schemes = [
    ('b1[0] norm (现方案)', run_model(imu_norm, 'checkpoint_28.pt')[:, 0], imu_norm),
    ('b1_z norm (最佳)', run_model(imu_norm, 'checkpoint_28.pt')[:, 2], imu_norm),
    ('b1[1] nonorm', run_model(imu_nonorm, 'checkpoint_28.pt')[:, 1], imu_nonorm),
    ('b1_z norm ×scale', run_model(imu_norm, 'checkpoint_28.pt')[:, 2] * (gt_vx_body_s.mean()/fwd_raw.mean()), imu_norm),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
pred_ref = run_model(imu_norm, 'checkpoint_28.pt')
idxs, dt_eff, gps_ref, rot_ref = get_aligned(pred_ref)

for idx, (name, fwd, _) in enumerate(best_schemes):
    if len(fwd) != len(gps_ref):
        idxs2, dt_eff2, gps2, rot2 = get_aligned(np.zeros((len(fwd), 3)))
    else:
        idxs2, dt_eff2, gps2, rot2 = idxs, dt_eff, gps_ref, rot_ref
    pos, ate = integrate_and_ate(fwd, gps2, rot2, dt_eff2)
    axes[idx].plot(gps2[:, 0], gps2[:, 1], 'k--', lw=1.5, label='GT')
    axes[idx].plot(pos[:, 0], pos[:, 1], 'r-', lw=1.5, label=f'ATE={ate:.1f}m')
    axes[idx].set_title(name, fontsize=10)
    axes[idx].axis('equal'); axes[idx].grid(True); axes[idx].legend()

plt.suptitle('Car Trajectory: Best Schemes Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('diagnose5_final.png', dpi=150)
print("图已保存: diagnose5_final.png")
