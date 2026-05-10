"""
全组合测试：找出 ATE 最低的速度分量组合
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
            v1 = c[1].cpu().numpy().squeeze()  # block2: [T,3]
            v2 = c[2].cpu().numpy().squeeze()  # block1_z: [T,1]
            if v0.ndim==1: v0=v0[:,None]
            if v1.ndim==1: v1=v1[:,None]
            if v2.ndim==1: v2=v2[:,None]
            all_out.append(np.hstack([v0, v1, v2]))  # [T, 6]
    return np.concatenate(all_out, axis=0)

imu_norm  = preprocess(normalize=True)
imu_nonorm= preprocess(normalize=False)
pred_n  = run_model(imu_norm)
pred_nn = run_model(imu_nonorm)

M = len(pred_n)
idxs = np.linspace(0, N-1, M).astype(int)
dt_eff = (ts[-1] - ts[0]) / M
gt_pos_s  = gt_pos[idxs]
gt_quat_s = gt_quat[idxs]
rot = R.from_quat(gt_quat_s)
rot0 = R.from_quat(gt_quat[0])   # 初始旋转

def ate(pred):
    return np.sqrt(np.mean(np.linalg.norm(pred - gt_pos_s, axis=1)**2))

def integrate(vel_body, use_rotation=True, fixed_rot=None):
    if use_rotation:
        r = fixed_rot if fixed_rot is not None else rot
        vel_w = r.apply(vel_body)
    else:
        vel_w = vel_body.copy()
    pos = np.zeros_like(gt_pos_s)
    pos[0] = gt_pos_s[0]
    for k in range(1, M):
        pos[k] = pos[k-1] + vel_w[k-1] * dt_eff
    return pos

# col indices: 0=b1[0], 1=b1[1], 2=b2[0], 3=b2[1], 4=b2[2], 5=b1_z
labels = ['b1[0]','b1[1]','b2[0]','b2[1]','b2[2]','b1_z']

results = []
best_ate = 1e9; best_name = ""

for pname, pred in [('norm', pred_n), ('nonorm', pred_nn)]:
    for i, lbl in enumerate(labels):
        for use_vz in [False]:  # 先只测前向+约束
            vb = np.zeros((M, 3))
            vb[:, 0] = pred[:, i]
            vb[:, 1] = 0.0
            vb[:, 2] = 0.0
            pos = integrate(vb)
            a = ate(pos)
            results.append((a, f"{pname} Vfwd={lbl} const", vb, pos))
            if a < best_ate: best_ate = a; best_name = results[-1][1]

    # 测试无约束：用全部分量
    for (i0,i1,i2), lbl in [((0,1,5), '3D:b1+bz'), ((2,3,4), '3D:b2')]:
        vb = np.zeros((M, 3))
        vb[:, 0] = pred[:, i0]
        vb[:, 1] = pred[:, i1]
        vb[:, 2] = pred[:, i2]
        pos = integrate(vb)
        a = ate(pos)
        results.append((a, f"{pname} {lbl} noconst", vb, pos))
        if a < best_ate: best_ate = a; best_name = results[-1][1]

    # 直接作为世界系速度（不旋转）
    for i, lbl in enumerate(labels):
        vb = np.zeros((M, 3))
        vb[:, 0] = pred[:, i]
        pos = integrate(vb, use_rotation=False)
        a = ate(pos)
        results.append((a, f"{pname} WORLD Vx={lbl}", vb, pos))
        if a < best_ate: best_ate = a; best_name = results[-1][1]

    # 直接以世界系3D不旋转
    for (i0,i1,i2), lbl in [((0,1,5),'3D:b1+bz'), ((2,3,4),'3D:b2')]:
        vb = np.zeros((M, 3))
        vb[:, 0] = pred[:, i0]; vb[:, 1] = pred[:, i1]; vb[:, 2] = pred[:, i2]
        pos = integrate(vb, use_rotation=False)
        a = ate(pos)
        results.append((a, f"{pname} WORLD {lbl}", vb, pos))
        if a < best_ate: best_ate = a; best_name = results[-1][1]

results.sort(key=lambda x: x[0])
print(f"{'ATE(m)':>8}  {'Scheme'}")
print("-" * 60)
for a, name, _, _ in results[:20]:
    print(f"{a:8.2f}  {name}")

print(f"\nBest: {best_name}  ATE={best_ate:.2f}m")

# 画前6名
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, (a, name, _, pos) in enumerate(results[:6]):
    ax = axes[idx]
    ax.plot(gt_pos_s[:, 0], gt_pos_s[:, 1], 'k--', lw=1.5, label='GT')
    ax.plot(pos[:, 0], pos[:, 1], 'r-', lw=1.5, label=f'ATE={a:.1f}m')
    ax.set_title(name, fontsize=9)
    ax.axis('equal'); ax.grid(True); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('diagnose3_top6.png', dpi=150)
print("图已保存: diagnose3_top6.png")
