"""找最佳视觉段：对每个平台测试不同时长的短段轨迹，保存预览图"""
import sys, os, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT,'test','car'))
from model import TartanIMUModel, load_checkpoint

def run(imu, model, device, head, chunk=1000):
    all_b1, all_bz = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imu), chunk):
            c = torch.FloatTensor(imu[i:i+chunk]).unsqueeze(0).permute(0,2,1).to(device)
            res = model(c); out = res[0] if isinstance(res,tuple) else res
            b1 = out[head][0].cpu().numpy().squeeze()
            bz = out[head][2].cpu().numpy().squeeze()
            if b1.ndim==1: b1=b1[:,None]
            if bz.ndim==1: bz=bz[:,None]
            all_b1.append(b1); all_bz.append(bz)
    return np.concatenate(all_b1), np.concatenate(all_bz)

def integ(vw, dt, s0):
    pos = np.zeros((len(vw),3)); pos[0]=s0
    for k in range(1,len(vw)): pos[k]=pos[k-1]+vw[k-1]*dt
    return pos

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TartanIMUModel().to(device)
load_checkpoint(model, os.path.join(ROOT,'checkpoint_28.pt'), device)
model.eval()

# ── CAR ──────────────────────────────────────────────────────────
d = np.load(os.path.join(ROOT,'test','car','pretrain_1.npz'))
iu,gp,gq,ts = d['retargetted_imu'],d['retargetted_pos'],d['retargetted_quat'],d['retargetted_ts']
N=len(iu)
gb=R.from_quat(gq).inv().apply([0,0,9.81])
acc=(iu[:,:3]-gb-np.mean((iu[:,:3]-gb)[:200],0))/9.81
gyro=iu[:,3:]-np.mean(iu[:200,3:],0)
imu_car=np.concatenate([acc,gyro],1).astype(np.float32)
b1c,bzc=run(imu_car,model,device,'car')
M=len(bzc); dt_car=(ts[-1]-ts[0])/M
gp_car=gp[np.linspace(0,N-1,M).astype(int)]
gq_car=gq[np.linspace(0,N-1,M).astype(int)]
vb_car=np.zeros((M,3)); vb_car[:,0]=bzc[:,0]
vw_car=R.from_quat(gq_car).apply(vb_car)
pp_car=integ(vw_car,dt_car,gp_car[0])

# ── HUMAN ────────────────────────────────────────────────────────
d=np.load(os.path.join(ROOT,'test','human','pretrain_1.npz'))
ts2=np.squeeze(d['retargetted_ts'])
iu2,gp2,gq2=d['retargetted_imu'],d['retargetted_pos'],d['retargetted_quat']
dur2=ts2[-1]-ts2[0]; nt=np.linspace(ts2[0],ts2[-1],int(dur2*200))
def rs(a): return interp1d(ts2,a,axis=0,fill_value='extrapolate')(nt)
iu2,gp2,gq2=rs(iu2).astype(np.float32),rs(gp2),rs(gq2)
gq2/=np.linalg.norm(gq2,1,keepdims=True); N2=len(iu2)
gb2=R.from_quat(gq2).inv().apply([0,0,9.81])
acc2=(iu2[:,:3]-gb2-np.mean((iu2[:,:3]-gb2)[:200],0))/9.81
gyro2=iu2[:,3:]-np.mean(iu2[:200,3:],0)
imu_hum=np.concatenate([acc2,gyro2],1).astype(np.float32)
b1h,bzh=run(imu_hum,model,device,'human')
M2=len(bzh); dt_h=dur2/M2
gp_hum=gp2[np.linspace(0,N2-1,M2).astype(int)]
gq_hum=gq2[np.linspace(0,N2-1,M2).astype(int)]
vb_h=np.hstack([b1h,bzh]); vw_h=R.from_quat(gq_hum).apply(vb_h)
vw_h[:,2]-=np.mean(vw_h[:,2])
pp_hum=integ(vw_h,dt_h,gp_hum[0])

# ── Analysis: short segments ──────────────────────────────────────
print(f'Car:   total_pts={len(gp_car)}, dt={dt_car:.3f}s, total_t={(len(gp_car)*dt_car):.0f}s')
print(f'Human: total_pts={len(gp_hum)}, dt={dt_h:.3f}s, total_t={(len(gp_hum)*dt_h):.0f}s')

def ate_seg(gt, pred):
    return float(np.sqrt(np.mean(np.linalg.norm(gt-pred,axis=1)**2)))

for name, gp_f, pp_f, dt_f in [('Car',gp_car,pp_car,dt_car),('Human',gp_hum,pp_hum,dt_h)]:
    print(f'\n--- {name} segment analysis ---')
    for t_sec in [30,60,90,120,180]:
        n = int(t_sec / dt_f)
        if n > len(gp_f): break
        a = ate_seg(gp_f[:n], pp_f[:n])
        dist = float(np.sum(np.linalg.norm(np.diff(gp_f[:n],axis=0),axis=1)))
        span_x = gp_f[:n,0].max()-gp_f[:n,0].min()
        span_y = gp_f[:n,1].max()-gp_f[:n,1].min()
        print(f'  {t_sec:3d}s: ATE={a:.2f}m  dist={dist:.1f}m  drift={a/max(dist,1)*100:.1f}%  span=({span_x:.1f},{span_y:.1f})m')

# ── Save individual segment preview ─────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for row, (name, gp_f, pp_f, dt_f) in enumerate([('Car',gp_car,pp_car,dt_car),('Human',gp_hum,pp_hum,dt_h)]):
    for col, t_sec in enumerate([60, 120, 180, 'full']):
        ax = axes[row][col]
        if t_sec == 'full':
            gt, pp = gp_f, pp_f
            title = f'{name} full'
        else:
            n = int(t_sec / dt_f)
            gt, pp = gp_f[:n], pp_f[:n]
            title = f'{name} first {t_sec}s'
        a = ate_seg(gt, pp)
        dist = float(np.sum(np.linalg.norm(np.diff(gt,axis=0),axis=1))) if len(gt)>1 else 0
        ax.plot(gt[:,0], gt[:,1], 'k--', lw=2, label='GT')
        ax.plot(pp[:,0], pp[:,1], 'r-', lw=2, label=f'Pred ATE={a:.1f}m')
        ax.plot(gt[0,0], gt[0,1], 'go', ms=8)
        ax.set_title(f'{title}\nATE={a:.1f}m drift={a/max(dist,1)*100:.1f}%', fontsize=9)
        ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(ROOT,'slides','slide_figures','seg_preview.png'), dpi=120, bbox_inches='tight')
print('\nSaved seg_preview.png')
