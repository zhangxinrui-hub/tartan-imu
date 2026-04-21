"""快速预览三条轨迹的 XY 形状，判断哪张最好看"""
import sys, os
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'test', 'car'))
from model import TartanIMUModel, load_checkpoint

def run(imu_np, model, device, head, chunk_size=1000):
    all_b1, all_bz = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imu_np), chunk_size):
            c = torch.FloatTensor(imu_np[i:i+chunk_size]).unsqueeze(0).permute(0,2,1).to(device)
            res = model(c)
            out = res[0] if isinstance(res, tuple) else res
            b1 = out[head][0].cpu().numpy().squeeze()
            bz = out[head][2].cpu().numpy().squeeze()
            if b1.ndim == 1: b1 = b1[:,None]
            if bz.ndim == 1: bz = bz[:,None]
            all_b1.append(b1); all_bz.append(bz)
    return np.concatenate(all_b1), np.concatenate(all_bz)

def integrate(vw, dt, s0):
    pos = np.zeros((len(vw),3)); pos[0]=s0
    for k in range(1,len(vw)): pos[k]=pos[k-1]+vw[k-1]*dt
    return pos

def align(gt,M,N): return gt[np.linspace(0,N-1,M).astype(int)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TartanIMUModel().to(device)
load_checkpoint(model, os.path.join(ROOT,'checkpoint_28.pt'), device)
model.eval()

# ---------- CAR ----------
d = np.load(os.path.join(ROOT,'test','car','pretrain_1.npz'))
iu,gp,gq,ts = d['retargetted_imu'],d['retargetted_pos'],d['retargetted_quat'],d['retargetted_ts']
N=len(iu)
gb=R.from_quat(gq).inv().apply([0,0,9.81])
acc=(iu[:,:3]-gb-np.mean((iu[:,:3]-gb)[:200],0))/9.81
gyro=iu[:,3:]-np.mean(iu[:200,3:],0)
imu=np.concatenate([acc,gyro],1).astype(np.float32)
b1,bz=run(imu,model,device,'car')
M=len(bz); gps=align(gp,M,N); gqs=align(gq,M,N)
dt=(ts[-1]-ts[0])/M
vb=np.zeros((M,3)); vb[:,0]=bz[:,0]
vw=R.from_quat(gqs).apply(vb)
pp_car=integrate(vw,dt,gps[0])
gt_car=gps

# ---------- HUMAN ----------
d=np.load(os.path.join(ROOT,'test','human','pretrain_1.npz'))
ts2=np.squeeze(d['retargetted_ts'])
iu2,gp2,gq2=d['retargetted_imu'],d['retargetted_pos'],d['retargetted_quat']
dur=ts2[-1]-ts2[0]; nt=np.linspace(ts2[0],ts2[-1],int(dur*200))
def rs(a): return interp1d(ts2,a,axis=0,fill_value='extrapolate')(nt)
iu2,gp2,gq2=rs(iu2).astype(np.float32),rs(gp2),rs(gq2)
gq2/=np.linalg.norm(gq2,1,keepdims=True); N2=len(iu2)
gb2=R.from_quat(gq2).inv().apply([0,0,9.81])
acc2=(iu2[:,:3]-gb2-np.mean((iu2[:,:3]-gb2)[:200],0))/9.81
gyro2=iu2[:,3:]-np.mean(iu2[:200,3:],0)
imu2=np.concatenate([acc2,gyro2],1).astype(np.float32)
b1h,bzh=run(imu2,model,device,'human')
M2=len(bzh); gps2=align(gp2,M2,N2); gqs2=align(gq2,M2,N2)
dt2=dur/M2
vb2=np.hstack([b1h,bzh]); vw2=R.from_quat(gqs2).apply(vb2); vw2[:,2]-=np.mean(vw2[:,2])
pp_human=integrate(vw2,dt2,gps2[0]); gt_human=gps2

# ---------- DRONE ----------
base=os.path.join(ROOT,'Dataset_drone')
iu3=np.load(os.path.join(base,'imu_data.npy')).astype(np.float32)
gp3=np.load(os.path.join(base,'gt_pos.npy')); gq3=np.load(os.path.join(base,'gt_quat.npy'))
N3=len(iu3); dur3=N3/200.
acc3=(iu3[:,:3]-np.mean(iu3[:200,:3],0))/9.81
gyro3=iu3[:,3:]-np.mean(iu3[:200,3:],0)
imu3=np.concatenate([acc3,gyro3],1).astype(np.float32)
b1d,bzd=run(imu3,model,device,'drone')
M3=len(bzd); gps3=align(gp3,M3,N3); gqs3=align(gq3,M3,N3)
dt3=dur3/M3
vb3=np.hstack([b1d,bzd]); vw3=R.from_quat(gqs3).apply(vb3)
pp_drone=integrate(vw3,dt3,gps3[0]); gt_drone=gps3

# ---------- PLOT ----------
fig,axes=plt.subplots(1,3,figsize=(18,6))
pairs=[('Car (UGV)',gt_car,pp_car,'#2E86AB'),
       ('Human',gt_human,pp_human,'#27AE60'),
       ('Drone',gt_drone,pp_drone,'#E67E22')]
for ax,(name,gt,pp,c) in zip(axes,pairs):
    ate=float(np.sqrt(np.mean(np.linalg.norm(gt-pp,axis=1)**2)))
    dist=float(np.sum(np.linalg.norm(np.diff(gt,axis=0),axis=1)))
    ax.plot(gt[:,0],gt[:,1],'k--',lw=2,label='GT')
    ax.plot(pp[:,0],pp[:,1],color=c,lw=2,label=f'Pred ATE={ate:.1f}m')
    ax.plot(gt[0,0],gt[0,1],'go',ms=9)
    ax.set_title(f'{name}  ATE={ate:.1f}m  drift={ate/dist*100:.1f}%',fontsize=11)
    ax.axis('equal'); ax.grid(True,alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(ROOT,'slides','slide_figures','traj_preview.png'),dpi=150,bbox_inches='tight')
print('Saved preview. Checking GT shapes:')
for name,gt,_,_ in pairs:
    span_x=gt[:,0].max()-gt[:,0].min()
    span_y=gt[:,1].max()-gt[:,1].min()
    print(f'  {name}: X_span={span_x:.1f}m  Y_span={span_y:.1f}m  pts={len(gt)}')
