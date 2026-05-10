import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

try:
    from model import TartanIMUModel, load_checkpoint
except ImportError:
    print("[ERROR] 错误: 找不到 model.py")
    sys.exit(1)

# ==========================================
# 1. 核心算法: ATE 计算与 Sim3 对齐
# ==========================================
def compute_ate_sim3(pred_pos, gt_pos):
    """
    计算 ATE (Absolute Trajectory Error) 并进行 Sim3 对齐
    返回: aligned_pred, ate_rmse, scale_factor
    """
    # 1. 确保点数一致 (重采样 Pred 以匹配 GT)
    if len(pred_pos) != len(gt_pos):
        t_pred = np.linspace(0, 1, len(pred_pos))
        t_gt = np.linspace(0, 1, len(gt_pos))
        interp_func = interp1d(t_pred, pred_pos, axis=0)
        pred_resampled = interp_func(t_gt)
    else:
        pred_resampled = pred_pos

    # 2. 去中心化 (Subtract Mean)
    mu_pred = np.mean(pred_resampled, axis=0)
    mu_gt = np.mean(gt_pos, axis=0)
    
    pred_centered = pred_resampled - mu_pred
    gt_centered = gt_pos - mu_gt

    # 3. 计算 Scale (缩放因子)
    # s = sqrt( sum(gt^2) / sum(pred^2) )
    sig_pred = np.mean(np.sum(pred_centered**2, axis=1))
    sig_gt = np.mean(np.sum(gt_centered**2, axis=1))
    scale = np.sqrt(sig_gt / sig_pred)

    # 4. 计算 Rotation (SVD 分解)
    # H = Pred^T * GT
    H = np.dot(pred_centered.T, gt_centered)
    U, S, Vt = np.linalg.svd(H)
    R_align = np.dot(Vt.T, U.T)

    # 处理反射矩阵 (Det = -1)
    if np.linalg.det(R_align) < 0:
        Vt[1, :] *= -1
        R_align = np.dot(Vt.T, U.T)

    # 5. 应用变换得到对齐后的轨迹
    # Aligned = s * (R * (Pred - mu_p)) + mu_g
    pred_aligned = np.dot(pred_pos - mu_pred, R_align.T) * scale + mu_gt

    # 6. 计算 ATE (RMSE)
    # 只需要计算对齐后 pred_aligned 和 gt_pos 的欧氏距离
    # 注意：这里需要用 resampled 后的点来算误差，保证一一对应
    pred_aligned_for_error = np.dot(pred_resampled - mu_pred, R_align.T) * scale + mu_gt
    errors = np.linalg.norm(gt_pos - pred_aligned_for_error, axis=1)
    ate_rmse = np.sqrt(np.mean(errors**2))

    return pred_aligned, ate_rmse, scale

# ==========================================
# 2. 数据处理类 (保持不变)
# ==========================================
class DatasetSimple:
    def __init__(self, file_path):
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        acc_raw = df[['acc_x', 'acc_y', 'acc_z']].values
        gyr_raw = df[['gyr_x', 'gyr_y', 'gyr_z']].values
        
        if np.max(np.abs(gyr_raw)) > 15.0:
            gyr_raw *= (np.pi / 180.0)

        self.acc_in = np.zeros_like(acc_raw)
        self.gyr_in = np.zeros_like(gyr_raw)

        # X=Up, Y=Left, Z=Fwd 映射到 Model (Fwd, Left, Up)
        self.acc_in[:, 0] = acc_raw[:, 2]  
        self.gyr_in[:, 0] = gyr_raw[:, 2]
        self.acc_in[:, 1] = acc_raw[:, 1]  
        self.gyr_in[:, 1] = gyr_raw[:, 1]
        self.acc_in[:, 2] = acc_raw[:, 0]  
        self.gyr_in[:, 2] = gyr_raw[:, 0]
        
        self.yaw_rate = self.gyr_in[:, 2]
        self.acc_net = self.acc_in.copy()
        self.acc_net[:, 2] -= 9.81 
        self.features = np.concatenate([self.acc_net, self.gyr_in], axis=1)

    def get_data(self):
        return torch.FloatTensor(self.features).permute(1, 0).unsqueeze(0), self.yaw_rate

# ==========================================
# 3. 主程序
# ==========================================
def main():
    CHECKPOINT = "checkpoint_28.pt" 
    IMU_FILE = "4d91_long_waist_imu.xlsx"      
    GT_FILE = "4d91_long_ground_truth.xlsx"    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 推理与积分 ---
    ds = DatasetSimple(IMU_FILE)
    inp, yaw_rate = ds.get_data()
    inp = inp.to(device)
    
    print("模型推理中...")
    model = TartanIMUModel().to(device)
    load_checkpoint(model, CHECKPOINT, device)
    model.eval()
    
    vel_list = []
    # 为了速度，分块推理
    chunk_size = 2000
    for i in range(0, inp.shape[2], chunk_size):
        end = min(i+chunk_size, inp.shape[2])
        with torch.no_grad():
            res = model(inp[:,:,i:end])
            val = torch.cat([res['human'][0], res['human'][2]], dim=-1)
        vel_list.append(val.cpu())
    
    body_vel = torch.cat(vel_list, dim=1).numpy()[0] 
    dt = 0.005
    yaw = np.cumsum(yaw_rate * dt)
    
    L = min(len(body_vel), len(yaw))
    body_vel = body_vel[:L]
    yaw = yaw[:L]
    
    vel_world = np.zeros_like(body_vel)
    vel_world[:, 0] = body_vel[:, 0] * np.cos(yaw) - body_vel[:, 1] * np.sin(yaw)
    vel_world[:, 1] = body_vel[:, 0] * np.sin(yaw) + body_vel[:, 1] * np.cos(yaw)
    
    pred_pos = np.cumsum(vel_world[:, :2] * dt, axis=0) # [N, 2]

    # --- 2. 加载真值 ---
    df_gt = pd.read_excel(GT_FILE, engine='openpyxl') if GT_FILE.endswith('xlsx') else pd.read_csv(GT_FILE)
    df_gt.columns = [c.strip() for c in df_gt.columns]
    
    if 'gt_pos_global_x' in df_gt.columns:
        cols = ['gt_pos_global_x', 'gt_pos_global_y']
    else:
        cols = df_gt.columns[4:6]
        
    gt_pos = df_gt[cols].values
    
    # 单位修正
    if np.max(np.abs(gt_pos)) < 10.0: 
        print("[WARN] 检测到 GT 单位可能是 km，修正为 m...")
        gt_pos *= 1000.0
    gt_pos -= gt_pos[0]

    # --- 3. 计算 ATE (Sim3 对齐) ---
    print("正在计算 ATE 并对齐...")
    
    # 调用我们的新函数
    pred_aligned, ate, scale = compute_ate_sim3(pred_pos, gt_pos)
    
    print("="*40)
    print(f"[OK] 最终评估结果 (Evaluation Results):")
    print(f"   - ATE (RMSE): {ate:.4f} m")
    print(f"   - Scale Factor: {scale:.4f}")
    print("="*40)

    # --- 4. 绘图 ---
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original (Raw Integration)")
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', label='GT')
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'b-', alpha=0.6, label='Pred')
    plt.axis('equal'); plt.grid(True); plt.legend()
    
    plt.subplot(1, 2, 2)
    # 标题显示 ATE
    plt.title(f"Aligned (ATE: {ate:.2f}m, Scale: {scale:.2f})")
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', linewidth=2, label='GT')
    plt.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'g-', linewidth=2, label='Pred (Aligned)')
    plt.axis('equal'); plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.savefig("ate_result.png")
    print("结果图已保存: ate_result.png")
    plt.show()

if __name__ == "__main__":
    main()