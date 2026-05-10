import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
import os

# 引用 model.py
try:
    from model import TartanIMUModel
except ImportError:
    print("[ERROR] 找不到 model.py")
    sys.exit(1)

def calculate_ate(gt_pos, pred_pos):
    errors = np.linalg.norm(gt_pos - pred_pos, axis=1)
    return np.sqrt(np.mean(errors**2))

def run_restore_test(ckpt_path, data_path, device='cuda'):
    print(f"正在尝试复原 9.11m 的结果...")
    print(f"目标文件: {ckpt_path}")

    # --- 1. 深度检查 Checkpoint ---
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in ckpt: state = ckpt['model_state_dict']
        else: state = ckpt
        
        keys = list(state.keys())
        print(f"文件包含 {len(keys)} 个参数")
        
        # 检查是否包含 LoRA
        has_lora = any('lora' in k for k in keys)
        if has_lora:
            print("[WARN] 警告: 这个文件包含 LoRA 参数！它可能不是纯 Base 模型。")
            print("   (这可能解释了为什么当成 Base 加载会出错)")
        else:
            print("[OK] 确认: 这是一个纯 Base 模型 (无 LoRA 关键字)。")
            
    except Exception as e:
        print(f"[ERROR] 文件损坏或无法读取: {e}")
        return

    # --- 2. 加载数据 & 关键的 Bias 处理 ---
    data = np.load(data_path)
    imu = data['retargetted_imu']
    gt_pos = data['retargetted_pos']
    gt_quat = data['retargetted_quat']
    
    # 关键步骤：强制去除 Bias 
    # 取前 50 帧 (假设静止) 的均值作为 Bias
    static_bias = np.mean(imu[:50], axis=0)
    print(f"检测到静态 Bias: {static_bias}")
    print("   -> 正在执行去 Bias 操作 (imu = imu - bias)...")
    processed_imu = imu - static_bias
    
    # --- 3. 模型初始化与加载 ---
    # 既然是跑 Base 结果，我们关闭 LoRA 模式
    model = TartanIMUModel(fine_tune_mode=None).to(device)
    
    # 清理 keys 并加载
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    
    # 尝试加载
    try:
        model.load_state_dict(clean_state, strict=False)
        print("[OK] 模型权重加载成功")
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        return

    model.eval()

    # --- 4. 推理 (带约束) ---
    print("开始推理...")
    all_vel = []
    chunk_size = 1000
    
    with torch.no_grad():
        for i in range(0, len(processed_imu), chunk_size):
            end = min(i + chunk_size, len(processed_imu))
            inp = torch.FloatTensor(processed_imu[i:end]).unsqueeze(0).permute(0, 2, 1).to(device)
            
            outputs = model(inp)
            if isinstance(outputs, dict):
                out = outputs.get('car', list(outputs.values())[0])
            else: out = outputs
            
            # 兼容不同输出格式
            if isinstance(out, (list, tuple)):
                v_xy = out[0].cpu().numpy()
                v_z = out[2].cpu().numpy() if len(out)>2 else np.zeros_like(v_xy[...,0:1])
            else:
                tmp = out.cpu().numpy()
                v_xy, v_z = tmp[...,:2], tmp[...,2:]
                
            if v_xy.ndim == 3: v_xy = v_xy[0]
            if v_z.ndim == 3: v_z = v_z[0]
            
            all_vel.append(np.hstack([v_xy, v_z]))

    # --- 5. 积分与评估 ---
    vel_pred = np.concatenate(all_vel, axis=0)
    min_len = min(len(vel_pred), len(gt_pos))
    
    # 截断
    vel_pred = vel_pred[:min_len]
    gt_pos = gt_pos[:min_len]
    gt_quat = gt_quat[:min_len]
    
    # Car 约束 (Vy=0, Vz=0)
    vel_constrained = np.zeros_like(vel_pred)
    vel_constrained[:, 0] = vel_pred[:, 0]
    
    # 旋转
    rot_mat = R.from_quat(gt_quat).as_matrix()
    vel_world = np.einsum('nij,nj->ni', rot_mat, vel_constrained)
    
    # 积分
    dt = 0.005
    pred_pos = np.zeros_like(gt_pos)
    pred_pos[0] = gt_pos[0]
    for k in range(1, min_len):
        pred_pos[k] = pred_pos[k-1] + vel_world[k] * dt
        
    ate = calculate_ate(gt_pos, pred_pos)
    
    print(f"\n{'='*40}")
    print(f"复原测试结果")
    print(f"ATE: {ate:.4f} m")
    if ate < 12.0:
        print("成功复原！")
        print("   -> 之前的 34m 是因为没去 Bias。")
        print("   -> 加载失败可能是因为 strict=True 或 LoRA 定义干扰。")
    else:
        print("依然不是 9m... 这说明 checkponit_28.pt 可能真的不是那个文件。")
    print(f"{'='*40}")

if __name__ == "__main__":
    # 填入你确信的那个文件
    CKPT = "checkpoint_28.pt"
    DATA = "pretrain_1.npz"
    run_restore_test(CKPT, DATA)