import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import sys

# 导入 model.py (确保它在同一目录下)
try:
    from model import TartanIMUModel
except ImportError:
    print("[ERROR] 错误: 找不到 model.py")
    sys.exit(1)

def calculate_ate(gt_pos, pred_pos):
    errors = np.linalg.norm(gt_pos - pred_pos, axis=1)
    return np.sqrt(np.mean(errors**2))

def check_base_performance(ckpt_path, data_path, device='cuda'):
    print(f"\n{'='*40}")
    print(f"正在执行基准模型体验证 (Base Model Check)")
    print(f"权重文件: {ckpt_path}")
    print(f"{'='*40}")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 文件不存在: {ckpt_path}")
        return

    # 1. 加载数据
    data = np.load(data_path)
    imu_raw = data['retargetted_imu']
    gt_pos  = data['retargetted_pos']
    gt_quat = data['retargetted_quat']
    timestamps = data['retargetted_ts']
    total_len = len(imu_raw)

    # 预处理：去重力 + 去Bias + 归一化
    acc_raw  = imu_raw[:, :3]
    gyro_raw = imu_raw[:, 3:]
    g_world  = np.array([0.0, 0.0, 9.81])
    r_obj    = R.from_quat(gt_quat)
    g_body   = r_obj.inv().apply(g_world)
    acc_net  = acc_raw - g_body
    acc_net  -= np.mean(acc_net[:200],  axis=0)
    gyro_raw  = gyro_raw - np.mean(gyro_raw[:200], axis=0)
    acc_net  /= 9.81
    imu = np.concatenate([acc_net, gyro_raw], axis=1).astype(np.float32)

    # 2. 初始化纯净的 Stage 1 模型 (关闭 LoRA)
    print("初始化 Stage 1 模型 (无 LoRA)...")
    model = TartanIMUModel(fine_tune_mode=None).to(device) # fine_tune_mode=None 关键！
    
    # 3. 加载权重
    print("正在加载权重...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 处理字典嵌套
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # 去除 module. 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 尝试加载 (允许不严格匹配，因为 checkpoint 可能包含多余的优化器状态)
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("[OK] 权重加载成功 (Strict Mode)")
    except Exception as e:
        print(f"[WARN] Strict 加载失败 (可能包含多余Keys), 尝试 Strict=False...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        # 过滤掉 LoRA 相关的缺失键 (如果有的话)
        real_missing = [k for k in missing if 'lora' not in k]
        if len(real_missing) > 0:
            print(f"[ERROR] 严重警告: 基础模型缺失关键参数! 例如: {real_missing[:3]}")
            print("   这意味着这个 Checkpoint 不完整，或者架构不匹配！")
        else:
            print("[OK] 权重加载成功 (Strict=False, 忽略了无关 Keys)")

    model.eval()
    
    # 4. 推理
    print("开始推理...")
    all_vel = []
    chunk_size = 1000
    
    with torch.no_grad():
        for i in range(0, len(imu), chunk_size):
            end = min(i + chunk_size, len(imu))
            inp = torch.FloatTensor(imu[i:end]).unsqueeze(0).permute(0, 2, 1).to(device)
            outputs, _ = model(inp)
            
            # 兼容性处理
            if isinstance(outputs, dict):
                # 优先用 car head，如果没有则尝试用 output 列表
                if 'car' in outputs: out = outputs['car']
                else: out = list(outputs.values())[0]
            else:
                out = outputs
            
            # 解析 (vx, vy, vz)
            # 假设 out 是 tuple (v_xy, v_rot, v_z) 或 list
            if isinstance(out, (list, tuple)):
                v_xy = out[0].cpu().numpy()
                v_z = out[2].cpu().numpy() if len(out) > 2 else np.zeros_like(v_xy[...,0:1])
            else:
                # 假如直接输出 Tensor
                tmp = out.cpu().numpy()
                v_xy = tmp[..., :2]
                v_z = tmp[..., 2:]
            
            if v_xy.ndim == 3: v_xy = v_xy[0]
            if v_z.ndim == 3: v_z = v_z[0]
            
            all_vel.append(np.hstack([v_xy, v_z]))

    vel_full = np.concatenate(all_vel, axis=0)
    output_len = len(vel_full)

    print(f"   -> 模型输出长度: {output_len}（输入 {total_len}，比例 {output_len/total_len:.3f}）")

    # 用均匀插值将 gt 对齐到模型输出长度
    idxs    = np.linspace(0, total_len - 1, output_len).astype(int)
    gt_pos  = gt_pos[idxs]
    gt_quat = gt_quat[idxs]

    # 5. 约束与积分 (Car Constraint)
    # 诊断发现：模型前向速度在 block1_z（索引2），不在 block1[0]（索引0）
    vel_constrained = np.zeros_like(vel_full)
    vel_constrained[:, 0] = vel_full[:, 2]  # block1_z → Vx_forward

    rot_mat   = R.from_quat(gt_quat).as_matrix()
    vel_world = np.einsum('nij,nj->ni', rot_mat, vel_constrained)

    # 使用实际输出时间分辨率（约 0.02s/步）
    dt_effective = (timestamps[-1] - timestamps[0]) / output_len
    pred_pos = np.zeros_like(gt_pos)
    pred_pos[0] = gt_pos[0]

    for k in range(1, output_len):
        pred_pos[k] = pred_pos[k - 1] + vel_world[k - 1] * dt_effective
        
    # 6. 结果
    ate = calculate_ate(gt_pos, pred_pos)
    min_len = output_len
    
    print(f"\n{'='*40}")
    print(f"诊断结果 (Diagnostic Result)")
    print(f"Base Model ATE: {ate:.4f} m")
    print(f"{'='*40}")
    
    if ate > 20:
        print("[ERROR] 结论: 这个 Checkpoint 是坏的/错误的！")
        print("   它本身就只能跑出 70m 的误差。")
        print("   -> 请必须找到作者提供的官方预训练模型 (Foundation Model) 替换它。")
    elif ate < 10:
        print("[OK] 结论: 这个 Checkpoint 是好的 (Foundation Model)。")
        print("   -> 问题出在 LoRA 代码参数或加载逻辑上。")
    else:
        print("[WARN] 结论: 模型性能中等，可能是训练不充分的中间版本。")

    # 画图
    plt.figure(figsize=(10,6))
    plt.plot(gt_pos[:,0], gt_pos[:,1], 'k--', label='GT')
    plt.plot(pred_pos[:,0], pred_pos[:,1], 'b-', label=f'Base Only (ATE={ate:.1f}m)')
    plt.title(f"Base Checkpoint Diagnostic: {ckpt_path}")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.savefig('diagnostic_base.png')
    print("图表已保存至 diagnostic_base.png")

if __name__ == "__main__":
    # [WARN] 填入你怀疑的那个文件
    CKPT = "checkpoint_28.pt" 
    DATA = "pretrain_1.npz"
    
    check_base_performance(CKPT, DATA)