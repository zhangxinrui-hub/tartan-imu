import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

DATA_PATH = "pretrain_1.npz"

def analyze_raw_data():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    
    # 提取所有数据
    imu = data['retargetted_imu']       # [N, 6]
    quat = data['retargetted_quat']     # [N, 4]
    pos = data['retargetted_pos']       # [N, 3]
    
    acc = imu[:, :3]
    gyro = imu[:, 3:]
    
    print("="*50)
    print("数据取证报告 (Data Forensics Report)")
    print("="*50)
    
    # ----------------------------------------------------
    # 1. 检查加速度单位 (Unit Check)
    # ----------------------------------------------------
    # 计算每一帧加速度的模长 (Norm)
    acc_norms = np.linalg.norm(acc, axis=1)
    avg_norm = np.mean(acc_norms)
    
    print(f"\n[1] 加速度单位检查:")
    print(f"    - 平均模长 (Avg Norm): {avg_norm:.4f}")
    
    unit_verdict = "Unknown"
    if 9.0 < avg_norm < 11.0:
        unit_verdict = "m/s^2 (标准)"
        print("    [OK] 结论: 单位是 m/s^2 (包含了重力)")
    elif 0.9 < avg_norm < 1.1:
        unit_verdict = "g (重力单位)"
        print("    [WARN] 结论: 单位是 g (包含了重力)")
    elif avg_norm < 0.5:
        unit_verdict = "Zero-mean (已去重力)"
        print("    [UNKNOWN] 结论: 看起来已经去除了重力 (纯运动加速度)")
    else:
        print("    [ERROR] 结论: 数据异常，既不是 m/s^2 也不是 g")

    # ----------------------------------------------------
    # 2. 检查重力方向 (Gravity Direction)
    # ----------------------------------------------------
    # 计算三个轴的均值
    acc_mean = np.mean(acc, axis=0)
    print(f"\n[2] 原始加速度均值 (Mean Acc Vector):")
    print(f"    X: {acc_mean[0]:.4f}")
    print(f"    Y: {acc_mean[1]:.4f}")
    print(f"    Z: {acc_mean[2]:.4f}")
    
    dominant_axis = np.argmax(np.abs(acc_mean))
    axis_names = ['X', 'Y', 'Z']
    sign = "+" if acc_mean[dominant_axis] > 0 else "-"
    print(f"    重力主要分布在: {sign}{axis_names[dominant_axis]} 轴")

    # ----------------------------------------------------
    # 3. 检查四元数与坐标系对齐 (Consistency Check)
    # ----------------------------------------------------
    print(f"\n[3] 坐标系一致性验证 (关键!):")
    
    # 我们假设几种常见的情况，看哪种能把重力消掉
    candidates = [
        {"name": "Z-up (+9.8)",   "g_vec": [0, 0, 9.81]},
        {"name": "Z-down (-9.8)", "g_vec": [0, 0, -9.81]},
        {"name": "Y-up (+9.8)",   "g_vec": [0, 9.81, 0]},
    ]
    
    # 如果单位是 g，我们需要缩放重力向量
    scale = 1.0 if "g" in unit_verdict else 9.81
    
    best_residual = float('inf')
    best_config = None
    
    # 取前 1000 帧来测试
    test_len = min(1000, len(acc))
    acc_test = acc[:test_len]
    quat_test = quat[:test_len]
    r_obj = R.from_quat(quat_test) # 默认 scalar-last [x,y,z,w]
    
    print(f"    正在测试不同的重力假设 (Scale={scale:.2f})...")
    
    for conf in candidates:
        g_world = np.array(conf["g_vec"]) / 9.81 * scale
        
        # 将世界系重力转到 Body 系: R_wb^T * g_world
        g_body_expected = r_obj.inv().apply(g_world)
        
        # 计算残差: |acc_raw - g_body_expected|
        # 如果假设正确，去重力后的加速度均值应该接近 0
        acc_net = acc_test - g_body_expected
        residual = np.mean(np.linalg.norm(acc_net, axis=1))
        
        print(f"    - 假设 {conf['name']:<15} -> 残差: {residual:.4f}")
        
        if residual < best_residual:
            best_residual = residual
            best_config = conf

    print(f"    [OK] 最可能的配置: {best_config['name']}")
    if best_residual > 2.0:
        print("    [WARN] 警告: 即使是最好的假设，残差依然很大。可能四元数格式不对 (Scalar-first?)")

    # ----------------------------------------------------
    # 4. 可视化原始数据
    # ----------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(acc[:2000, 0], label='Acc X', alpha=0.5)
    plt.plot(acc[:2000, 1], label='Acc Y', alpha=0.5)
    plt.plot(acc[:2000, 2], label='Acc Z', alpha=0.5)
    plt.title(f"Raw Accelerometer Data (First 2000 frames)\nAvg Norm: {avg_norm:.2f}")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(gyro[:2000, 0], label='Gyro X', alpha=0.5)
    plt.plot(gyro[:2000, 1], label='Gyro Y', alpha=0.5)
    plt.plot(gyro[:2000, 2], label='Gyro Z', alpha=0.5)
    plt.title("Raw Gyroscope Data")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("forensics_report.png")
    print("\n原始数据图表已保存为 'forensics_report.png'")

if __name__ == "__main__":
    analyze_raw_data()