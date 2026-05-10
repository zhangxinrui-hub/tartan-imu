import numpy as np
import pandas as pd
import os

def inspect_custom_data(imu_csv, gt_csv):
    print(f"==================================================")
    print(f"正在检查用户数据...")
    print(f"IMU文件: {imu_csv}")
    print(f"真值文件: {gt_csv}")
    print(f"==================================================\n")

    # 1. 加载数据 (使用 Pandas，因为它能自动处理表头)
    # 尝试加载，如果报错可能需要调整分隔符 (delimiter)
    try:
        df_imu = pd.read_csv(imu_csv)
        df_gt = pd.read_csv(gt_csv)
    except Exception as e:
        print(f"读取 CSV 失败，请检查文件路径或格式。错误: {e}")
        return

    # 2. 打印前 5 行 (关键：确认列的顺序)
    print(">>> [1] IMU 数据前 5 行 (请确认列顺序: Time, Acc, Gyro):")
    print(df_imu.head().to_string())
    print("-" * 20)
    
    print(">>> [2] GT 数据前 5 行 (请确认列顺序: Time, Pos, Quat):")
    print(df_gt.head().to_string())
    print("\n")

    # 3. 自动转换与计算 (假设第一列是时间，后六列是数据)
    # 注意：你需要根据上面打印的结果，修改下面的列索引！
    # 这里默认假设：
    # IMU: [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    # GT:  [timestamp, pos_x, pos_y, pos_z, q_x, q_y, q_z, q_w]
    
    imu_values = df_imu.values
    ts_imu = imu_values[:, 0]
    
    # 4. 检查采样率
    dts = np.diff(ts_imu)
    avg_dt = np.mean(dts)
    
    # 如果时间戳是纳秒 (间隔很大)，自动转秒
    if avg_dt > 1000: 
        print("[WARN] 检测到时间戳可能是纳秒 (ns) 或毫秒，正在尝试判断...")
        if avg_dt > 1e8: # ns
            print("   -> 看起来是纳秒，除以 1e9")
            ts_imu = ts_imu / 1e9
            avg_dt /= 1e9
        elif avg_dt > 1e5: # us
            ts_imu = ts_imu / 1e6
            avg_dt /= 1e6
            
    freq = 1.0 / avg_dt if avg_dt > 0 else 0
    print(f">>> [3] 采样率分析:")
    print(f"    平均时间间隔: {avg_dt:.6f} s")
    print(f"    估算频率: {freq:.2f} Hz")
    if abs(freq - 200) > 10:
        print(f"    [WARN] 注意：频率不是 200Hz (TartanIMU 要求)，可能需要重采样！")
    else:
        print(f"    [OK] 频率符合要求 (~200Hz)")

    # 5. 检查单位与重力方向 (前 50 帧均值)
    # 假设 1-3 列是 Acc, 4-6 列是 Gyro
    # 如果你的 CSV 列顺序不一样，请务必人工核对！
    acc_data = imu_values[:50, 1:4] 
    acc_mean = np.mean(acc_data, axis=0)
    g_norm = np.linalg.norm(acc_mean)
    
    print(f"\n>>> [4] 静态偏差与单位分析 (基于前50帧):")
    print(f"    Acc Mean (Col 1,2,3): {acc_mean}")
    print(f"    模长 (Norm): {g_norm:.2f}")
    
    if 9.0 < g_norm < 11.0:
        print("    [OK] 单位看起来是 m/s^2")
    elif 0.9 < g_norm < 1.1:
        print("    [WARN] 单位看起来是 g (重力加速度)，需要 * 9.8")
    else:
        print("    [WARN] 单位异常！请检查数据列是否选对。")
        
    # 判断轴向
    max_axis = np.argmax(np.abs(acc_mean))
    axis_name = ['X', 'Y', 'Z'][max_axis]
    val = acc_mean[max_axis]
    print(f"    重力主要分布在: {axis_name} 轴 (值: {val:.2f})")
    
    if max_axis == 2 and val > 8:
        print("    [OK] 符合 Z-up (Z轴朝上)")
    elif max_axis == 2 and val < -8:
        print("    [WARN] 看起来是 Z-down (Z轴朝下)，可能需要翻转 Z 轴")
    else:
        print(f"    [WARN] 看起来不是 Z轴朝上，可能是 {axis_name} 轴朝上/下")

if __name__ == "__main__":
    # 替换成你的文件名
    imu_file = "car_imu_data_full.csv"
    gt_file = "car_ground_truth.csv"
    
    if os.path.exists(imu_file) and os.path.exists(gt_file):
        inspect_custom_data(imu_file, gt_file)
    else:
        print("[ERROR] 找不到文件，请确认文件名是否正确。")