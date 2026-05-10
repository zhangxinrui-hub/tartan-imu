import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def preprocess_car_data_perfectly():
    print("1. 读取数据...")
    df = pd.read_csv('car_imu_data_full.csv')
    
    ts_raw = df['timestamp'].values
    acc_raw = df[['ax', 'ay', 'az']].values
    gyr_raw = df[['gx', 'gy', 'gz']].values
    quat_raw = df[['qx', 'qy', 'qz', 'qw']].values 
    
    print(f"原始数据范围: {ts_raw[0]:.4f} -> {ts_raw[-1]:.4f}")
    print(f"原始频率: {1/np.mean(np.diff(ts_raw)):.1f} Hz")

    # 2. 降采样 (400Hz -> 200Hz)
    target_fs = 200.0
    dt_target = 1.0 / target_fs
    
    # --- 修复开始 ---
    # 生成新时间轴
    ts_new = np.arange(ts_raw[0], ts_raw[-1], dt_target)
    
    # 强制过滤：去掉任何可能因为浮点误差而超出 ts_raw[-1] 的点
    ts_new = ts_new[ts_new <= ts_raw[-1]]
    
    # 再次检查：确保最小值也不越界（虽然通常不会）
    ts_new = ts_new[ts_new >= ts_raw[0]]
    # --- 修复结束 ---

    print(f"新时间轴长度: {len(ts_new)} (从 {ts_new[0]:.4f} 到 {ts_new[-1]:.4f})")

    # 插值函数
    print("正在插值...")
    f_acc = interp1d(ts_raw, acc_raw, axis=0, kind='linear')
    f_gyr = interp1d(ts_raw, gyr_raw, axis=0, kind='linear')
    f_quat = interp1d(ts_raw, quat_raw, axis=0, kind='linear') 
    
    acc_new = f_acc(ts_new)
    gyr_new = f_gyr(ts_new)
    quat_new = f_quat(ts_new)
    
    # 归一化插值后的四元数
    quat_new /= np.linalg.norm(quat_new, axis=1, keepdims=True)

    # 3. 完美去重力
    print("正在执行完美去重力...")
    r = R.from_quat(quat_new) 
    g_world = np.array([0.0, 0.0, 9.81]) 
    
    # 计算重力在机体坐标系下的分量
    g_body = r.apply(g_world, inverse=True)
    
    # 执行减法
    acc_no_gravity = acc_new - g_body
    
    print(f"去重力前均值: {np.mean(np.linalg.norm(acc_new, axis=1)):.2f}")
    print(f"去重力后均值: {np.mean(np.linalg.norm(acc_no_gravity, axis=1)):.2f}")

    # 4. 角速度单位转换 (如果需要)
    if np.max(np.abs(gyr_new)) > 15.0:
        print("检测到角速度可能是 deg/s，转换为 rad/s")
        gyr_new *= (np.pi / 180.0)

    # 5. 保存
    processed_data = np.concatenate([acc_no_gravity, gyr_new], axis=1)
    
    save_name = 'processed_car_data_final.npy'
    np.save(save_name, processed_data)
    print(f"[OK] 处理完成！已保存为 {save_name}")
    
    return ts_new, processed_data

if __name__ == "__main__":
    ts, data = preprocess_car_data_perfectly()