import pandas as pd
import pymap3d as pm
import numpy as np

# --- 配置 ---
gt_file = 'gt_large_loop.txt'     # 你的真值文件名
bag_start_time = 1751959993.0     # Bag 开始时间
bag_end_time = 1751960594.0       # Bag 结束时间
output_csv = 'clean_ground_truth.csv'
# -----------

print("正在读取真值文件 (可能需要几秒钟)...")

try:
    # 1. 读取 TXT 文件
    df = pd.read_csv(gt_file, delim_whitespace=True, skiprows=[1])

    # 2. 裁剪时间
    mask = (df['UTCTime'] >= bag_start_time - 5) & (df['UTCTime'] <= bag_end_time + 5)
    df_segment = df.loc[mask].copy()

    print(f"原始数据 {len(df)} 行，裁剪后剩余 {len(df_segment)} 行。")

    if len(df_segment) == 0:
        print("错误：裁剪后没有数据！请检查时间范围。")
        exit()

    # 3. 坐标转换 (LLA -> ENU 局部坐标)
    lat0 = df_segment.iloc[0]['Latitude']
    lon0 = df_segment.iloc[0]['Longitude']
    h0 = df_segment.iloc[0]['H-Ell']

    print(f"设定真值原点: Lat={lat0}, Lon={lon0}")

    x, y, z = pm.geodetic2enu(
        df_segment['Latitude'].values,
        df_segment['Longitude'].values,
        df_segment['H-Ell'].values,
        lat0, lon0, h0
    )

    # 4. 保存结果
    result_df = pd.DataFrame({
        'timestamp': df_segment['UTCTime'],
        'x_gt': x,
        'y_gt': y,
        'z_gt': z,
        'roll_gt': df_segment['Roll'],
        'pitch_gt': df_segment['Pitch'],
        'yaw_gt': df_segment['Heading'],
        'v_east': df_segment['VEast'],
        'v_north': df_segment['VNorth'],
        'v_up': df_segment['VUp']
    })

    result_df.to_csv(output_csv, index=False)
    print(f"处理完成！已生成标准真值文件: {output_csv}")

except Exception as e:
    print(f"发生错误: {e}")
