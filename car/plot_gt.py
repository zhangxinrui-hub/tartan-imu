import pandas as pd
import matplotlib.pyplot as plt

# 读取刚才生成的真值文件
file_path = 'clean_ground_truth.csv'

try:
    df = pd.read_csv(file_path)
    
    # 提取 X 和 Y 坐标
    x = df['x_gt']
    y = df['y_gt']
    
    # 开始画图
    plt.figure(figsize=(10, 8))
    
    # 画轨迹线 (红色)
    plt.plot(x, y, label='Ground Truth (RTK-GPS)', color='red', linewidth=1.5)
    
    # 标记起点 (绿色星号)
    plt.scatter(x.iloc[0], y.iloc[0], color='green', marker='*', s=300, label='Start', zorder=5)
    
    # 标记终点 (蓝色叉号)
    plt.scatter(x.iloc[-1], y.iloc[-1], color='blue', marker='x', s=200, label='End', zorder=5)

    plt.title('Ground Truth Trajectory (Top View)', fontsize=15)
    plt.xlabel('East (meters)', fontsize=12)
    plt.ylabel('North (meters)', fontsize=12)
    plt.axis('equal')  # 这一点非常重要，保证长宽比例一致，否则圆会变成椭圆
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 保存并显示
    plt.savefig('gt_trajectory.png')
    print("图表已保存为 'gt_trajectory.png'")
    plt.show()

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请确认你刚才的脚本运行成功了。")
except Exception as e:
    print(f"发生了意料之外的错误: {e}")
