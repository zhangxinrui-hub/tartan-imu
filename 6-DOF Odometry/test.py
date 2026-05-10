import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from keras.models import load_model

from dataset import *
from util import *
# 引入 LSTM 用于 CPU 兼容性修复
from keras.layers import LSTM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('model', help='Model path')
    parser.add_argument('input', help='Input sequence path (e.g. \"Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv\" for OxIOD, \"MH_02_easy/mav0/imu0/data.csv\" for EuRoC)')
    parser.add_argument('gt', help='Ground truth path (e.g. \"Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv\" for OxIOD, \"MH_02_easy/mav0/state_groundtruth_estimate0/data.csv\" for EuRoC)')
    args = parser.parse_args()

    window_size = 200
    stride = 10

    # 加载模型时使用 custom_objects 解决 CuDNNLSTM 在 CPU 上无法运行的问题
    model = load_model(args.model, custom_objects={'CuDNNLSTM': LSTM})

    if args.dataset == 'oxiod':
        gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(args.input, args.gt)
    elif args.dataset == 'euroc':
        gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(args.input, args.gt)

    [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)

    # 修改点 1：删除了 [0:200] 的切片限制，使用完整数据进行预测
    if args.dataset == 'oxiod':
        [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=1)
    elif args.dataset == 'euroc':
        [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=1)

    gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
    pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

    # 修改点 2：删除了对 Ground Truth 的截取，确保对比的是完整轨迹
    # if args.dataset == 'oxiod':
    #     gt_trajectory = gt_trajectory[0:200, :]

    # --- 新增：计算并打印 RMSE ---
    # 计算每一帧的位置误差
    diff = gt_trajectory - pred_trajectory
    # 对每一帧的误差平方求和，取平均，再开根号
    rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    print(f"==============================================")
    print(f"Trajectory RMSE (误差): {rmse:.4f} m")
    print(f"==============================================")
    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=[14.4, 10.8])
    ax = fig.gca(projection='3d')
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 自动调整坐标轴范围
    min_x = np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0]))
    min_y = np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1]))
    min_z = np.minimum(np.amin(gt_trajectory[:, 2]), np.amin(pred_trajectory[:, 2]))
    max_x = np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0]))
    max_y = np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1]))
    max_z = np.maximum(np.amax(gt_trajectory[:, 2]), np.amax(pred_trajectory[:, 2]))
    range_x = np.absolute(max_x - min_x)
    range_y = np.absolute(max_y - min_y)
    range_z = np.absolute(max_z - min_z)
    max_range = np.maximum(np.maximum(range_x, range_y), range_z)
    ax.set_xlim(min_x, min_x + max_range)
    ax.set_ylim(min_y, min_y + max_range)
    ax.set_zlim(min_z, min_z + max_range)
    
    ax.legend(['ground truth', 'predicted'], loc='upper right')
    
    # 修改点 3：添加图片保存功能
    print("Saving trajectory plot to 'result_trajectory.png'...")
    plt.savefig('result_trajectory.png')
    
    plt.show()

if __name__ == '__main__':
    main()