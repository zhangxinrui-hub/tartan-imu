# TartanIMU Multi-Platform Evaluation

基于 [TartanIMU（Zhao et al., CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_CVPR_2025_paper.pdf) 的完整评估与部署实验，涵盖多平台测试、EKF 融合、LoRA 域适应以及作为基线对比的 6-DOF 惯性里程计（Lima 2019）。

---

## 项目内容

- **代码与脚本**：训练、评估、LoRA/车头训练、EKF 等
- **`Drone_Dataset1/` 未纳入 Git**（体积过大，约 36GB）；若需复现无人机实验，请从实验室硬盘或网盘单独拷贝
- **权重文件**（如 `checkpoint_28.pt`）已包含在仓库中；建议使用 Conda 环境并安装 PyTorch（`tartan_gpu` 环境）

---

## 网络结构

从 `checkpoint_28.pt` 逆向还原，完整结构如下：

```
Input [B, 6, T]  (acc + gyro，200 Hz，去重力 + 去 Bias + acc÷9.81)
  │
  ├─ CNN Backbone: Conv1d(6→64, k=7) + 3 组 × 2 ResBlock (64→128→256，stride-2)
  ├─ Post-processing: Conv1d 256→128→128（1×1 核）
  ├─ Unfold: 滑窗 (win=13, step=1) → [B, T', 1664]
  ├─ LSTM: input=1664, hidden=64
  ├─ IMU_Trunk: 6× Transformer Block (d=64, heads=4, ff=256)
  │
  └─ 4 输出头 (dog / human / car / drone)，每头 3 分支：
       output_block1   → [B, T', 2]   体坐标系速度 (vx, vy)
       output_block2   → [B, T', 3]   不确定性代理 σ（Stage 2B 使用）
       output_block1_z → [B, T', 1]   体坐标系速度 (vz / 前向速度)
```

> Unfold 窗口宽 13 决定 LSTM 输入维度（128 × 13 = 1664），原论文未提及。  
> 参数量 ~2.4M，checkpoint ~26 MB，单次前向 <2 ms（CPU）。

---

## 仓库结构

```
tartan/
├── checkpoint_24.pt            # 早期版本权重
├── checkpoint_28.pt            # 主权重（Base model, ~30 MB）
├── checkpoint_37.pt            # LoRA 微调版本权重
├── model.py                    # TartanIMU 模型定义（根目录主版本）
├── ekf_backend.py              # ESKF + 速度辅助 ESKF
├── evaluate_all.py             # 多平台基线评估（作者 NPZ 数据）
├── evaluate_real.py            # 多平台基线评估（自采数据）
├── evaluate_ekf.py             # EKF 融合评估
├── evaluate_ekf_real.py        # EKF 融合评估（自采数据）
├── evaluate_stage2b_real.py    # Stage 2B 在线评估
│
├── car/                        # 车辆平台：LoRA 训练 + 评估
│   ├── train_lora.py           #   LoRA / 车头训练脚本
│   ├── lora_trained.pt         #   LoRA 微调权重
│   ├── head_trained.pt         #   仅车头微调权重
│   ├── model.py                #   模型定义（同根目录）
│   ├── inference_my_car.py     #   自采车辆数据推理
│   ├── processed_car_data.py   #   数据预处理
│   ├── process_gt.py           #   真值处理
│   ├── validate_trajectory.py  #   轨迹验证
│   └── eval_*.png              #   评估结果图
│
├── test/car/                   # 车辆平台：LoRA 评估
│   ├── model.py                #   LoRA 版模型定义（含 load_dual_checkpoints）
│   ├── run_eval_lora.py        #   LoRA 评估脚本
│   └── eval_result_*.png       #   评估结果图
│
├── human/                      # 人体步态平台
│   ├── model.py                #   模型定义（同根目录）
│   ├── inference.py            #   推理脚本
│   ├── preprocess_stage1_waist.py  # 数据预处理
│   ├── verify_tartan.py        #   模型验证
│   └── *.png                   #   评估结果图
│
├── test/human/                 # 人体步态平台：辅助评估
│   ├── model.py                #   模型定义（同根目录）
│   ├── inference.py            #   推理与轨迹重建
│   └── result_*.png            #   评估结果图
│
├── Dataset_drone/              # 无人机平台（小规模预处理数据）
│   ├── model.py                #   模型定义（无人机特化版本）
│   ├── Preprocessing.py        #   mocap/IMU 数据对齐与预处理
│   ├── test_stage1*.py         #   Stage 1 推理
│   └── *.png                   #   评估结果图
│
├── 6-DOF Odometry/             # Lima 2019 基线对比
│   ├── model.py                #   Keras 模型定义
│   ├── model_torch.py          #   PyTorch 模型定义
│   ├── train.py                #   Keras 训练（OxIOD, 500 epochs）
│   ├── train_torch.py          #   PyTorch 训练脚本
│   ├── dataset.py              #   数据加载（OxIOD / EuRoC）
│   ├── test.py / test_torch.py #   评估与轨迹重建
│   ├── my_torch_model_*.pth    #   训练权重
│   └── result_*.png            #   评估结果图
│
├── results/                    # 各阶段综合评估结果图（本仓库 evaluate_*.py / EKF 脚本产出）
├── slides/                     # 汇报图表生成脚本（含 slide_figures/）
├── legacy/                     # 早期探索脚本（可选保留）
├── docs/                       # 参考论文 PDF
│
├── Drone_Dataset1/             # [未纳入 Git] 完整无人机数据（~36 GB）
└── .gitignore                  # 排除大文件与环境文件
```

> **注意**：`car/model.py`、`human/model.py`、`test/human/model.py` 与根目录 `model.py` 内容一致，各子目录保留副本以支持独立运行。`test/car/model.py` 为 LoRA 扩展版本，`Dataset_drone/model.py` 为无人机特化版本。

---

## 数据说明

| 平台 | 来源 | 原始频率 | 主要文件 |
|------|------|---------|---------|
| Car | 自采（道路驾驶） | 480 Hz IMU / 100 Hz GPS | `car/car_imu_data_full.csv` |
| Human | 自采（腰部 + 脚背） | ~205 Hz IMU / 100 Hz mocap | `human/4d91_long_*.xlsx` |
| Drone | Drone_Dataset1（未纳入 Git） | 500 Hz 同步 | `Drone_Dataset1/piloted/` |
| 所有（retargeted） | 作者格式 NPZ | 200 Hz | `test/*/pretrain_1.npz` |

所有数据推理前重采样至 200 Hz，四元数插值使用 SLERP。

