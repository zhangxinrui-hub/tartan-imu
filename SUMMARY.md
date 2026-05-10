# Project Summary — TartanIMU Multi-Platform Evaluation

## What This Project Does

This repository takes the TartanIMU foundation model (Zhao et al., CVPR 2025) and builds a complete evaluation and deployment pipeline around it:

1. **Model reverse-engineering** — Reconstructed the full architecture (CNN + LSTM + 6-layer Transformer + multi-class heads) from a PyTorch checkpoint, achieving strict `load_state_dict` compatibility.
2. **Multi-platform evaluation** — Systematic ATE evaluation on car, human, and drone data, using both author-format retargeted data and self-collected real-world data.
3. **EKF sensor fusion** — Implemented an error-state EKF for orientation estimation and a velocity-aided EKF for full pose estimation, progressively removing GT dependency.
4. **LoRA domain adaptation** — Applied low-rank adapters (LoRA) to the CNN backbone for car-specific fine-tuning.

## Architecture Details

The model was reverse-engineered from `checkpoint_28.pt` (primary) and verified against `checkpoint_24.pt` (earlier version).

```
Input [B, 6, T] (acc + gyro, 200 Hz, gravity-removed, bias-corrected, acc/9.81)
  │
  ├─ CNN Backbone: Conv1d(6→64, k=7) + 3 groups × 2 ResBlocks (64→128→256, stride-2 downsample)
  ├─ Post-processing: Conv1d 256→128→128 (1×1 kernels)
  ├─ Unfold: sliding window (win=13, step=1) → [B, T', 1664]
  ├─ LSTM: input=1664, hidden=64 (supports hidden state carry-over for streaming)
  ├─ IMU_Trunk: 6× Transformer blocks (d=64, heads=4, ff=256, pre-norm, add_bias_kv=True)
  │
  └─ 4 Output Heads (dog / human / car / drone), each with 3 MLP branches:
       output_block1   → [B, T', 2]   body-frame velocity (vx, vy)
       output_block2   → [B, T', 3]   uncertainty proxy (σ, used in Stage 2B)
       output_block1_z → [B, T', 1]   body-frame velocity (vz or forward speed)
```

**Key discovery**: The unfold operation with window size 13 explains the LSTM input dimension (128 channels × 13 = 1664). This was not documented in the original paper.

**Parameters**: ~2.4M | **Checkpoint size**: ~26 MB | **Inference**: <2ms per forward pass (CPU)

## Evaluation Stages

### Stage 1 — Baseline

**Scripts**: `evaluate_all.py` (retargeted NPZ), `evaluate_real.py` (real data)

Pipeline: raw IMU → preprocess (GT-quat gravity removal + bias correction + /9.81) → TartanIMU → body-frame velocity → GT-quat body→world rotation → integrate → ATE.

**Limitation**: Both gravity removal and body→world rotation use GT quaternion. This is an oracle upper bound.

Velocity assembly per platform:
- **Car**: forward speed from `output_block1_z`, lateral/vertical zeroed (non-holonomic constraint)
- **Human / Drone**: full 3D from `[output_block1[:, 0:2], output_block1_z]`

ATE computed as RMSE without Sim3 scale alignment.

### Stage 2 — EKF Orientation Ablation

**Scripts**: `evaluate_ekf.py` (retargeted NPZ), `evaluate_ekf_real.py` (real data)

Compares three orientation sources for the body→world rotation step:
- **GT-quat**: Oracle upper bound
- **EKF**: Gyro propagation + accelerometer gravity update (pitch/roll observable, yaw drifts)
- **EKF + calibrated bias**: Offline gyro mean as initial bias (valid for car, invalid for human/drone)

The preprocessing (gravity removal) still uses GT quaternion in this stage — only the body→world rotation changes.

### Stage 2B — Velocity-Aided EKF (Deployment-Realistic)

**Script**: `evaluate_stage2b_real.py`

Full online pipeline with NO GT in the estimation loop:
1. Raw IMU → **orientation ESKF** → estimated quaternion
2. Estimated quaternion → gravity removal → TartanIMU input
3. TartanIMU → velocity `v_hat` + uncertainty proxy `σ_hat` (from `output_block2`)
4. `VelocityAidedESKF` (15-DOF): IMU propagation + neural velocity updates with learned noise → position/velocity/attitude

GT is used ONLY for offline ATE computation.

### LoRA Fine-Tuning

**Scripts**: `car/train_lora.py` (training), `test/car/model.py` (LoRA architecture), `test/car/run_eval_lora.py` (evaluation)

- `LoRAConv1d`: Low-rank delta on Conv1d weights (rank=32, alpha=8, scaling=0.25)
- Applied to all convolutions in the CNN backbone (input block + residual blocks)
- Training: random short windows on self-collected car data, MSE loss on `block1_z` (forward speed), AdamW optimizer with gradient clipping
- Two modes: `--train-head` (only update `heads.car`) or default (only update LoRA adapters)
- Dual checkpoint loading at evaluation: base `checkpoint_28.pt` + LoRA `car/lora_trained.pt`

## EKF Backend (`ekf_backend.py`)

### ESKF (Orientation Only)

- **State**: quaternion (world-from-body) + gyro bias → 6-DOF error state
- **Predict**: gyro-driven quaternion integration
- **Update**: accelerometer as gravity sensor (pitch/roll correction; yaw unobservable without magnetometer)
- **Gate**: rejects accelerometer updates when `|‖a‖ − 9.81| / 9.81 > threshold` (dynamic motion rejection)

### VelocityAidedESKF (Full Navigation)

- **State**: position + velocity + quaternion + gyro bias + acc bias → 15-DOF error state
- **Predict**: full IMU mechanization (a_world = R(q)·f_b − g)
- **Update**: neural body-frame velocity with `sigma_to_std` noise mapping:
  `std = σ_floor + σ_scale × exp(clamp(proxy))`
- **Joseph-form** covariance update for numerical stability

## Data Sources

| Platform | Source | Raw frequency | Files |
|----------|--------|---------------|-------|
| Car | Self-collected (road driving) | 480 Hz IMU, 100 Hz GPS | `car/car_imu_data_full.csv`, `car/car_ground_truth.csv` |
| Human | Self-collected (waist + instep) | ~205 Hz IMU, 100 Hz mocap | `human/4d91_long_*.xlsx` |
| Drone | Drone_Dataset1 (piloted + autonomous) | 500 Hz synced | `Drone_Dataset1/piloted/`, `Drone_Dataset1/autonomous/` |
| All (retargeted) | Author-format NPZ | 200 Hz | `test/car/pretrain_1.npz`, `test/human/pretrain_1.npz`, `Dataset_drone/*.npy` |

All data resampled to 200 Hz before inference. Quaternion interpolation uses SLERP.

## Preprocessing Details

### Standard Path (`preprocess_imu_with_gravity`)

```
1. g_body = R(gt_quat)⁻¹ · [0, 0, 9.81]     # gravity in body frame
2. acc_bias = mean(acc_raw[:200] − g_body[:200])
3. acc_net = acc_raw − acc_bias − g_body        # specific force (gravity removed)
4. gyro -= mean(gyro[:200])                      # bias removal
5. [optional] FRD → FLU (negate Y, Z)           # drone only
6. acc_net /= 9.81                               # normalize
```

### Human Real-Data Shortcut (`preprocess_human_real`)

Axis remap (user Z→model X, user Y→model Y, user X→model Z) then `acc[:, 2] -= 9.81`. Assumes near-upright orientation — no full attitude required.

## 6-DOF Inertial Odometry Baseline (`6-DOF Odometry/`)

A re-implementation of Lima et al. (Sensors 2019) serving as a traditional deep-learning baseline for comparison with TartanIMU.

### Architecture (`model_torch.py`)

CNN-LSTM network: separate gyro/acc encoders (Conv1d 3→64→128, k=11) → concatenate → 2-layer bidirectional LSTM (hidden=128) → FC heads for relative translation (3) and quaternion (4). Multi-task loss with learned task weights (TMAE + QME).

### Training (`train.py`)

- **Dataset**: Oxford Inertial Odometry Dataset (OxIOD), 17 handheld sequences
- **Config**: window_size=200, stride=10, batch_size=32, 500 epochs, Adam lr=1e-4
- **Output**: `model_checkpoint.hdf5` (best val loss), `my_model.hdf5` (final pred model)
- Training loss curve saved to `training_loss.png`
- Keras implementation (CuDNNLSTM → LSTM fallback for CPU compatibility)

### Evaluation (`test.py`, `test_torch.py`)

Reconstructs full trajectory from predicted relative poses. Results visualized in `result_*.png`.

**Note**: The primary training was done with the Keras pipeline (`train.py`). A fully runnable PyTorch version (`train_torch.py`) is also provided, reusing `model_torch.py` and `dataset.py`.

## Known Limitations

- Baseline evaluation uses GT quaternion for preprocessing — results are oracle upper bounds
- `output_block2` is used as uncertainty proxy but was not trained with NLL loss (assumption, not verified)
- EKF runs as Python for-loop (slow for long sequences)
- No comparison with other learning-based INS methods (RoNIN, IONet, etc.)
- `dog` output head has no evaluation data or pipeline

## Checkpoints

| File | Role | Used by |
|------|------|---------|
| `checkpoint_28.pt` | Base model (primary) | All evaluation scripts |
| `checkpoint_37.pt` | LoRA fine-tuned (car) | `test/car/run_eval_lora.py` |
| `checkpoint_24.pt` | Earlier base model (legacy) | Some `Dataset_drone/` scripts |
