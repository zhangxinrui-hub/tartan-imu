import torch
from model import TartanIMUModel, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 创建模型并加载 checkpoint
model = TartanIMUModel().to(device)
model = load_checkpoint(model, "checkpoint_28.pt", device=device)

# 2. 构造一个假的 IMU batch（先别急着用真数据）
# shape: [B, 6, T]，T=200
imu_batch = torch.randn(1, 6, 200).to(device)

# 3. 推理
model.eval()
with torch.no_grad():
    out = model(imu_batch)

# 4. 取 car head
v_xy, v_xyz, v_z = out["car"]

# 5. 明确打印结果（否则你什么都看不到）
print("=== Forward result ===")
print("v_xy shape :", v_xy.shape)
print("v_xyz shape:", v_xyz.shape)
print("v_z shape  :", v_z.shape)

print("\nSample v_xyz (first 3 timesteps):")
print(v_xyz[0, :3])
