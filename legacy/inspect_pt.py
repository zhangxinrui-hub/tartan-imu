import torch

pt_path = 'checkpoint_24.pt'  # 换成你的文件路径

# 加载 checkpoint
checkpoint = torch.load(pt_path, map_location='cpu')

# 获取 model_state_dict（适配不同项目格式）
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    print("找不到模型参数字典")
    exit()

# 输出所有参数名和shape
print("%-60s  %20s" % ("parameter name", "shape"))
print("-" * 85)
for k, v in state_dict.items():
    try:
        shape = tuple(v.shape)
    except AttributeError:
        shape = type(v)
    print("%-60s  %20s" % (k, shape))