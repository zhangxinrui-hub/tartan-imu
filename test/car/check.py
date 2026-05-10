import torch

ckpt_path = 'checkpoint_37.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu')

# 打开 'model_state_dict' 这一层
if 'model_state_dict' in checkpoint:
    model_weights = checkpoint['model_state_dict']
    print(f"成功解包 model_state_dict，包含 {len(model_weights)} 个参数")
    
    # 检查 LoRA 关键字
    keys = list(model_weights.keys())
    lora_keys = [k for k in keys if 'lora' in k]
    
    if len(lora_keys) > 0:
        print(f"[OK] 确认：这是一个 LoRA 微调模型！")
        print(f"发现 {len(lora_keys)} 个 LoRA 参数，例如：{lora_keys[:3]}")
    else:
        print(f"未发现 LoRA 关键字，这可能是一个全量微调 (Full Fine-tune) 模型或 Base 模型。")
        print(f"前 5 个参数名：{keys[:5]}")
else:
    print("[ERROR] 结构异常：未找到 model_state_dict")