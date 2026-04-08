import torch
import os

def remap_convfuser_to_multiscale(state_dict):
    """
    将 ConvFuser (nn.Sequential) 权重重映射到 MultiScaleConvFuser 格式
    
    ConvFuser 结构:
        fuser.0.weight: Conv2d
        fuser.1.weight, fuser.1.bias: BatchNorm2d
        fuser.1.running_mean, fuser.1.running_var: BatchNorm2d statistics
        (ReLU 没有参数)
    
    MultiScaleConvFuser 结构:
        fuser.fusion_conv.0.weight: Conv2d
        fuser.fusion_conv.1.weight, fuser.fusion_conv.1.bias: BatchNorm2d
        fuser.fusion_conv.1.running_mean, fuser.fusion_conv.1.running_var: BatchNorm2d statistics
        fuser.channel_attention.*: 通道注意力层
        fuser.refine_conv.*: 细化卷积层
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # 重映射 fuser 相关键
        if key.startswith('fuser.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                # 这是 fuser.0.weight 或 fuser.1.* 格式
                idx = int(parts[1])
                if idx == 0:  # Conv2d
                    new_key = 'fuser.fusion_conv.0.' + '.'.join(parts[2:])
                elif idx == 1:  # BatchNorm2d
                    new_key = 'fuser.fusion_conv.1.' + '.'.join(parts[2:])
                else:
                    # ReLU 层 (idx==2) 没有参数，跳过
                    continue
                new_state_dict[new_key] = value
                print(f"重映射: {key} -> {new_key}")
            else:
                # 其他 fuser 键，直接保留
                new_state_dict[key] = value
        else:
            # 非 fuser 键，直接保留
            new_state_dict[key] = value
    
    return new_state_dict

def main():
    checkpoint_path = 'train_result/latest.pth'
    output_path = 'train_result/latest_remapped.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 权重文件不存在: {checkpoint_path}")
        return
    
    print(f"加载权重文件: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' not in checkpoint:
        print("错误: checkpoint 中没有 state_dict")
        return
    
    print("原始 state_dict 键 (fuser 相关):")
    fuser_keys = [k for k in checkpoint['state_dict'].keys() if 'fuser' in k]
    for key in fuser_keys:
        print(f"  {key}")
    
    print("\n开始重映射权重...")
    checkpoint['state_dict'] = remap_convfuser_to_multiscale(checkpoint['state_dict'])
    
    print("\n重映射后的 state_dict 键 (fuser 相关):")
    fuser_keys = [k for k in checkpoint['state_dict'].keys() if 'fuser' in k]
    for key in fuser_keys:
        print(f"  {key}")
    
    # 保存重映射后的权重
    torch.save(checkpoint, output_path)
    print(f"\n重映射后的权重已保存到: {output_path}")
    
    # 验证重映射
    print("\n验证重映射:")
    loaded = torch.load(output_path, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    
    # 模拟 MultiScaleConvFuser 期望的键
    expected_keys = [
        'fuser.fusion_conv.0.weight',
        'fuser.fusion_conv.1.weight',
        'fuser.fusion_conv.1.bias',
        'fuser.fusion_conv.1.running_mean',
        'fuser.fusion_conv.1.running_var',
    ]
    
    for key in expected_keys:
        if key not in loaded['state_dict']:
            missing_keys.append(key)
    
    for key in loaded['state_dict'].keys():
        if key.startswith('fuser.') and key not in expected_keys:
            if not (key.startswith('fuser.fusion_conv.') or 
                    key.startswith('fuser.channel_attention.') or 
                    key.startswith('fuser.refine_conv.')):
                unexpected_keys.append(key)
    
    if missing_keys:
        print(f"警告: 仍缺少以下键: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 仍有意外键: {unexpected_keys}")
    
    if not missing_keys and not unexpected_keys:
        print("✓ 重映射成功，所有期望的键都存在")
    else:
        print("⚠ 重映射可能不完全，但已改善权重兼容性")

if __name__ == '__main__':
    main()