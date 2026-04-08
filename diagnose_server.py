#!/usr/bin/env python
"""
服务器端诊断脚本：检查当前代码构建的模型结构
"""
import torch
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

def main():
    config_path = 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
    
    print(f"加载配置：{config_path}")
    configs.load(config_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_path)
    
    print("\n构建模型...")
    model = build_model(cfg.model)
    
    print("\n模型 fuser 类型:", type(model.fuser).__name__)
    print("\nFuser 结构详情:")
    if hasattr(model.fuser, '__dict__'):
        print("  属性:", list(model.fuser.__dict__.keys()))
    
    print("\nFuser 参数键名:")
    for name, param in model.fuser.named_parameters():
        print(f"  fuser.{name}")
    
    print("\n尝试加载权重...")
    checkpoint = torch.load('train_result/latest.pth', map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    print("\n权重文件中的 fuser 键名 (前 10 个):")
    fuser_keys = [k for k in state_dict.keys() if 'fuser' in k][:10]
    for key in fuser_keys:
        print(f"  {key}")
    
    print("\n对比模型与权重:")
    model_keys = set(f"fuser.{name}" for name, _ in model.fuser.named_parameters())
    weight_keys = set(k for k in state_dict.keys() if 'fuser' in k)
    
    matching = model_keys & weight_keys
    missing_in_weight = model_keys - weight_keys
    unexpected_in_weight = weight_keys - model_keys
    
    print(f"  匹配的键：{len(matching)}")
    print(f"  模型期望但权重中没有：{len(missing_in_weight)}")
    if missing_in_weight:
        for key in list(missing_in_weight)[:5]:
            print(f"    - {key}")
    
    print(f"  权重中有但模型不需要：{len(unexpected_in_weight)}")
    if unexpected_in_weight:
        for key in list(unexpected_in_weight)[:5]:
            print(f"    - {key}")

if __name__ == '__main__':
    main()