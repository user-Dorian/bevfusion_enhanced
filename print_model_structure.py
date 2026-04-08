"""直接打印模型结构"""
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model

config_path = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"

print("=" * 80)
print("构建模型并打印结构")
print("=" * 80)

configs.load(config_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=config_path)

# 构建模型
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

print("\n模型结构:")
print(model)

print("\n\nFuser 详细结构:")
print(model.fuser)

print("\n\nFuser 类型:", type(model.fuser))
print("Fuser 父类:", model.fuser.__class__.__bases__)

if hasattr(model.fuser, 'fusion_conv'):
    print("\n⚠️  发现 fusion_conv 属性！")
    print("fusion_conv 类型:", type(model.fuser.fusion_conv))
    
if hasattr(model.fuser, '0'):
    print("\n✅ 发现索引 '0' 属性（Sequential 特征）")
    print("0 类型:", type(model.fuser[0]))

print("=" * 80)
