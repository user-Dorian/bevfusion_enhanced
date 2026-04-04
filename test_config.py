import sys
sys.path.insert(0, 'd:/workbench/bev/bevfusion_enhanced_2')

from mmcv import Config

cfg = Config.fromfile('d:/workbench/bev/bevfusion_enhanced_2/configs/nuscenes/det/centerhead/lssfpn/camera+radar/resnet50/dlss.yaml')

print("配置加载成功！")
print(f"vtransform 类型：{cfg.model.encoders.camera.vtransform.type}")
print(f"depth_input: {cfg.model.encoders.camera.vtransform.get('depth_input', '未设置')}")
print(f"add_depth_features: {cfg.model.encoders.camera.vtransform.get('add_depth_features', '未设置')}")
print(f"use_points: {cfg.model.encoders.camera.vtransform.get('use_points', '未设置')}")
print(f"height_expand: {cfg.model.encoders.camera.vtransform.get('height_expand', '未设置')}")

# 检查继承的 dbound
if hasattr(cfg.model.encoders.camera.vtransform, 'dbound'):
    print(f"dbound: {cfg.model.encoders.camera.vtransform.dbound}")
else:
    print("dbound: 未在当前配置中设置（可能从基类继承）")
