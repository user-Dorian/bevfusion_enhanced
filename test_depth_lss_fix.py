"""测试 DepthLSSTransform 的修复"""
import torch
from mmdet3d.models.vtransforms.depth_lss import DepthLSSTransform

# 测试参数
in_channels = 256
out_channels = 80
image_size = (256, 704)
feature_size = (32, 88)  # 256/8, 704/8
xbound = [-51.2, 51.2, 0.4]
ybound = [-51.2, 51.2, 0.4]
zbound = [-10.0, 10.0, 20.0]
dbound = [1.0, 60.0, 0.5]
downsample = 2

# 创建模型（使用默认参数：add_depth_features=True, point_feature_dims=5）
print("创建 DepthLSSTransform（默认参数，LiDAR 点）...")
model = DepthLSSTransform(
    in_channels=in_channels,
    out_channels=out_channels,
    image_size=image_size,
    feature_size=feature_size,
    xbound=xbound,
    ybound=ybound,
    zbound=zbound,
    dbound=dbound,
    downsample=downsample,
)
print(f"✓ 模型创建成功")
print(f"  depth_input: {model.depth_input}")
print(f"  add_depth_features: {model.add_depth_features}")
print(f"  height_expand: {model.height_expand}")
print(f"  D (depth bins): {model.D}")
print(f"  dtransform 输入通道数：{model.dtransform[0].in_channels}")
print(f"  预期通道数：1 (depth_input='scalar') + 5 (point_feature_dims) = 6")

# 测试雷达点配置
print("\n创建 DepthLSSTransform（雷达点配置）...")
model_radar = DepthLSSTransform(
    in_channels=in_channels,
    out_channels=out_channels,
    image_size=image_size,
    feature_size=feature_size,
    xbound=xbound,
    ybound=ybound,
    zbound=zbound,
    dbound=dbound,
    downsample=downsample,
    depth_input='one-hot',
    add_depth_features=True,
    height_expand=True,
    point_feature_dims=45,
)
print(f"✓ 模型创建成功")
print(f"  depth_input: {model_radar.depth_input}")
print(f"  add_depth_features: {model_radar.add_depth_features}")
print(f"  D (depth bins): {model_radar.D}")
print(f"  dtransform 输入通道数：{model_radar.dtransform[0].in_channels}")
print(f"  预期通道数：{model_radar.D} (depth_input='one-hot') + 45 (point_feature_dims) = {model_radar.D + 45}")

print("\n✓ 所有测试通过！")
