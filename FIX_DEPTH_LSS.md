# DepthLSSTransform 修复说明

## 问题描述

训练时出现以下错误：
```
RuntimeError: Given groups=1, weight of size [8, 1, 1, 1], expected input[12, 6, 256, 704] to have 1 channels, but got 6 channels instead
```

## 问题原因

`DepthLSSTransform` 的 `dtransform` 模块硬编码为 `nn.Conv2d(1, 8, 1)`，期望输入 1 个通道。

但实际运行时，`BaseDepthTransform.forward` 方法会根据配置创建深度图：
- 当 `depth_input='scalar'` 且 `add_depth_features=True` 时
- 深度图通道数 = 1 + point_feature_dims
- 对于 LiDAR 点，point_feature_dims=5，所以总通道数=6

这导致通道数不匹配。

## 修复方案

修改 `mmdet3d/models/vtransforms/depth_lss.py`：

1. 添加参数支持：
   - `depth_input` (默认：'scalar')
   - `add_depth_features` (默认：True，与 BaseDepthTransform 一致)
   - `height_expand` (默认：True)
   - `point_feature_dims` (默认：5，适用于 LiDAR 点)

2. 动态计算 `dtransform` 的输入通道数：
   ```python
   depth_in_channels = 1 if depth_input == 'scalar' else self.D
   if add_depth_features:
       depth_in_channels += point_feature_dims
   ```

3. 使用计算出的通道数创建 `dtransform`：
   ```python
   self.dtransform = nn.Sequential(
       nn.Conv2d(depth_in_channels, 8, 1),
       ...
   )
   ```

## 配置说明

### 使用 LiDAR 点（默认）
```yaml
model:
  encoders:
    camera:
      vtransform:
        type: DepthLSSTransform
        # 以下参数使用默认值即可
        depth_input: 'scalar'  # 默认
        add_depth_features: true  # 默认
        height_expand: true  # 默认
        point_feature_dims: 5  # LiDAR 点维度：x, y, z, 强度，其他
```

### 使用雷达点
```yaml
model:
  encoders:
    camera:
      vtransform:
        type: DepthLSSTransform
        depth_input: 'one-hot'  # 或 'scalar'
        add_depth_features: true
        height_expand: true
        point_feature_dims: 45  # 雷达点维度（根据 radar_use_dims）
        use_points: radar  # 在 BaseDepthTransform 中定义
```

## 训练命令

使用原始训练命令即可：

```bash
# Camera + LiDAR 训练
torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result

# Camera + Radar 训练（需要修改配置）
torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera+radar/resnet50/dlss.yaml \
  --model.encoders.camera.vtransform.point_feature_dims 45 \
  --run-dir train_result
```

## 测试命令

```bash
# 测试
torchpack dist-run -np 1 python tools/test.py train_result/configs.yaml train_result/latest.pth \
  --eval bbox \
  --out box.pkl
```

## 验证修复

运行测试脚本：
```bash
python test_depth_lss_fix.py
```

预期输出：
```
创建 DepthLSSTransform（默认参数，LiDAR 点）...
✓ 模型创建成功
  depth_input: scalar
  add_depth_features: True
  height_expand: True
  D (depth bins): 118
  dtransform 输入通道数：6
  预期通道数：1 (depth_input='scalar') + 5 (point_feature_dims) = 6

创建 DepthLSSTransform（雷达点配置）...
✓ 模型创建成功
  depth_input: one-hot
  add_depth_features: True
  D (depth bins): 118
  dtransform 输入通道数：163
  预期通道数：118 (depth_input='one-hot') + 45 (point_feature_dims) = 163

✓ 所有测试通过！
```

## 文件修改清单

- `mmdet3d/models/vtransforms/depth_lss.py`: 添加参数支持，动态计算通道数

## 兼容性

- ✅ 向后兼容：默认参数适用于 LiDAR 点配置
- ✅ 前向兼容：支持雷达点配置（通过显式设置参数）
- ✅ 配置灵活：可通过配置文件或命令行覆盖参数

## 注意事项

1. **雷达点维度**：根据 `radar_use_dims` 配置确定，通常是 45 维
2. **LiDAR 点维度**：根据 `use_dim` 配置确定，通常是 5 维
3. **depth_input 类型**：
   - `'scalar'`: 深度值为标量，通道数=1
   - `'one-hot'`: 深度值为 one-hot 编码，通道数=D（深度 bin 数量）
4. **add_depth_features**：是否添加点特征到深度图，默认为 True
5. **height_expand**：是否在高度方向扩展点，默认为 True

## 相关文档

- 距离自适应体素化方案：`DISTANCE_ADAPTIVE_VOXELIZATION_PROPOSAL.md`
- 原始 BEVFusion 论文：https://github.com/mit-han-lab/bevfusion
