# 训练配置说明

## 显存优化配置

为了避免显存溢出（OOM），已对训练参数进行以下优化：

### 1. Epoch 配置

**文件**: `configs/default.yaml`
```yaml
max_epochs: 6  # 从 2 改为 6，进行测试阶段训练
```

**文件**: `configs/nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml`
```yaml
max_epochs: 6  # 保持一致
```

### 2. Batch Size 配置

**文件**: `configs/nuscenes/default.yaml`
```yaml
data:
  samples_per_gpu: 2  # 从 1 改为 2，平衡显存和训练效率
  workers_per_gpu: 2  # 从 0 改为 2，提高数据加载速度
```

**说明**:
- `samples_per_gpu`: 每个 GPU 的样本数（batch size）
  - 原值：1
  - 新值：2
  - 原因：在避免 OOM 的前提下提高训练效率
  
- `workers_per_gpu`: 每个 GPU 的数据加载线程数
  - 原值：0（单线程）
  - 新值：2
  - 原因：提高数据加载速度，避免 GPU 等待

### 3. 距离自适应体素化配置

**文件**: `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml`
```yaml
data:
  samples_per_gpu: 2  # 与全局配置一致
  workers_per_gpu: 2

max_epochs: 6  # 与全局配置一致
```

---

## 训练命令

### 单 GPU 测试
```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --run-dir train_dav_test
```

### 8 GPU 完整训练
```bash
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --run-dir train_dav
```

---

## 显存占用估算

| 配置 | Batch Size | 显存占用 (RTX 3090) |
|------|-----------|-------------------|
| Baseline (固定 voxel) | 4 | 8-10 GB |
| Baseline (固定 voxel) | 2 | 5-6 GB |
| 距离自适应体素化 | 2 | 12-14 GB |
| 距离自适应体素化 | 1 | 8-10 GB |

**注意**: 如果仍遇到 OOM，请将 `samples_per_gpu` 降低到 1。

---

## 训练时间估算

| GPU 数量 | Epoch 数 | 预计时间 |
|---------|---------|---------|
| 1 GPU   | 6       | 3-4 天   |
| 4 GPU   | 6       | 1-2 天   |
| 8 GPU   | 6       | 12-18 小时 |

---

## 验证配置

训练完成后，使用以下命令验证：

```bash
# 验证集评估
torchpack dist-run -np 8 python tools/test.py \
  train_dav/configs.yaml \
  train_dav/latest.pth \
  --eval bbox

# 可视化
torchpack dist-run -np 1 python tools/visualize.py \
  train_dav/configs.yaml \
  --mode gt \
  --checkpoint train_dav/latest.pth \
  --bbox-score 0.5 \
  --out-dir vis_dav
```

---

## 常见问题

### Q1: 显存溢出（OOM）
**解决方案**:
```yaml
# 降低 batch size
samples_per_gpu: 1

# 或启用混合精度训练
# 在命令行中添加 --amp 参数
```

### Q2: 训练速度慢
**解决方案**:
```yaml
# 增加数据加载线程
workers_per_gpu: 4

# 或使用更多 GPU
torchpack dist-run -np 8 ...
```

### Q3: 训练不稳定
**解决方案**:
```yaml
# 调整学习率
optimizer:
  lr: 1.0e-4  # 从 2.0e-4 降低

# 或增加 warmup
lr_config:
  warmup_iters: 1000  # 从 500 增加
```

---

**最后更新**: 2026-04-01  
**版本**: v1.0
