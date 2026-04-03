# 解决 mmcv CUDA 扩展兼容性问题

## 问题描述

服务器环境出现严重的 mmcv CUDA 扩展兼容性问题：
```
ImportError: /root/anaconda3/envs/bevfusion/lib/python3.8/site-packages/mmcv/_ext.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN2at5sliceERKNS_6TensorElN3c108optionalIlEES5_l
```

这是因为 mmcv-full 的预编译版本与服务器上的 PyTorch/CUDA 版本不匹配。

## 解决方案

### 方案 1：使用纯 PyTorch 实现（推荐）

我们已经创建了纯 PyTorch 的 Focal Loss 实现，不依赖 mmcv 的 CUDA 扩展。

**步骤：**

1. **在服务器上创建纯 PyTorch 损失函数文件**

```bash
cd /root/workbench/test_bevfusion

cat > mmdet3d/models/losses/pytorch_focal_loss.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchFocalLoss(nn.Module):
    """
    Pure PyTorch implementation of Focal Loss to avoid mmcv CUDA extension issues.
    """
    
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        if self.use_sigmoid:
            pred_sigmoid = pred.sigmoid()
            
            if target.dim() != pred.dim():
                target = F.one_hot(target, num_classes=pred.size(1)).float().transpose(1, -1)
            
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            loss = focal_weight * bce
        else:
            if target.dim() != pred.dim():
                target = target.long()
            else:
                target = target.argmax(dim=1)
            
            logpt = F.log_softmax(pred, dim=1)
            logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = logpt.exp()
            focal_weight = (1 - pt).pow(self.gamma)
            loss = -focal_weight * logpt
        
        if weight is not None:
            if weight.dim() != loss.dim():
                if weight.dim() == 1:
                    weight = weight.view(-1, 1)
                else:
                    weight = weight.unsqueeze(1)
            loss = loss * weight
        
        if reduction == 'none':
            pass
        elif reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight


from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class PyTorchFocalLossWrapper(PyTorchFocalLoss):
    """Wrapper for mmdet compatibility."""
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, **kwargs):
        super().__init__(
            use_sigmoid=use_sigmoid,
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            loss_weight=loss_weight
        )
EOF
```

2. **修改配置文件，使用 PyTorch 实现**

编辑配置文件（以 `distance_adaptive_voxel_scheme_a.yaml` 为例）：

```bash
# 备份原配置
cp configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml \
   configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml.bak

# 使用 sed 替换 FocalLoss 为 PyTorchFocalLossWrapper
sed -i 's/type: FocalLoss/type: PyTorchFocalLossWrapper/g' \
   configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml

# 同样替换 GaussianFocalLoss
sed -i 's/type: GaussianFocalLoss/type: PyTorchFocalLossWrapper/g' \
   configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml
```

3. **清理缓存并重新运行**

```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 重新运行训练
torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result
```

### 方案 2：完全重新编译 mmcv（备选）

如果方案 1 不可行，可以尝试从源码完全重新编译 mmcv：

```bash
cd /root/workbench/test_bevfusion

# 1. 完全卸载 mmcv
pip uninstall mmcv-full mmcv -y

# 2. 检查 PyTorch 和 CUDA 版本
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 3. 安装与 PyTorch 匹配的 mmcv
# 根据输出的 CUDA 版本选择：
# CUDA 11.1 + PyTorch 1.9:
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html

# CUDA 11.0 + PyTorch 1.9:
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.9/index.html

# CUDA 10.2 + PyTorch 1.9:
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9/index.html

# 4. 验证安装
python -c "from mmcv.ops import sigmoid_focal_loss; print('OK')"
```

### 方案 3：使用 Docker 容器（最彻底）

如果上述方案都失败，建议使用官方提供的 Docker 镜像：

```bash
# 在宿主机上执行
cd /root/workbench/test_bevfusion/docker
docker build . -t bevfusion

# 运行 Docker 容器
nvidia-docker run -it -v /root/workbench/test_bevfusion:/workspace --shm-size 16g bevfusion /bin/bash

# 在容器内运行训练
cd /workspace
torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --run-dir train_result
```

## 推荐方案

**强烈推荐使用方案 1**（纯 PyTorch 实现），原因：

1. ✅ 不依赖 mmcv 的 CUDA 扩展，避免兼容性问题
2. ✅ 可以在 CPU 和 GPU 上运行
3. ✅ 性能差异不大（focal loss 计算量相对较小）
4. ✅ 易于调试和维护
5. ✅ 未来 PyTorch 版本升级不会受到影响

## 注意事项

1. **GaussianFocalLoss** 也需要替换，如果有使用的话
2. 确保所有配置文件都使用新的 `PyTorchFocalLossWrapper`
3. 如果训练过程中出现其他 mmcv CUDA 扩展错误，请报告具体错误信息

## 验证

运行以下命令验证修复是否成功：

```bash
# 测试 PyTorch Focal Loss
python -c "from mmdet3d.models.losses.pytorch_focal_loss import PyTorchFocalLossWrapper; print('PyTorch Focal Loss OK')"

# 测试训练
torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --run-dir train_result_test
```

如果成功启动训练，说明修复完成！
