# BEVFusion 项目代码结构规范文档

本文档详细说明了 BEVFusion 项目的代码组织结构、模块注册机制、导入规范和配置文件语法，确保新建文件完全符合项目现有架构。

---

## 目录

1. [项目整体架构](#1-项目整体架构)
2. [模块注册机制](#2-模块注册机制)
3. [导入规范](#3-导入规范)
4. [配置文件规范](#4-配置文件规范)
5. [文件命名与组织规范](#5-文件命名与组织规范)
6. [新建文件检查清单](#6-新建文件检查清单)

---

## 1. 项目整体架构

### 1.1 核心目录结构

```
bevfusion_enhanced/
├── mmdet3d/                          # 核心代码包
│   ├── apis/                         # 训练/测试 API
│   ├── core/                         # 核心组件（bbox、points、voxel 等）
│   ├── datasets/                     # 数据集和数据处理 pipelines
│   ├── models/                       # 模型定义（重点）
│   │   ├── backbones/                # 骨干网络
│   │   ├── necks/                    # 颈部网络
│   │   ├── heads/                    # 检测头/分割头
│   │   ├── fusion_models/            # 融合模型
│   │   ├── fusers/                   # 融合模块
│   │   ├── vtransforms/              # 视角转换模块
│   │   ├── losses/                   # 损失函数
│   │   └── utils/                    # 模型工具
│   ├── ops/                          # 自定义算子（CUDA/C++ 扩展）
│   │   ├── voxel/                    # 体素化算子
│   │   ├── spconv/                   # 稀疏卷积
│   │   ├── bev_pool/                 # BEV 池化
│   │   ├── ball_query/               # 球查询
│   │   ├── ...                       # 其他算子
│   ├── runner/                       # 训练 Runner
│   └── utils/                        # 工具函数
├── configs/                          # 配置文件
│   ├── nuscenes/                     # nuScenes 数据集配置
│   │   ├── det/                      # 检测任务
│   │   └── seg/                      # 分割任务
│   └── default.yaml                  # 默认配置
└── tools/                            # 工具脚本
```

### 1.2 包导入层级

```python
# Level 1: 顶层包导入
from mmdet3d import models
from mmdet3d import ops
from mmdet3d import datasets

# Level 2: 子模块导入
from mmdet3d.models import builder
from mmdet3d.ops import voxel

# Level 3: 具体类/函数导入
from mmdet3d.models.builder import build_backbone
from mmdet3d.ops.voxel import Voxelization
```

---

## 2. 模块注册机制

### 2.1 Registry 注册器定义

项目在 [`mmdet3d/models/builder.py`](d:\workbench\bev\bevfusion_enhanced\mmdet3d\models\builder.py) 中定义了多个注册器：

```python
from mmcv.utils import Registry
from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS

# 自定义注册器
FUSIONMODELS = Registry("fusion_models")
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")
```

**注册器层级：**
- `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`：继承自 MMDetection
- `FUSIONMODELS`, `VTRANSFORMS`, `FUSERS`：BEVFusion 自定义

### 2.2 模块注册方式

#### 方式 1：装饰器注册（推荐）

```python
from mmdet.models import BACKBONES

@BACKBONES.register_module()
class MyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x
```

#### 方式 2：模块级自动注册

在 [`mmdet3d/models/backbones/__init__.py`](d:\workbench\bev\bevfusion_enhanced\mmdet3d\models\backbones\__init__.py) 中使用：

```python
from .resnet import *
from .second import *
from .sparse_encoder import *

__all__ = ["GeneralizedResNet", "SparseEncoder", ...]
```

**关键点：**
1. 使用 `*` 导入会自动导出模块中 `__all__` 定义的所有内容
2. 被导入的文件（如 `resnet.py`）必须自己定义 `__all__`
3. 装饰器注册在模块导入时自动执行

### 2.3 各模块注册位置

| 模块类型 | 注册器 | 文件路径 | 导出方式 |
|---------|--------|---------|---------|
| Backbones | `BACKBONES` | `mmdet3d/models/backbones/` | `from .xxx import *` |
| Necks | `NECKS` | `mmdet3d/models/necks/` | `from .xxx import *` |
| Heads | `HEADS` | `mmdet3d/models/heads/` | 分层导出 |
| Fusion Models | `FUSIONMODELS` | `mmdet3d/models/fusion_models/` | `from .xxx import *` |
| Fusers | `FUSERS` | `mmdet3d/models/fusers/` | `from .xxx import *` |
| VTransforms | `VTRANSFORMS` | `mmdet3d/models/vtransforms/` | `from .xxx import *` |
| Ops | - | `mmdet3d/ops/` | 直接导出函数/类 |

### 2.4 Ops 模块特殊处理

自定义算子（CUDA/C++ 扩展）在 [`mmdet3d/ops/__init__.py`](d:\workbench\bev\bevfusion_enhanced\mmdet3d\ops\__init__.py) 中统一导出：

```python
from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization
from .bev_pool import *
from .spconv import *

__all__ = [
    "Voxelization", "voxelization",
    "DynamicScatter", "dynamic_scatter",
    "bev_pool",
    ...
]
```

**注意：**
- 算子模块使用 `from .module import *` 或显式导入
- 必须在 `__all__` 中明确列出所有导出项
- CUDA 扩展在 [`setup.py`](d:\workbench\bev\bevfusion_enhanced\setup.py) 中编译

---

## 3. 导入规范

### 3.1 标准导入顺序

```python
# 1. Python 标准库
import os
import random
from typing import Any, Dict, List, Tuple

# 2. 第三方库
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32

# 3. MM 系列框架（MMDet, MMSeg）
from mmdet.models import BACKBONES, HEADS
from mmdet.models.builder import build_backbone

# 4. 项目内部导入（绝对路径）
from mmdet3d.models.builder import FUSIONMODELS, build_fuser
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models.backbones import SparseEncoder

# 5. 相对导入（仅限同包内）
from .base import Base3DFusionModel
from ..builder import build_model
```

### 3.2 绝对导入 vs 相对导入

**推荐：使用绝对导入**
```python
# ✓ 推荐
from mmdet3d.ops import Voxelization
from mmdet3d.models.builder import build_backbone

# ✗ 避免（除非在同包内）
from ...ops import Voxelization
```

**例外：子模块导入父模块或兄弟模块**
```python
# 在 mmdet3d/models/fusion_models/bevfusion.py 中
from .base import Base3DFusionModel  # ✓ 同包内相对导入
from ..builder import build_backbone  # ✓ 父包导入
```

### 3.3 __init__.py 导出规范

每个 `__init__.py` 必须定义 `__all__`：

```python
# mmdet3d/models/backbones/__init__.py
from .resnet import *
from .sparse_encoder import *

__all__ = [
    "GeneralizedResNet",
    "SparseEncoder",
    # ... 所有导出的类名
]
```

**关键规则：**
1. 使用 `from .module import *` 时，被导入模块必须有 `__all__`
2. `__init__.py` 的 `__all__` 应包含所有需要公开导出的内容
3. 避免循环导入：不要在 A 的 `__init__.py` 中导入 B，同时 B 的 `__init__.py` 导入 A

### 3.4 bevfusion.py 加载模块示例

[`mmdet3d/models/fusion_models/bevfusion.py`](d:\workbench\bev\bevfusion_enhanced\mmdet3d\models\fusion_models\bevfusion.py) 展示了标准加载方式：

```python
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel

@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(self, encoders, fuser, decoder, heads, **kwargs):
        super().__init__()
        
        # 通过 builder 动态创建模块
        self.encoders["camera"] = nn.ModuleDict({
            "backbone": build_backbone(encoders["camera"]["backbone"]),
            "neck": build_neck(encoders["camera"]["neck"]),
            "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
        })
        
        # 直接实例化 ops（不需要注册）
        voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
```

---

## 4. 配置文件规范

### 4.1 YAML 基础语法

配置文件使用 YAML 格式，支持嵌套结构和列表：

```yaml
model:
  encoders:
    camera:
      backbone:
        type: ResNet
        depth: 50
      neck:
        type: GeneralizedLSSFPN
        in_channels: [2048]
      vtransform:
        type: AwareBEVDepth
        in_channels: 512
        out_channels: 64
    lidar:
      voxelize:
        voxel_size: [0.075, 0.075, 0.2]
        point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        max_num_points: -1  # -1 表示动态体素化
      backbone:
        type: SparseEncoder
        in_channels: 4
        sparse_shape: [416, 416, 40]
  
  fuser:
    type: ConvFuser
    in_channels: [80, 256]
    out_channels: 256
  
  decoder:
    backbone:
      type: GeneralizedResNet
      in_channels: 256
      blocks: [[3, 256, 2], [3, 512, 2]]
    neck:
      type: FPN
      in_channels: [256, 512]
  
  heads:
    object:
      type: CenterHead
      in_channels: 256
```

### 4.2 变量引用语法 `${...}`

项目支持通过 `${...}` 语法引用配置中的其他变量：

**示例 1：列表计算**
```yaml
model:
  encoders:
    camera:
      vtransform:
        type: AwareBEVDepth
        feature_size: ${[image_size[0] // 16, image_size[1] // 16]}
        # 假设 image_size = [256, 704]
        # 结果：feature_size = [16, 44]
```

**示例 2：引用顶层配置**
```yaml
# configs/default.yaml
max_epochs: 20

runner:
  type: CustomEpochBasedRunner
  max_epochs: ${max_epochs}  # 引用顶层变量
```

**实现原理：**
在 [`mmdet3d/utils/config.py`](d:\workbench\bev\bevfusion_enhanced\mmdet3d\utils\config.py) 中通过 `recursive_eval` 实现：

```python
def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)
    
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)  # 执行 Python 表达式
        obj = recursive_eval(obj, globals)
    
    return obj
```

**支持的表达式：**
- 列表索引：`${image_size[0]}`
- 算术运算：`${image_size[0] // 16}`
- 列表推导：`${[x for x in range(10)]}`
- 任何合法的 Python 表达式

### 4.3 配置文件加载流程

在 [`tools/train.py`](d:\workbench\bev\bevfusion_enhanced\tools\train.py) 中：

```python
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval

# 1. 加载配置文件（支持递归）
configs.load(args.config, recursive=True)
configs.update(opts)  # 支持命令行覆盖

# 2. 解析变量引用
cfg = Config(recursive_eval(configs), filename=args.config)

# 3. 使用配置构建模型
model = build_model(cfg.model)
```

### 4.4 配置继承与覆盖

配置文件支持层级继承：

```
configs/
├── default.yaml                    # 全局默认配置
└── nuscenes/
    ├── default.yaml                # nuScenes 默认配置
    └── det/
        └── centerhead/
            └── lssfpn/
                └── camera/
                    ├── default.yaml
                    └── 256x704/
                        └── resnet/
                            ├── default.yaml
                            └── bevdepth.yaml  # 最终配置
```

**加载顺序：** 子配置覆盖父配置

---

## 5. 文件命名与组织规范

### 5.1 目录命名规范

| 目录类型 | 命名风格 | 示例 |
|---------|---------|------|
| 功能模块 | 复数名词 | `backbones/`, `heads/`, `ops/` |
| 数据集 | 数据集名称 | `nuscenes/`, `kitti/` |
| 任务类型 | 缩写 | `det/` (detection), `seg/` (segmentation) |
| 分辨率 | 格式 | `256x704/` |

### 5.2 文件命名规范

#### 类定义文件
```python
# 规则：小写 + 下划线，与类名对应
# 类名：大驼峰（PascalCase）

# ✓ 正确
sparse_encoder.py      # 包含 SparseEncoder 类
bevdepth.py            # 包含 BEVDepth 类
centerpoint.py         # 包含 CenterHead 类

# ✗ 错误
SparseEncoder.py
sparseEncoder.py
```

#### 工具函数文件
```python
# 规则：小写 + 下划线，描述性命名
builder.py             # 构建器函数
utils.py              # 通用工具
config.py             # 配置处理
```

#### CUDA/C++ 扩展
```python
# Python 接口文件
voxelize.py           # Voxelization 类的 Python 封装
scatter_points.py     # DynamicScatter 的 Python 封装

# C++/CUDA 源文件
voxelization.cpp      # C++ 实现
voxelization_cuda.cu  # CUDA 实现
voxelization_cpu.cpp  # CPU 实现
```

### 5.3 类命名规范

```python
# 骨干网络：描述性名称 + Backbone/Encoder
SparseEncoder         # 稀疏编码器
PillarEncoder         # Pillar 编码器
GeneralizedResNet     # 广义 ResNet

# 颈部网络：架构名称 + FPN/Neck
GeneralizedLSSFPN     # 广义 LSS FPN
DetectronFPN          # Detectron FPN

# 检测头：方法名称 + Head
CenterHead            # CenterPoint 检测头
TransFusionHead       # TransFusion 检测头

# 融合模块：方法名称 + Fuser
ConvFuser             # 卷积融合器
AddFuser              # 加法融合器

# 视角转换：方法名称
BEVDepth              # BEVDepth 方法
AwareBEVDepth         # 带注意力机制的 BEVDepth
LSS                   # Lift-Splat-Shoot
```

### 5.4 __all__ 定义规范

每个模块文件必须定义 `__all__`：

```python
# 规则 1: 放在文件顶部（在类定义之前）
__all__ = ["ClassName1", "ClassName2", "function_name"]

# 规则 2: 使用双引号
__all__ = ['ClassName']  # ✗ 错误
__all__ = ["ClassName"]  # ✓ 正确

# 规则 3: 按字母顺序排列（推荐）
__all__ = [
    "AwareBEVDepth",
    "BEVDepth",
    "LSS",
]

# 规则 4: 必须包含所有公开导出的内容
__all__ = ["MyClass"]  # 如果文件中有 MyClass 和 MyFunction
                       # 但只导出 MyClass，则 MyFunction 是私有的
```

### 5.5 注释与文档字符串

```python
# 类文档字符串
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module. Defaults to ('conv', 'norm', 'act').
    
    Returns:
        dict: Backbone features.
    """

# 函数文档字符串
def build_backbone(cfg):
    """Build backbone network from config.
    
    Args:
        cfg (dict): Configuration dictionary containing 'type' key.
    
    Returns:
        nn.Module: Constructed backbone network.
    """
    return BACKBONES.build(cfg)
```

---

## 6. 新建文件检查清单

### 6.1 创建新模块前的检查

**Step 1: 确定模块类型和位置**

| 模块类型 | 应放入目录 | 注册器 |
|---------|-----------|--------|
| 骨干网络 | `mmdet3d/models/backbones/` | `BACKBONES` |
| 颈部网络 | `mmdet3d/models/necks/` | `NECKS` |
| 检测头 | `mmdet3d/models/heads/bbox/` | `HEADS` |
| 分割头 | `mmdet3d/models/heads/segm/` | `HEADS` |
| 融合模型 | `mmdet3d/models/fusion_models/` | `FUSIONMODELS` |
| 融合模块 | `mmdet3d/models/fusers/` | `FUSERS` |
| 视角转换 | `mmdet3d/models/vtransforms/` | `VTRANSFORMS` |
| 自定义算子 | `mmdet3d/ops/` | 不需要注册 |

**Step 2: 检查命名规范**

- [ ] 文件名：小写 + 下划线（如 `my_backbone.py`）
- [ ] 类名：大驼峰（如 `MyBackbone`）
- [ ] 函数名：小写 + 下划线（如 `build_my_module`）

**Step 3: 创建文件模板**

```python
# mmdet3d/models/backbones/my_backbone.py
"""My custom backbone implementation."""

from typing import List, Optional

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16

from mmdet.models import BACKBONES

__all__ = ["MyBackbone"]


@BACKBONES.register_module()
class MyBackbone(nn.Module):
    """My custom backbone network.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config of normalization layer.
            Defaults to dict(type='BN2d', requires_grad=True).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg or dict(type='BN2d', requires_grad=True)
        
        # 定义网络层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        
        self.fp16_enabled = False
    
    def init_weights(self) -> None:
        """Initialize weights."""
        # 初始化逻辑
    
    @auto_fp16(apply_to=('x', ))
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            List[torch.Tensor]: List of feature maps.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return [x]
```

**Step 4: 更新 __init__.py**

```python
# mmdet3d/models/backbones/__init__.py
from .my_backbone import *  # 添加这一行
# 确保 my_backbone.py 中有 __all__ 定义
```

**Step 5: 验证导入**

```bash
# 测试导入是否成功
python -c "from mmdet3d.models.backbones import MyBackbone; print('Success!')"
python -c "from mmdet3d.models import build_backbone; print('Builder works!')"
```

### 6.2 创建新算子的检查

**Step 1: 创建目录结构**

```
mmdet3d/ops/my_op/
├── __init__.py
├── my_op.py              # Python 接口
├── src/
│   ├── my_op_cpu.cpp     # CPU 实现
│   └── my_op_cuda.cu     # CUDA 实现
└── include/              # (可选) 头文件
    └── my_op.h
```

**Step 2: 编写 Python 接口**

```python
# mmdet3d/ops/my_op/my_op.py
import torch
from torch import nn
from torch.autograd import Function

from .my_op_ext import my_op_cuda, my_op_cpu  # CUDA/C++ 扩展

__all__ = ["my_op", "MyOp"]


class _MyOp(Function):
    @staticmethod
    def forward(ctx, x):
        # 前向逻辑
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向逻辑
        return grad_input


my_op = _MyOp.apply


class MyOp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return my_op(x)
```

**Step 3: 更新 ops/__init__.py**

```python
# mmdet3d/ops/__init__.py
from .my_op import my_op, MyOp  # 添加导入

__all__ = [
    # ... 现有导出
    "my_op",
    "MyOp",
]
```

**Step 4: 更新 setup.py**

```python
# setup.py
ext_modules=[
    # ... 现有扩展
    make_cuda_ext(
        name="my_op_ext",
        module="mmdet3d.ops.my_op",
        sources=[
            "src/my_op_cpu.cpp",
            "src/my_op_cuda.cu",
        ],
    ),
]
```

**Step 5: 编译并测试**

```bash
python setup.py build_ext --inplace
python -c "from mmdet3d.ops import my_op; print('Success!')"
```

### 6.3 创建配置文件的检查

**Step 1: 确定配置文件位置**

```
configs/nuscenes/det/my_method/
├── default.yaml            # 默认配置
└── camera_lidar/
    └── resnet50/
        └── my_config.yaml  # 具体配置
```

**Step 2: 使用变量引用**

```yaml
# configs/nuscenes/det/my_method/camera_lidar/resnet50/my_config.yaml
model:
  encoders:
    camera:
      vtransform:
        type: MyMethod
        feature_size: ${[image_size[0] // 16, image_size[1] // 16]}
        xbound: ${xbound}
        ybound: ${ybound}
        zbound: ${zbound}

# 在父配置中定义变量
# configs/nuscenes/det/my_method/default.yaml
image_size: [256, 704]
xbound: [-51.2, 51.2, 0.8]
ybound: [-51.2, 51.2, 0.8]
zbound: [-10.0, 10.0, 20.0]
```

**Step 3: 验证配置加载**

```bash
python tools/train.py configs/nuscenes/det/my_method/camera_lidar/resnet50/my_config.yaml
```

---

## 7. 常见错误与解决方案

### 7.1 导入错误

**错误 1: ModuleNotFoundError**
```python
# 错误原因：循环导入
# A/__init__.py
from .b import BClass  # B 导入 C

# B/__init__.py
from ..a import AClass  # A 导入 B -> 循环！

# 解决方案：使用绝对导入，重构依赖关系
```

**错误 2: ImportError: cannot import name 'X'**
```python
# 错误原因：__all__ 未定义或遗漏
# 解决方案：在模块文件中添加 __all__

# my_module.py
__all__ = ["MyClass"]  # 添加此行

class MyClass:
    pass
```

### 7.2 注册失败

**错误：KeyError: 'MyModule' is not in the registry**

```python
# 错误原因 1：忘记添加装饰器
@BACKBONES.register_module()  # 确保添加此行
class MyModule(nn.Module):
    pass

# 错误原因 2：__init__.py 未导入
# backbones/__init__.py
from .my_module import *  # 确保添加此行

# 错误原因 3：导入顺序问题
# 确保在使用模块之前已经导入
import mmdet3d.models.backbones  # 先导入
model = build_backbone(cfg)     # 再使用
```

### 7.3 配置解析错误

**错误：NameError: name 'image_size' is not defined**

```yaml
# 错误原因：变量未定义
model:
  vtransform:
    feature_size: ${[image_size[0] // 16, image_size[1] // 16]}

# 解决方案：在配置文件中定义变量
image_size: [256, 704]

model:
  vtransform:
    feature_size: ${[image_size[0] // 16, image_size[1] // 16]}
```

---

## 8. 最佳实践总结

### 8.1 代码组织

1. **单一职责**：每个文件只定义一个主要类或相关的一组函数
2. **明确导出**：使用 `__all__` 明确控制导出内容
3. **避免循环**：使用绝对导入，避免循环依赖
4. **分层清晰**：保持 `backbones`、`necks`、`heads` 等层次分离

### 8.2 命名一致

1. **类名**：大驼峰，描述性（`SparseEncoder`）
2. **文件名**：小写 + 下划线，与类名对应（`sparse_encoder.py`）
3. **注册器**：使用预定义的 `BACKBONES`、`FUSIONMODELS` 等

### 8.3 配置管理

1. **变量复用**：使用 `${...}` 避免重复
2. **层级继承**：通过目录结构组织配置
3. **注释说明**：在配置中添加注释说明关键参数

### 8.4 测试验证

1. **导入测试**：新建文件后立即测试导入
2. **注册测试**：验证 `build_xxx` 能正确创建模块
3. **配置测试**：使用 `tools/train.py` 验证配置加载

---

## 附录 A：快速参考表

### 注册器速查

```python
from mmdet.models import BACKBONES, NECKS, HEADS, LOSSES
from mmdet3d.models.builder import FUSIONMODELS, VTRANSFORMS, FUSERS

@BACKBONES.register_module()      # 骨干网络
@NECKS.register_module()          # 颈部网络
@HEADS.register_module()          # 检测头/分割头
@FUSIONMODELS.register_module()   # 融合模型
@VTRANSFORMS.register_module()    # 视角转换
@FUSERS.register_module()         # 融合模块
```

### 导入路径速查

```python
from mmdet3d.models.builder import build_backbone, build_neck, build_head
from mmdet3d.ops import Voxelization, DynamicScatter, bev_pool
from mmdet3d.datasets import build_dataset
from mmdet3d.apis import train_model
```

### 配置文件变量速查

```yaml
# 常用变量定义
image_size: [256, 704]
xbound: [-51.2, 51.2, 0.8]    # X 轴范围 [min, max, step]
ybound: [-51.2, 51.2, 0.8]    # Y 轴范围
zbound: [-10.0, 10.0, 20.0]   # Z 轴范围
dbound: [1.0, 60.0, 1.0]      # 深度范围

# 变量引用示例
feature_size: ${[image_size[0] // 16, image_size[1] // 16]}
bev_size: ${[xbound[1] - xbound[0]] / xbound[2]}
```

---

**文档版本**: 1.0  
**最后更新**: 2026-04-01  
**维护者**: 架构管家
