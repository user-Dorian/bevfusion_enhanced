# 🚨 紧急修复：重新编译和部署指南

## 问题诊断

**错误信息**：
```
ImportError: cannot import name 'DistanceAdaptiveVoxelization' from 'mmdet3d.ops'
```

**根本原因**：
1. ❌ 服务器上的代码**未更新**到最新版本
2. ❌ 新建的 Python 模块**未编译安装**
3. ❌ `mmdet3d/ops` 下的新模块需要**重新编译 CUDA 扩展**

---

## 🔧 解决方案（按顺序执行）

### 步骤 1：拉取最新代码

```bash
# SSH 登录服务器
ssh root@6mjuuswg

# 进入项目目录
cd ~/workbench/bevfusion_enhanced

# 拉取最新代码（包含所有新建文件）
git pull origin main

# 验证文件是否存在
ls -la mmdet3d/ops/voxel/distance_adaptive_voxelize.py
ls -la mmdet3d/models/backbones/multi_res_sparse_encoder.py
ls -la mmdet3d/ops/bev_pool/bev_align.py
```

**预期输出**：应该能看到这 3 个文件

---

### 步骤 2：清理旧的编译产物（重要！）

```bash
# 清理旧的构建文件
rm -rf build/
rm -rf dist/
rm -rf *.egg-info

# 清理 Python 缓存
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 清理 CUDA 编译产物（如果有）
rm -rf mmdet3d/ops/.mim/
rm -rf mmdet3d/ops/build/
```

---

### 步骤 3：重新编译安装（关键步骤）

```bash
# 确保在正确的环境中
which python
# 应该输出：/root/mina/envs/bevfusion/bin/python 或类似路径

# 重新编译安装（重要！）
python setup.py develop

# 或者使用 pip（推荐）
pip install -e . -v
```

**预期输出**：
```
running develop
running build_ext
Installing /root/mina/envs/bevfusion/lib/python3.8/site-packages/mmdet3d.egg-link
...
Successfully installed bevfusion-0.1.0
```

---

### 步骤 4：验证模块导入

```bash
# 测试导入（不运行训练，只测试导入）
python -c "
from mmdet3d.ops import DistanceAdaptiveVoxelization
from mmdet3d.models.backbones import MultiResSparseEncoder
from mmdet3d.ops.bev_pool import BEVFeatureAligner
print('✅ 所有模块导入成功！')
"
```

**如果仍然失败**，检查：
```bash
# 检查 mmdet3d/ops/__init__.py 是否包含导出
cat mmdet3d/ops/__init__.py | grep DistanceAdaptiveVoxelization

# 应该输出：
# from .voxel import DistanceAdaptiveVoxelization
# 或类似内容
```

---

### 步骤 5：验证配置文件

```bash
# 检查配置文件是否存在
ls -la configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml

# 验证 YAML 语法
python -c "
import yaml
with open('configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ 配置文件语法正确')
print(f'Voxelization 类型：{config[\"model\"][\"encoders\"][\"lidar\"][\"voxelize\"][\"type\"]}')
"
```

---

### 步骤 6：运行验证脚本

```bash
# 运行快速验证脚本
python scripts/validate_distance_adaptive.py
```

**预期输出**：
```
============================================================
距离自适应体素化方案 - 快速验证
============================================================

测试 1: 模块导入
============================================================
✅ DistanceAdaptiveVoxelization 导入成功
✅ MultiResSparseEncoder 导入成功
✅ BEVFeatureAligner 导入成功
✅ 所有模块导入测试通过

测试 2: 距离自适应体素化
============================================================
✅ 体素化测试通过
...

总计：4/4 测试通过
🎉 所有测试通过！可以开始部署到服务器。
```

---

### 步骤 7：测试训练（小规模）

```bash
# 单 GPU 测试 1 个 epoch
torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --run-dir test_dav \
  --max-epochs 1
```

---

## ⚠️ 常见问题排查

### 问题 1：`ImportError` 仍然存在

**解决方案 A**：检查 `__init__.py` 文件

```bash
# 检查 mmdet3d/ops/voxel/__init__.py
cat mmdet3d/ops/voxel/__init__.py

# 应该包含：
# from .distance_adaptive_voxelize import DistanceAdaptiveVoxelization, distance_adaptive_voxelize
# __all__ = [..., "DistanceAdaptiveVoxelization", ...]
```

如果缺失，手动添加：
```bash
# 编辑文件
nano mmdet3d/ops/voxel/__init__.py

# 添加内容：
from .distance_adaptive_voxelize import DistanceAdaptiveVoxelization, distance_adaptive_voxelize

__all__ = [
    "Voxelization", 
    "voxelization", 
    "dynamic_scatter", 
    "DynamicScatter",
    "DistanceAdaptiveVoxelization",
    "distance_adaptive_voxelize",
]
```

**解决方案 B**：强制重新安装

```bash
# 卸载
pip uninstall bevfusion -y

# 清理
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete

# 重新安装
python setup.py develop
```

---

### 问题 2：CUDA 扩展编译失败

**症状**：
```
error: cuda.h: No such file or directory
```

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version

# 应该输出 CUDA 版本信息
# 如果没有输出，需要安装 CUDA Toolkit
```

**如果 CUDA 未安装**：
```bash
# 安装 CUDA 11.1+（根据服务器环境）
# Ubuntu/Debian:
apt-get update
apt-get install cuda-toolkit-11-1

# 或者使用 conda
conda install cudatoolkit=11.1
```

---

### 问题 3：`mmcv` 版本不兼容

**症状**：
```
ImportError: cannot import name 'xxx' from 'mmcv'
```

**解决方案**：
```bash
# 检查 mmcv 版本
python -c "import mmcv; print(mmcv.__version__)"

# 应该是 1.4.0
# 如果不是，重新安装
pip install mmcv==1.4.0
```

---

### 问题 4：Git 拉取失败

**症状**：
```
error: The following untracked working tree files would be overwritten by merge
```

**解决方案**：
```bash
# 强制拉取（会覆盖本地修改）
git fetch --all
git reset --hard origin/main

# 或者备份后拉取
git stash
git pull origin main
git stash pop
```

---

## 📋 完整命令清单（一键复制）

```bash
#!/bin/bash
# 一键部署脚本 - 复制粘贴执行

set -e  # 遇到错误立即退出

echo "=== 步骤 1: 拉取最新代码 ==="
cd ~/workbench/bevfusion_enhanced
git pull origin main

echo "=== 步骤 2: 清理旧文件 ==="
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

echo "=== 步骤 3: 重新编译安装 ==="
python setup.py develop

echo "=== 步骤 4: 验证模块导入 ==="
python -c "
from mmdet3d.ops import DistanceAdaptiveVoxelization
from mmdet3d.models.backbones import MultiResSparseEncoder
print('✅ 所有模块导入成功！')
"

echo "=== 步骤 5: 运行验证脚本 ==="
python scripts/validate_distance_adaptive.py

echo "=== 部署完成！==="
echo "现在可以运行训练："
echo "torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --run-dir train_dav"
```

---

## ✅ 成功标志

执行完所有步骤后，应该看到：

1. ✅ `git pull` 成功，显示 "Already up to date" 或拉取新提交
2. ✅ `python setup.py develop` 成功，无报错
3. ✅ 模块导入测试输出 "✅ 所有模块导入成功！"
4. ✅ 验证脚本输出 "🎉 所有测试通过！"

---

## 🆘 如果仍然失败

请提供以下信息：

1. **完整的错误日志**（复制粘贴）
2. **执行 `git log -1` 的输出**（确认版本）
3. **执行 `python -c "import mmdet3d; print(mmdet3d.__file__)"` 的输出**（确认安装路径）
4. **执行 `ls -la mmdet3d/ops/voxel/` 的输出**（确认文件存在）

---

**生成时间**: 2026-04-01  
**紧急程度**: 🔴 高  
**预计解决时间**: 10-15 分钟
