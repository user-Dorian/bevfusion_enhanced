#!/bin/bash
# 强制修复脚本 - 解决 spconv 重复注册问题
# 使用方法：在服务器上执行 bash force_fix.sh

echo "=== 开始强制修复 ==="
echo ""

# 1. 进入项目目录
cd ~/workbench/test_bevfusion

echo "1. 强制拉取最新代码..."
git fetch origin
git reset --hard origin/main

echo ""
echo "2. 验证代码已更新..."
echo "   检查 mmdet3d/ops/__init__.py:"
grep -n "Lazy import sparse_block" mmdet3d/ops/__init__.py
if [ $? -eq 0 ]; then
    echo "   ✅ __init__.py 已更新"
else
    echo "   ❌ __init__.py 未更新"
    exit 1
fi

echo ""
echo "3. 检查 sparse_encoder.py:"
grep -n "from mmdet3d.ops.sparse_block import" mmdet3d/models/backbones/sparse_encoder.py
if [ $? -eq 0 ]; then
    echo "   ✅ sparse_encoder.py 已更新"
else
    echo "   ❌ sparse_encoder.py 未更新"
    exit 1
fi

echo ""
echo "4. 检查 multi_res_sparse_encoder.py:"
grep -n "from mmdet3d.ops.sparse_block import" mmdet3d/models/backbones/multi_res_sparse_encoder.py
if [ $? -eq 0 ]; then
    echo "   ✅ multi_res_sparse_encoder.py 已更新"
else
    echo "   ❌ multi_res_sparse_encoder.py 未更新"
    exit 1
fi

echo ""
echo "5. 清除 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "   ✅ 缓存已清除"

echo ""
echo "6. 验证导入..."
python -c "from mmdet3d.models.backbones import SparseEncoder; print('✅ 导入成功！')"
if [ $? -eq 0 ]; then
    echo "   ✅ 所有修复成功！"
else
    echo "   ❌ 导入失败"
    exit 1
fi

echo ""
echo "=== 修复完成！ ==="
echo ""
echo "现在可以运行训练命令了："
echo "torchpack dist-run -np 8 python tools/train.py \\"
echo "  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml \\"
echo "  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \\"
echo "  --run-dir train_scheme_a_6epochs"
echo ""
