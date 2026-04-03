#!/usr/bin/env python
# 终极修复脚本 - 解决 spconv 重复注册问题
# 使用方法：python fix_spconv_registration.py

import os
import sys

def modify_spconv_init():
    """修改 spconv/__init__.py，移除直接导入，改为延迟导入"""
    
    spconv_init_path = "mmdet3d/ops/spconv/__init__.py"
    
    if not os.path.exists(spconv_init_path):
        print(f"❌ 文件不存在：{spconv_init_path}")
        return False
    
    with open(spconv_init_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到第 15-24 行的导入并注释掉
    new_lines = []
    in_import_block = False
    
    for i, line in enumerate(lines, 1):
        # 检测是否是从 .conv 导入的块
        if i >= 15 and i <= 24:
            if not line.strip().startswith('#'):
                # 注释掉这一行
                new_lines.append('# ' + line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # 添加新的导入方式（延迟导入）
    # 在第 14 行后添加新的导入逻辑
    final_lines = []
    for i, line in enumerate(new_lines):
        final_lines.append(line)
        if i == 13:  # 在第 14 行后
            # 添加延迟导入的函数
            final_lines.append('\n')
            final_lines.append('# Lazy imports to avoid registration conflicts\n')
            final_lines.append('def __getattr__(name):\n')
            final_lines.append('    """Lazy loading to prevent spconv registration conflicts"""\n')
            final_lines.append('    if name in ["SparseConv2d", "SparseConv3d", "SubMConv2d", "SubMConv3d",\n')
            final_lines.append('                "SparseConvTranspose2d", "SparseConvTranspose3d",\n')
            final_lines.append('                "SparseInverseConv2d", "SparseInverseConv3d"]:\n')
            final_lines.append('        from .conv import (\n')
            final_lines.append('            SparseConv2d, SparseConv3d, SparseConvTranspose2d,\n')
            final_lines.append('            SparseConvTranspose3d, SparseInverseConv2d, SparseInverseConv3d,\n')
            final_lines.append('            SubMConv2d, SubMConv3d\n')
            final_lines.append('        )\n')
            final_lines.append('        return locals().get(name)\n')
            final_lines.append('    elif name in ["SparseModule", "SparseSequential"]:\n')
            final_lines.append('        from .modules import SparseModule, SparseSequential\n')
            final_lines.append('        return locals().get(name)\n')
            final_lines.append('    elif name in ["SparseMaxPool2d", "SparseMaxPool3d"]:\n')
            final_lines.append('        from .pool import SparseMaxPool2d, SparseMaxPool3d\n')
            final_lines.append('        return locals().get(name)\n')
            final_lines.append('    elif name == "SparseConvTensor":\n')
            final_lines.append('        from .structure import SparseConvTensor\n')
            final_lines.append('        return SparseConvTensor\n')
            final_lines.append('    elif name == "scatter_nd":\n')
            final_lines.append('        from .structure import scatter_nd\n')
            final_lines.append('        return scatter_nd\n')
            final_lines.append('    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n')
            final_lines.append('\n')
    
    with open(spconv_init_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    
    print(f"✅ 已修改 {spconv_init_path}")
    return True


def clear_cache():
    """清除 Python 缓存"""
    import subprocess
    
    print("清除 Python 缓存...")
    subprocess.run(['find', '.', '-type', 'd', '-name', '__pycache__', '-exec', 'rm', '-rf', '{}', '+'], 
                   stderr=subprocess.DEVNULL)
    subprocess.run(['find', '.', '-type', 'f', '-name', '*.pyc', '-delete'],
                   stderr=subprocess.DEVNULL)
    print("✅ 缓存已清除")


if __name__ == '__main__':
    print("=== 开始终极修复 ===\n")
    
    # 1. 修改 spconv/__init__.py
    print("1. 修改 spconv/__init__.py...")
    if not modify_spconv_init():
        print("❌ 修改失败")
        sys.exit(1)
    
    # 2. 清除缓存
    print("\n2. 清除 Python 缓存...")
    clear_cache()
    
    # 3. 验证
    print("\n3. 验证修复...")
    try:
        # 重新导入测试
        if 'mmdet3d.ops.spconv' in sys.modules:
            del sys.modules['mmdet3d.ops.spconv']
        if 'mmdet3d.ops' in sys.modules:
            del sys.modules['mmdet3d.ops']
        
        from mmdet3d.models.backbones import SparseEncoder
        print("✅ 导入成功！修复完成！")
    except Exception as e:
        print(f"❌ 导入失败：{e}")
        sys.exit(1)
    
    print("\n=== 修复完成！===\n")
    print("现在可以运行训练命令了：")
    print("torchpack dist-run -np 8 python tools/train.py \\")
    print("  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml \\")
    print("  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \\")
    print("  --run-dir train_scheme_a_6epochs")
