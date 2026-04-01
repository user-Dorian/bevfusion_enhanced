#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 用于在服务器上快速测试距离自适应体素化方案

使用方法:
    python scripts/validate_distance_adaptive.py

验证内容:
    1. 模块导入测试
    2. 配置文件语法测试
    3. 简单前向传播测试
"""

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        from mmdet3d.ops.voxel.distance_adaptive_voxelize import DistanceAdaptiveVoxelization
        print("✅ DistanceAdaptiveVoxelization 导入成功")
    except Exception as e:
        print(f"❌ DistanceAdaptiveVoxelization 导入失败：{e}")
        return False
    
    try:
        from mmdet3d.models.backbones.multi_res_sparse_encoder import MultiResSparseEncoder
        print("✅ MultiResSparseEncoder 导入成功")
    except Exception as e:
        print(f"❌ MultiResSparseEncoder 导入失败：{e}")
        return False
    
    try:
        from mmdet3d.ops.bev_pool.bev_align import BEVFeatureAligner
        print("✅ BEVFeatureAligner 导入成功")
    except Exception as e:
        print(f"❌ BEVFeatureAligner 导入失败：{e}")
        return False
    
    print("✅ 所有模块导入测试通过\n")
    return True


def test_voxelization():
    """测试体素化功能"""
    print("=" * 60)
    print("测试 2: 距离自适应体素化")
    print("=" * 60)
    
    try:
        from mmdet3d.ops.voxel.distance_adaptive_voxelize import DistanceAdaptiveVoxelization
        
        # 配置
        voxel_configs = [
            {'range': (0, 20), 'voxel_size': [0.05, 0.05, 0.1], 'max_voxels': 50000},
            {'range': (20, 40), 'voxel_size': [0.075, 0.075, 0.15], 'max_voxels': 80000},
            {'range': (40, 54), 'voxel_size': [0.15, 0.15, 0.3], 'max_voxels': 40000},
        ]
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        
        # 创建体素化器
        voxelizer = DistanceAdaptiveVoxelization(
            voxel_configs=voxel_configs,
            point_cloud_range=point_cloud_range,
            max_num_points=10,
        )
        
        # 创建测试点云
        points = torch.randn(10000, 4)
        points[:, :2] *= 50  # 缩放到 50m 范围
        points[:, 3] = 100  # 模拟强度
        
        # 测试
        results = voxelizer(points)
        
        print(f"✅ 体素化测试通过")
        print(f"   - 输入点数：{len(points)}")
        print(f"   - 输出区域数：{len(results)}")
        for i, result in enumerate(results):
            print(f"   - 区域 {i}: {len(result['voxels'])} voxels")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 体素化测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder():
    """测试编码器功能"""
    print("=" * 60)
    print("测试 3: 多分辨率稀疏编码")
    print("=" * 60)
    
    try:
        from mmdet3d.models.backbones.multi_res_sparse_encoder import MultiResSparseEncoder
        
        # 配置
        voxel_configs = [
            {'range': (0, 20), 'voxel_size': [0.05, 0.05, 0.1], 'max_voxels': 50000, 'sparse_shape': [800, 800, 81]},
            {'range': (20, 40), 'voxel_size': [0.075, 0.075, 0.15], 'max_voxels': 80000, 'sparse_shape': [533, 533, 54]},
            {'range': (40, 54), 'voxel_size': [0.15, 0.15, 0.3], 'max_voxels': 40000, 'sparse_shape': [187, 187, 27]},
        ]
        
        base_config = {
            'out_channels': 128,
            'kwargs': {
                'order': ['conv', 'norm', 'act'],
                'encoder_channels': [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
                'block_type': 'basicblock',
            }
        }
        
        # 创建编码器
        encoder = MultiResSparseEncoder(
            in_channels=5,
            base_config=base_config,
            zone_configs=voxel_configs,
        )
        
        # 创建模拟数据
        zone_results = [
            {
                'voxels': torch.randn(1000, 10, 5),
                'coords': torch.randint(0, 100, (1000, 4)),
                'num_points': torch.randint(1, 10, (1000,)),
                'zone_idx': 0,
            },
            {
                'voxels': torch.randn(800, 10, 5),
                'coords': torch.randint(0, 100, (800, 4)),
                'num_points': torch.randint(1, 10, (800,)),
                'zone_idx': 1,
            },
        ]
        
        # 测试
        output = encoder(zone_results, batch_size=1)
        
        print(f"✅ 编码器测试通过")
        print(f"   - 输出形状：{output.shape}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 编码器测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置文件"""
    print("=" * 60)
    print("测试 4: 配置文件语法")
    print("=" * 60)
    
    try:
        import yaml
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel.yaml'
        )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件语法测试通过")
        print(f"   - 配置文件：{config_path}")
        print(f"   - voxel_configs 数量：{len(config['voxel_configs'])}")
        print(f"   - 模型类型：{config['model']['encoders']['lidar']['voxelize']['type']}")
        print(f"   - Backbone 类型：{config['model']['encoders']['lidar']['backbone']['type']}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败：{e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("距离自适应体素化方案 - 快速验证")
    print("=" * 60 + "\n")
    
    tests = [
        ("模块导入", test_imports),
        ("体素化", test_voxelization),
        ("编码器", test_encoder),
        ("配置文件", test_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 测试异常：{e}")
            results.append((name, False))
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    total_passed = sum([1 for _, r in results if r])
    total_tests = len(results)
    
    print(f"\n总计：{total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！可以开始部署到服务器。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
