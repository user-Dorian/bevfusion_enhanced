"""验证配置文件是否正确加载"""
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval

config_path = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"

print("=" * 80)
print("测试配置文件:", config_path)
print("=" * 80)

try:
    configs.load(config_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_path)
    
    print("\n✅ 配置加载成功！")
    
    # 检查关键配置
    if hasattr(cfg.model, 'fuser'):
        print("\n✅ fuser 配置存在:")
        print(cfg.model.fuser)
    else:
        print("\n❌ fuser 配置不存在!")
    
    if hasattr(cfg.model, 'decoder'):
        print("\n✅ decoder 配置存在:")
        print(cfg.model.decoder)
    else:
        print("\n❌ decoder 配置不存在!")
    
    if hasattr(cfg.model, 'heads'):
        print("\n✅ heads 配置存在:")
        print(cfg.model.heads)
    else:
        print("\n❌ heads 配置不存在!")
    
    print("\n" + "=" * 80)
    print("完整配置摘要:")
    print("=" * 80)
    print(f"voxel_size: {cfg.voxel_size}")
    print(f"point_cloud_range: {cfg.point_cloud_range}")
    print(f"model.encoders.camera.backbone.type: {cfg.model.encoders.camera.backbone.type}")
    print(f"model.decoder.backbone.type: {cfg.model.decoder.backbone.type if hasattr(cfg.model, 'decoder') else 'MISSING'}")
    print(f"model.fuser.type: {cfg.model.fuser.type if hasattr(cfg.model, 'fuser') else 'MISSING'}")
    
except Exception as e:
    print(f"\n❌ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
