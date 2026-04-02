# BEVFusion Enhanced - Small Object Optimization (Scheme A)

## Overview

This project implements **Scheme A** optimizations for BEVFusion, focusing on improving small object detection performance in 3D LiDAR-camera fusion tasks.

## Key Optimizations

### 1. Distance-Adaptive Voxelization ✅
- **Purpose**: Handle non-uniform LiDAR point density (dense near, sparse far)
- **Implementation**: Three distance zones with different voxel sizes
  - Near (0-20m): 0.05m voxel size (high resolution)
  - Mid (20-40m): 0.075m voxel size (baseline)
  - Far (40-54m): 0.15m voxel size (low resolution)
- **Files**: `mmdet3d/ops/voxel/distance_adaptive_voxelize.py`

### 2. CBAM Attention Module ✅
- **Purpose**: Enhance feature representation for small objects
- **Implementation**: Channel + Spatial attention in SparseEncoder
- **Configuration**: `use_cbam: true`, `cbam_ratio: 16`
- **Files**: 
  - `mmdet3d/models/utils/cbam.py` (new)
  - `mmdet3d/models/backbones/sparse_encoder.py` (modified)

### 3. Small Object Anchor Optimization ✅
- **Purpose**: Better anchor matching for small objects
- **Implementation**: Class-specific anchor sizes
  - Pedestrian/Motorcycle/Bicycle: 0.6-0.8m
  - Traffic Cone/Barrier: 0.4-0.5m
- **Configuration**: `train_cfg.anchor_sizes`

### 4. Class-Specific Loss Weighting ✅
- **Purpose**: Focus training on small objects
- **Implementation**: Higher loss weights for small object classes
  - Pedestrian: 2.0x
  - Motorcycle/Bicycle: 2.5x
  - Traffic Cone/Barrier: 3.0x
- **Files**: `mmdet3d/models/heads/bbox/transfusion.py` (modified)

### 5. Copy-Paste Data Augmentation ✅
- **Purpose**: Increase small object training samples
- **Implementation**: Copy small objects and paste into scene
- **Configuration**: `data.train.copy_paste`
- **Files**: `mmdet3d/datasets/pipelines/copy_paste.py` (new)

## Expected Performance Improvements

| Metric | Baseline | Expected (Scheme A) | Improvement |
|--------|----------|---------------------|-------------|
| **mAP** | 35.05% | 45-50% | +10-15% |
| **NDS** | 42.05% | 50-55% | +8-13% |
| **Pedestrian AP** | ~60% | 70-75% | +10-15% |
| **Motorcycle AP** | ~23% | 40-50% | +17-27% |
| **Traffic Cone AP** | ~13% | 30-40% | +17-27% |

## Training Configuration

### Basic Training (6 epochs)
```bash
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --run-dir train_scheme_a_6epochs
```

### Large-Scale Training (20 epochs)
Edit `configs/default.yaml`:
```yaml
max_epochs: 20
```

Then run:
```bash
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --run-dir train_scheme_a_20epochs
```

## File Structure

```
bevfusion_enhanced/
├── configs/
│   └── nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/
│       └── distance_adaptive_voxel_scheme_a.yaml  # Optimized config
├── mmdet3d/
│   ├── datasets/pipelines/
│   │   ├── copy_paste.py  # NEW: Copy-Paste augmentation
│   │   └── __init__.py  # Updated
│   ├── models/
│   │   ├── backbones/
│   │   │   └── sparse_encoder.py  # Modified: CBAM integration
│   │   ├── heads/bbox/
│   │   │   └── transfusion.py  # Modified: Class weights
│   │   └── utils/
│   │       └── cbam.py  # NEW: CBAM attention
│   └── ops/voxel/
│       └── distance_adaptive_voxelize.py  # Existing: DAV module
└── README_SCHEME_A.md  # This file
```

## Requirements

- PyTorch >= 1.9
- MMCV == 1.4.0
- MMDetection3D == 0.11.0
- CUDA 11.0+
- GPU: RTX 3090 (recommended) or similar

## Compatibility Notes

⚠️ **Important**: This project is based on an early version of BEVFusion. The modifications are designed to be compatible with the existing codebase structure.

- All new modules use the existing registration system
- CBAM module uses standard PyTorch operations
- Copy-Paste augmentation follows MMDetection pipeline conventions
- No breaking changes to existing APIs

## Usage Guide

### Enable/Disable Specific Optimizations

All optimizations can be controlled via the configuration file:

```yaml
# 1. CBAM Attention
model:
  encoders:
    lidar:
      backbone:
        base_config:
          use_cbam: true  # Set to false to disable
          cbam_ratio: 16

# 2. Copy-Paste Augmentation
data:
  train:
    copy_paste:
      paste_probability: 0.5  # Set to 0.0 to disable

# 3. Class Weights
model:
  heads:
    object:
      train_cfg:
        class_weights: null  # Set to null to disable
```

### Customizing for Your Dataset

Adjust these parameters based on your specific needs:

1. **Voxel Sizes**: Modify `voxel_configs` for different distance ranges
2. **Anchor Sizes**: Adjust `anchor_sizes` for object dimensions in your dataset
3. **Class Weights**: Tune `class_weights` based on class imbalance
4. **Copy-Paste**: Modify `max_num_pasted` and `paste_probability`

## Testing

### Validation
```bash
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/distance_adaptive_voxel_scheme_a.yaml \
  work_dirs/train_scheme_a_6epochs/latest.pth \
  --eval bbox
```

### Visualization
Results will be saved in `work_dirs/train_scheme_a_*/` including:
- Training logs
- Validation results
- Checkpoints

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `samples_per_gpu` from 2 to 1
   - Reduce `max_num_pasted` in Copy-Paste config

2. **Import Errors**
   - Ensure all new files are in correct locations
   - Check `__init__.py` files include new modules

3. **Configuration Errors**
   - Verify YAML syntax
   - Check parameter names match code expectations

## Performance Benchmarks

Coming soon after large-scale training (20 epochs)...

## Citation

If you use this code, please cite:

```bibtex
@misc{bevfusion_enhanced,
  title={BEVFusion Enhanced: Small Object Optimization for Multi-Modal 3D Detection},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/user-Dorian/test_bevfusion}},
}
```

## License

This project follows the same license as the original BEVFusion.

## Contact

For questions or issues, please open an issue on GitHub.
