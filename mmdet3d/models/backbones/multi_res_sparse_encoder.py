# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Dict
import torch
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops.sparse_block import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet.models import BACKBONES
from .sparse_encoder import SparseEncoder


@BACKBONES.register_module()
class MultiResSparseEncoder(nn.Module):
    """Multi-resolution sparse encoder for distance-adaptive voxelization.
    
    This encoder processes voxel features from multiple distance zones with
    different resolutions and fuses them into a unified BEV representation.
    
    Args:
        in_channels (int): Input channels (point features, typically 4-5).
        base_config (dict): Base configuration for SparseEncoder.
        zone_configs (list[dict]): Configuration for each distance zone:
            - sparse_shape: Sparse shape for this zone
            - out_channels: Output channels for this zone
        fusion_method (str): Feature fusion method, options: ['concat', 'add'].
            Defaults to 'concat'.
    """
    
    def __init__(
        self,
        in_channels: int,
        base_config: Dict,
        zone_configs: List[Dict],
        fusion_method: str = "concat",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_config = base_config
        self.zone_configs = zone_configs
        self.fusion_method = fusion_method
        
        # Create independent SparseEncoder for each distance zone
        self.zone_encoders = nn.ModuleList()
        for zone_idx, zone_config in enumerate(zone_configs):
            encoder = SparseEncoder(
                in_channels=in_channels,
                sparse_shape=zone_config['sparse_shape'],
                **base_config.get('kwargs', {}),
            )
            self.zone_encoders.append(encoder)
        
        # Calculate output channels after fusion
        if fusion_method == 'concat':
            total_out_channels = sum([cfg.get('out_channels', 128) for cfg in zone_configs])
        else:  # 'add'
            total_out_channels = zone_configs[0].get('out_channels', 128)
        
        # Fusion layer to combine multi-resolution features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_out_channels, base_config.get('out_channels', 128), 3, padding=1),
            nn.BatchNorm2d(base_config.get('out_channels', 128)),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = base_config.get('out_channels', 128)
    
    @auto_fp16(apply_to=("voxel_features",))
    def forward(self, zone_results: List[Dict], batch_size: int, **kwargs):
        """Forward pass for multi-resolution sparse encoding.
        
        Args:
            zone_results: List of zone outputs from DistanceAdaptiveVoxelization,
                each dict contains:
                - voxels: Voxel features
                - coords: Voxel coordinates (with zone_idx)
                - num_points: Points per voxel
                - zone_idx: Distance zone index
            batch_size: Batch size
            **kwargs: Additional arguments
        
        Returns:
            BEV features tensor of shape (B, C, H, W)
        """
        zone_bev_features = []
        
        for zone_idx, encoder in enumerate(self.zone_encoders):
            # Find results for this zone
            zone_data = None
            for result in zone_results:
                if result['zone_idx'] == zone_idx:
                    zone_data = result
                    break
            
            # Skip if no data for this zone
            if zone_data is None:
                continue
            
            # Prepare sparse tensor for this zone
            voxel_features = zone_data['voxels']
            coords = zone_data['coords']
            
            # Create sparse tensor
            # Note: coords format is [batch_idx, z, y, x]
            sparse_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=coords,
                spatial_shape=self.zone_configs[zone_idx]['sparse_shape'],
                batch_size=batch_size,
            )
            
            # Encode this zone
            zone_features = encoder(sparse_tensor)
            
            zone_bev_features.append(zone_features)
        
        # Fuse multi-resolution features
        if len(zone_bev_features) == 0:
            # Return zero tensor if no valid features
            return torch.zeros(
                batch_size, 
                self.out_channels, 
                self.zone_configs[1]['sparse_shape'][1],  # Use mid-range H
                self.zone_configs[1]['sparse_shape'][0],  # Use mid-range W
                device=voxel_features.device if 'voxel_features' in locals() else 'cpu',
                dtype=torch.float32
            )
        
        # Align all features to the same resolution (mid-range zone)
        aligned_features = []
        target_h = self.zone_configs[1]['sparse_shape'][1]
        target_w = self.zone_configs[1]['sparse_shape'][0]
        
        for zone_feat in zone_bev_features:
            if zone_feat.shape[-2:] != (target_h, target_w):
                # Resize to target resolution
                zone_feat = torch.nn.functional.interpolate(
                    zone_feat,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )
            aligned_features.append(zone_feat)
        
        # Concatenate and fuse
        if self.fusion_method == 'concat':
            fused = torch.cat(aligned_features, dim=1)
        else:  # 'add'
            # Sum features (requires same channels)
            fused = sum(aligned_features)
        
        # Final fusion convolution
        fused = self.fusion_conv(fused)
        
        return fused
