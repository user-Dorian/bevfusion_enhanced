# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from typing import List, Dict

from .voxelize import Voxelization


class DistanceAdaptiveVoxelization(nn.Module):
    """Distance-adaptive voxelization for multi-range LiDAR processing.
    
    This module divides point cloud into multiple distance zones and applies
    different voxel sizes for each zone to handle the non-uniform point density.
    
    Args:
        voxel_configs: List of dicts, each containing:
            - range: Tuple (min, max) distance range in meters
            - voxel_size: List [vx, vy, vz] voxel size in meters
            - max_voxels: int maximum number of voxels for this zone
        point_cloud_range: List [x_min, y_min, z_min, x_max, y_max, z_max]
        max_num_points: int, maximum points per voxel (default: 10)
    """
    
    def __init__(
        self,
        voxel_configs: List[Dict],
        point_cloud_range: List[float],
        max_num_points: int = 10,
    ):
        super().__init__()
        self.voxel_configs = voxel_configs
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        
        # Create independent voxelizer for each distance zone
        self.voxelizers = nn.ModuleList()
        for config in voxel_configs:
            voxelizer = Voxelization(
                voxel_size=config['voxel_size'],
                point_cloud_range=point_cloud_range,
                max_num_points=max_num_points,
                max_voxels=config['max_voxels'],
            )
            self.voxelizers.append(voxelizer)
    
    def forward(self, points: torch.Tensor) -> List[Dict]:
        """Forward pass for distance-adaptive voxelization.
        
        Args:
            points: Input points, shape (N, 4) where N is number of points,
                   columns are [x, y, z, intensity]
        
        Returns:
            List of dicts, each containing:
                - voxels: Voxel features
                - coords: Voxel coordinates
                - num_points: Number of points per voxel
                - zone_idx: Distance zone index
        """
        # Calculate distance from LiDAR origin (0, 0)
        distances = torch.norm(points[:, :2], dim=1)
        
        zone_results = []
        
        for zone_idx, (config, voxelizer) in enumerate(
            zip(self.voxel_configs, self.voxelizers)
        ):
            # Filter points in this distance range
            range_min, range_max = config['range']
            mask = (distances >= range_min) & (distances < range_max)
            zone_points = points[mask]
            
            # Skip if no points in this zone
            if len(zone_points) == 0:
                continue
            
            # Apply voxelization
            voxels, coords, num_points = voxelizer(zone_points)
            
            # Store results with zone index
            zone_results.append({
                'voxels': voxels,
                'coords': coords,
                'num_points': num_points,
                'zone_idx': zone_idx,
                'voxel_size': config['voxel_size'],
            })
        
        return zone_results


def distance_adaptive_voxelize(
    points: torch.Tensor,
    voxel_configs: List[Dict],
    point_cloud_range: List[float],
    max_num_points: int = 10,
) -> List[Dict]:
    """Functional interface for distance-adaptive voxelization.
    
    Args:
        points: Input points (N, 4)
        voxel_configs: List of zone configurations
        point_cloud_range: Point cloud range
        max_num_points: Max points per voxel
    
    Returns:
        List of voxelization results for each zone
    """
    voxelizer = DistanceAdaptiveVoxelization(
        voxel_configs=voxel_configs,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
    )
    return voxelizer(points)
