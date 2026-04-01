# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Utility functions for multi-resolution BEV feature processing."""

import torch
import torch.nn as nn
from typing import List, Tuple


class BEVFeatureAligner(nn.Module):
    """BEV feature alignment module for multi-resolution features.
    
    This module aligns BEV features from different distance zones to a unified
    resolution for fusion with camera features.
    
    Args:
        target_resolution (Tuple[int, int]): Target (H, W) resolution.
        align_method (str): Alignment method, options: ['bilinear', 'nearest'].
            Defaults to 'bilinear'.
    """
    
    def __init__(
        self,
        target_resolution: Tuple[int, int],
        align_method: str = "bilinear",
    ):
        super().__init__()
        self.target_resolution = target_resolution
        self.align_method = align_method
        
        assert align_method in ["bilinear", "nearest"], \
            f"Unsupported align_method: {align_method}"
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Align features to target resolution.
        
        Args:
            features: Input features of shape (B, C, H, W)
        
        Returns:
            Aligned features of shape (B, C, target_H, target_W)
        """
        if features.shape[-2:] == self.target_resolution:
            return features
        
        aligned = torch.nn.functional.interpolate(
            features,
            size=self.target_resolution,
            mode=self.align_method,
            align_corners=False if self.align_method == 'bilinear' else None,
        )
        
        return aligned


def align_multi_resolution_features(
    features_list: List[torch.Tensor],
    target_resolution: Tuple[int, int],
) -> torch.Tensor:
    """Align and fuse multi-resolution BEV features.
    
    Args:
        features_list: List of BEV features from different zones
        target_resolution: Target (H, W) resolution
    
    Returns:
        Fused features of shape (B, C, H, W)
    """
    if len(features_list) == 0:
        raise ValueError("features_list cannot be empty")
    
    aligner = BEVFeatureAligner(target_resolution)
    
    # Align all features to target resolution
    aligned_features = []
    for features in features_list:
        aligned = aligner(features)
        aligned_features.append(aligned)
    
    # Concatenate along channel dimension
    fused = torch.cat(aligned_features, dim=1)
    
    return fused
