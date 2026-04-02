# Copyright (c) OpenMMLab. All rights reserved.
"""Copy-Paste data augmentation for 3D object detection.

This module implements Copy-Paste augmentation specifically designed for
small objects in LiDAR point cloud data.
"""

import numpy as np
import torch
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CopyPaste3D:
    """Copy-Paste augmentation for 3D point cloud.
    
    This augmentation copies small objects from other samples and pastes
    them into the current sample to increase the number of small objects.
    
    Args:
        classes (list[str]): List of object classes to apply Copy-Paste.
        max_num_pasted (int): Maximum number of objects to paste per sample.
            Default: 10.
        paste_probability (float): Probability of applying Copy-Paste.
            Default: 0.5.
        scale_range (tuple[float]): Range of random scaling for pasted objects.
            Default: (0.9, 1.1).
        rotate_range (tuple[float]): Range of random rotation (in radians) 
            for pasted objects. Default: (-0.157, 0.157).
    """
    
    def __init__(
        self,
        classes,
        max_num_pasted=10,
        paste_probability=0.5,
        scale_range=(0.9, 1.1),
        rotate_range=(-0.157, 0.157),
    ):
        self.classes = classes
        self.max_num_pasted = max_num_pasted
        self.paste_probability = paste_probability
        self.scale_range = scale_range
        self.rotate_range = rotate_range
        
        # Cache for storing objects from other samples
        self.object_cache = []
        self.cache_size = 100  # Maximum cache size
        
    def __call__(self, input_dict):
        """Call function to copy-paste objects.
        
        Args:
            input_dict (dict): Input dict containing points, gt_bboxes_3d, 
                gt_labels_3d, etc.
        
        Returns:
            dict: Results after copy-paste augmentation.
        """
        import random
        
        # Randomly decide whether to apply Copy-Paste
        if random.random() > self.paste_probability:
            return input_dict
        
        points = input_dict['points']
        gt_bboxes_3d = input_dict.get('gt_bboxes_3d', None)
        gt_labels_3d = input_dict.get('gt_labels_3d', None)
        
        # Skip if no ground truth boxes
        if gt_bboxes_3d is None or len(gt_bboxes_3d) == 0:
            return input_dict
        
        # Convert to numpy if tensor
        if isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.cpu().numpy()
        if isinstance(gt_labels_3d, torch.Tensor):
            gt_labels_3d = gt_labels_3d.cpu().numpy()
        
        # Get indices of small objects to potentially copy
        small_object_indices = self._get_small_object_indices(
            gt_bboxes_3d, gt_labels_3d
        )
        
        if len(small_object_indices) == 0:
            return input_dict
        
        # Randomly select objects to paste
        num_to_paste = min(
            random.randint(1, self.max_num_pasted),
            len(small_object_indices)
        )
        selected_indices = random.sample(
            small_object_indices, num_to_paste
        )
        
        # Extract and transform objects
        pasted_points = []
        pasted_bboxes = []
        pasted_labels = []
        
        for idx in selected_indices:
            # Extract object points (simplified - in practice, need segmentation)
            # Here we use a simple approach based on bbox
            obj_bbox = gt_bboxes_3d[idx]
            obj_label = gt_labels_3d[idx]
            
            # Create object points from bbox (simplified)
            obj_points = self._create_object_points(obj_bbox, points)
            
            if obj_points is not None and len(obj_points) > 0:
                # Apply random transformations
                obj_points, obj_bbox = self._transform_object(
                    obj_points, obj_bbox
                )
                
                # Check if the pasted location is valid (not colliding)
                if self._is_valid_location(obj_bbox, gt_bboxes_3d):
                    pasted_points.append(obj_points)
                    pasted_bboxes.append(obj_bbox)
                    pasted_labels.append(obj_label)
        
        # Add pasted objects to the scene
        if len(pasted_points) > 0:
            all_points = [points] + pasted_points
            input_dict['points'] = np.vstack(all_points)
            
            # Update ground truth
            if len(pasted_bboxes) > 0:
                all_bboxes = np.vstack([gt_bboxes_3d, np.stack(pasted_bboxes)])
                all_labels = np.concatenate([gt_labels_3d, pasted_labels])
                
                input_dict['gt_bboxes_3d'] = all_bboxes
                input_dict['gt_labels_3d'] = all_labels
        
        return input_dict
    
    def _get_small_object_indices(self, bboxes, labels):
        """Get indices of small objects suitable for Copy-Paste.
        
        Args:
            bboxes (np.ndarray): Ground truth bounding boxes.
            labels (np.ndarray): Ground truth labels.
        
        Returns:
            list[int]: Indices of small objects.
        """
        small_indices = []
        
        # Define small object classes (pedestrian, motorcycle, traffic cone, etc.)
        small_class_names = [
            'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone',
            'barrier'
        ]
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Check if object is small based on dimensions
            # bbox format: [x, y, z, x_size, y_size, z_size, yaw]
            if len(bbox) >= 6:
                x_size, y_size, z_size = bbox[3:6]
                
                # Heuristic: small if volume < 2 m³
                volume = x_size * y_size * z_size
                if volume < 2.0:
                    small_indices.append(i)
        
        return small_indices
    
    def _create_object_points(self, bbox, all_points):
        """Create object points from bounding box.
        
        This is a simplified implementation. In practice, you would
        extract actual object points from segmented point clouds.
        
        Args:
            bbox (np.ndarray): Object bounding box.
            all_points (np.ndarray): All points in the scene.
        
        Returns:
            np.ndarray: Object points.
        """
        # Extract points within the bbox
        x, y, z = bbox[:3]
        x_size, y_size, z_size = bbox[3:6]
        yaw = bbox[6] if len(bbox) > 6 else 0
        
        # Create a simple mask for points within bbox
        dx = np.abs(all_points[:, 0] - x)
        dy = np.abs(all_points[:, 1] - y)
        dz = np.abs(all_points[:, 2] - z)
        
        mask = (
            (dx < x_size / 2) & 
            (dy < y_size / 2) & 
            (dz < z_size / 2)
        )
        
        obj_points = all_points[mask]
        
        # Return subset if too many points
        if len(obj_points) > 100:
            indices = np.random.choice(len(obj_points), 100, replace=False)
            obj_points = obj_points[indices]
        
        return obj_points if len(obj_points) > 0 else None
    
    def _transform_object(self, points, bbox):
        """Apply random transformations to object.
        
        Args:
            points (np.ndarray): Object points.
            bbox (np.ndarray): Object bounding box.
        
        Returns:
            tuple: (transformed_points, transformed_bbox)
        """
        import random
        
        # Random scaling
        scale = random.uniform(*self.scale_range)
        points[:, :3] *= scale
        bbox[3:6] *= scale
        
        # Random rotation around z-axis
        rotation = random.uniform(*self.rotate_range)
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        
        # Rotate points
        rot_matrix = np.array([
            [cos_rot, -sin_rot, 0],
            [sin_rot, cos_rot, 0],
            [0, 0, 1]
        ])
        points[:, :3] = points[:, :3] @ rot_matrix.T
        
        # Update bbox yaw
        if len(bbox) > 6:
            bbox[6] += rotation
        
        return points, bbox
    
    def _is_valid_location(self, bbox, existing_bboxes):
        """Check if the pasted location is valid (no collision).
        
        Args:
            bbox (np.ndarray): Bounding box to check.
            existing_bboxes (np.ndarray): Existing bounding boxes.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        # Simple distance-based collision check
        x, y, z = bbox[:3]
        
        for existing_bbox in existing_bboxes:
            ex, ey, ez = existing_bbox[:3]
            distance = np.sqrt((x - ex)**2 + (y - ey)**2)
            
            # Minimum distance threshold (2 meters)
            if distance < 2.0:
                return False
        
        return True
    
    def __repr__(self):
        """Print the representation of the transform."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes}, '
        repr_str += f'max_num_pasted={self.max_num_pasted}, '
        repr_str += f'paste_probability={self.paste_probability})'
        return repr_str
