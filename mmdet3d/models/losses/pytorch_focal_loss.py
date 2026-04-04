import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchFocalLoss(nn.Module):
    """
    Pure PyTorch implementation of Focal Loss to avoid mmcv CUDA extension issues.
    
    This is a CPU/GPU compatible focal loss that doesn't rely on mmcv's CUDA extensions.
    """
    
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        """
        Args:
            use_sigmoid (bool): Whether to use sigmoid activation. Default: True
            gamma (float): Focusing parameter. Default: 2.0
            alpha (float): Alpha balancing parameter. Default: 0.25
            reduction (str): Reduction method. Default: 'mean'
            loss_weight (float): Loss weight. Default: 1.0
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """
        Forward pass for focal loss.
        
        Args:
            pred (Tensor): Predicted logits, shape (N, num_classes, ...)
            target (Tensor): Target labels, shape (N, num_classes, ...) or (N, ...)
            weight (Tensor, optional): Sample weights. Default: None
            avg_factor (int, optional): Average factor. Default: None
            reduction_override (str, optional): Override reduction method. Default: None
        
        Returns:
            Tensor: Focal loss value
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        if self.use_sigmoid:
            # Sigmoid focal loss
            pred_sigmoid = pred.sigmoid()
            
            # Ensure target is float and same shape as pred
            if target.dim() != pred.dim():
                # Target is class indices, convert to one-hot
                target = F.one_hot(target, num_classes=pred.size(1)).float().transpose(1, -1)
            
            # Calculate focal weight
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
            
            # Binary cross entropy with logits
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            
            # Apply focal weight
            loss = focal_weight * bce
            
        else:
            # Softmax focal loss for multi-class
            if target.dim() != pred.dim():
                target = target.long()
            else:
                target = target.argmax(dim=1)
            
            logpt = F.log_softmax(pred, dim=1)
            logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = logpt.exp()
            
            focal_weight = (1 - pt).pow(self.gamma)
            loss = -focal_weight * logpt
        
        # Apply sample weights if provided
        if weight is not None:
            if weight.dim() != loss.dim():
                if weight.dim() == 1:
                    weight = weight.view(-1, 1)
                else:
                    weight = weight.unsqueeze(1)
            loss = loss * weight
        
        # Apply reduction
        if reduction == 'none':
            pass
        elif reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight


# Register the loss in mmdet
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class PyTorchFocalLossWrapper(PyTorchFocalLoss):
    """
    Wrapper for PyTorchFocalLoss to be compatible with mmdet's loss building system.
    """
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, **kwargs):
        super().__init__(
            use_sigmoid=use_sigmoid,
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            loss_weight=loss_weight
        )
