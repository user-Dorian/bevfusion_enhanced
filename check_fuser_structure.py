import torch
from torch import nn

# 模拟 ConvFuser (nn.Sequential) 的权重命名
class ConvFuserSequential(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

# 模拟 MultiScaleConvFuser 的权重命名  
class MultiScaleConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Sequential(...)
        self.refine_conv = nn.Sequential(...)

# 打印两种结构的权重命名
print("ConvFuser (nn.Sequential) 权重命名:")
model1 = ConvFuserSequential([80, 256], 256)
for name, param in model1.named_parameters():
    print(f"  fuser.{name}")

print("\nMultiScaleConvFuser 权重命名:")
model2 = MultiScaleConvFuser([80, 256], 256)
for name, param in model2.named_parameters():
    print(f"  fuser.{name}")