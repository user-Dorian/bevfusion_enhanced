# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair, _triple, _single
from mmcv.cnn import CONV_LAYERS
from .ops import *
from .core import SparseConvolution, SparseConvTensor


# 只在第一次导入时注册，避免重复注册
if not hasattr(CONV_LAYERS, '_spconv_registered'):
    @CONV_LAYERS.register_module()
    class SparseConv2d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SparseConv2d, self).__init__(
                2,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
            )


    @CONV_LAYERS.register_module()
    class SparseConv3d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SparseConv3d, self).__init__(
                3,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
            )


    @CONV_LAYERS.register_module()
    class SparseConv4d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SparseConv4d, self).__init__(
                4,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
            )


    @CONV_LAYERS.register_module()
    class SparseConvTranspose2d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SparseConvTranspose2d, self).__init__(
                2,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
                transposed=True,
            )


    @CONV_LAYERS.register_module()
    class SparseConvTranspose3d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SparseConvTranspose3d, self).__init__(
                3,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
                transposed=True,
            )


    @CONV_LAYERS.register_module()
    class SparseInverseConv2d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            indice_key=None,
        ):
            super(SparseInverseConv2d, self).__init__(
                2,
                in_channels,
                out_channels,
                kernel_size,
                1,
                0,
                1,
                1,
                True,
                indice_key=indice_key,
                inverse=True,
            )


    @CONV_LAYERS.register_module()
    class SparseInverseConv3d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            indice_key=None,
        ):
            super(SparseInverseConv3d, self).__init__(
                3,
                in_channels,
                out_channels,
                kernel_size,
                1,
                0,
                1,
                1,
                True,
                indice_key=indice_key,
                inverse=True,
            )


    @CONV_LAYERS.register_module()
    class SubMConv2d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SubMConv2d, self).__init__(
                2,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
                subm=True,
            )


    @CONV_LAYERS.register_module()
    class SubMConv3d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SubMConv3d, self).__init__(
                3,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
                subm=True,
            )


    @CONV_LAYERS.register_module()
    class SubMConv4d(SparseConvolution):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            indice_key=None,
        ):
            super(SubMConv4d, self).__init__(
                4,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                indice_key=indice_key,
                subm=True,
            )

    # 标记已注册
    CONV_LAYERS._spconv_registered = True
