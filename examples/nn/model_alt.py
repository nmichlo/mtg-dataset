#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

from typing import Callable
from typing import Optional
from typing import Type

import torch
import torch.nn as nn
from disent.nn.weights import init_model_weights

from examples.nn.components import Activation
from examples.nn.model import BaseAutoEncoder
from examples.nn.model import ReprDown
from examples.nn.model import ReprUp


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


@torch.no_grad()
def compute_gamma(activation_fn: Callable[[torch.Tensor], torch.Tensor], batch_size: int = 1024, samples: int = 256, device=None) -> float:
    # from appendix D: https://arxiv.org/pdf/2101.08692.pdf
    x = torch.randn(batch_size, samples, dtype=torch.float32, device=device)
    y = activation_fn(x)
    gamma = torch.mean(torch.var(y, dim=1))**-0.5
    return gamma.item()


def replace_conv(module: nn.Module, conv_class: Type[nn.Conv2d]):
    for name, mod in module.named_children():
        target_mod = getattr(module, name)
        if isinstance(mod, nn.Conv2d):
            print(f'replaced conv layer: {name}')
            conv_instance = conv_class(
                in_channels=target_mod.in_channels,
                out_channels=target_mod.out_channels,
                kernel_size=target_mod.kernel_size,
                stride=target_mod.stride,
                padding=target_mod.padding,
                dilation=target_mod.dilation,
                groups=target_mod.groups,
                bias=hasattr(target_mod, 'bias'),
                # padding_mode=target_mod.padding_mode,
            )
            setattr(module, name, conv_instance)
        if isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d, nn.GroupNorm)):
            print(f'replaced norm layer: {name}')
            setattr(module, name, nn.Identity())
    for name, mod in module.named_children():
        replace_conv(mod, conv_class)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def Conv(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)


def ConvTranspose(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
    assert stride == 2
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)


def SingleConv(in_channels: int, out_channels: int):
    return Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


def ConvDown(in_channels: int, out_channels: int, kernel_size: int = 4, pool='stride'):
    if pool == 'stride':
        return Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2)
    elif pool in ('max', 'ave'):
        return nn.Sequential(
            Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.AvgPool2d(kernel_size=2) if (pool == 'ave') else nn.MaxPool2d(kernel_size=2),
        )
    else:
        raise KeyError('invalid pool type')


def ConvUp(in_channels: int, out_channels: int, kernel_size: int = 4, upsample='stride'):
    if upsample == 'stride':
        return ConvTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2)
    elif upsample in ('nearest', 'bilinear'):
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2) if (upsample == 'nearest') else nn.UpsamplingBilinear2d(scale_factor=2),
            Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        )


# ========================================================================= #
# AE                                                                        #
# ========================================================================= #


class AutoEncoderSkips(BaseAutoEncoder):

    def __init__(self, z_size: int = 128, c_repr: int = 16, repr_hidden_size: Optional[int] = None, channel_mul=1.5, channel_start=16, channel_dec_mul: float = 1.0, skip_mode='some', skip_upsample='bilinear', skip_downsample='ave', upsample='stride', downsample='stride', activation='leaky_relu', norm='instance', sigmoid_out=False, weight_init=None):
        super().__init__()

        # weight scaled
        weight_scale, norm = {
            'scaled_std': ('scaled_std', None),
            'ws': ('ws', None),
        }.get(norm, (None, norm))

        # HELPER

        def ce(i: int):
            # encoder channel levels
            return int(channel_start * (channel_mul**i))

        def cd(i: int):
            # decoder channel levels
            return int((channel_dec_mul * channel_start) * (channel_mul**i))

        def Act(shape_or_features):
            return Activation(shape_or_features=shape_or_features, activation=activation, norm=norm)

        # skip connection upsampling/downsampling
        Pooling  = {'ave':      nn.AvgPool2d,            'max':     nn.MaxPool2d}[skip_downsample]
        Upsample = {'bilinear': nn.UpsamplingBilinear2d, 'nearest': nn.UpsamplingNearest2d}[skip_upsample]
        # convolution upsampling/downsampling
        pool     = {'stride': 'stride', 'max':     'max',     'ave':      'ave'}[downsample]
        upsample = {'stride': 'stride', 'nearest': 'nearest', 'bilinear': 'bilinear'}[upsample]
        # convolution kernel sizes
        down_k = 4 if (pool     == 'stride') else 3
        up_k   = 4 if (upsample == 'stride') else 3

        # ENCODER

        class _EncSkips(nn.Module):
            def __init__(self):
                super().__init__()
                # encoder skips
                self.enc1 = ConvDown(in_channels=3,     out_channels=ce(0),   kernel_size=down_k, pool=pool);   self.act1 = Act(shape_or_features=[ce(0), 112, 80])
                self.enc2 = ConvDown(in_channels=ce(0), out_channels=ce(1),   kernel_size=down_k, pool=pool);   self.act2 = Act(shape_or_features=[ce(1),  56, 40])
                self.enc3 = ConvDown(in_channels=ce(1), out_channels=ce(2),   kernel_size=down_k, pool=pool);   self.act3 = Act(shape_or_features=[ce(2),  28, 20])
                self.enc4 = ConvDown(in_channels=ce(2), out_channels=ce(3),   kernel_size=down_k, pool=pool);   self.act4 = Act(shape_or_features=[ce(3),  14, 10])
                self.enc5 = ConvDown(in_channels=ce(3), out_channels=c_repr, kernel_size=down_k, pool=pool); self.act5   = Act(shape_or_features=[c_repr, 7,  5])
                # skip_connections starting at x1
                self.s1_2 = nn.Sequential(Pooling(kernel_size=2),  SingleConv(ce(0), ce(1)))
                self.s1_3 = nn.Sequential(Pooling(kernel_size=4),  SingleConv(ce(0), ce(2)))
                self.s1_4 = nn.Sequential(Pooling(kernel_size=8),  SingleConv(ce(0), ce(3)))
                self.s1_5 = nn.Sequential(Pooling(kernel_size=16), SingleConv(ce(0), c_repr))
                # skip_connections starting at x2
                self.s2_3 = nn.Sequential(Pooling(kernel_size=2), SingleConv(ce(1), ce(2)))
                self.s2_4 = nn.Sequential(Pooling(kernel_size=4), SingleConv(ce(1), ce(3)))
                self.s2_5 = nn.Sequential(Pooling(kernel_size=8), SingleConv(ce(1), c_repr))
                # skip_connections starting at x3
                self.s3_4 = nn.Sequential(Pooling(kernel_size=2), SingleConv(ce(2), ce(3)))
                self.s3_5 = nn.Sequential(Pooling(kernel_size=4), SingleConv(ce(2), c_repr))
                # skip_connections starting at x4
                self.s4_5 = nn.Sequential(Pooling(kernel_size=2), SingleConv(ce(3), c_repr))

            if skip_mode == 'all':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1) + self.s1_2(x1))
                    x3 = self.act3(self.enc3(x2) + self.s1_3(x1) + self.s2_3(x2))
                    x4 = self.act4(self.enc4(x3) + self.s1_4(x1) + self.s2_4(x2) + self.s3_4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s1_5(x1) + self.s2_5(x2) + self.s3_5(x3) + self.s4_5(x4))
                    return x5
            elif skip_mode == 'none':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4))
                    return x5
            elif skip_mode == 'inner':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s1_5(x1) + self.s2_5(x2) + self.s3_5(x3) + self.s4_5(x4))
                    return x5
            elif skip_mode in 'inner_some':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s1_5(x1) + self.s2_5(x2) + self.s3_5(x3))
                    return x5

        # DECODER

        class _DecSkips(nn.Module):
            def __init__(self):
                super().__init__()
                # decoder skips
                self.dec1 = ConvUp(in_channels=c_repr, out_channels=cd(3),  kernel_size=up_k, upsample=upsample); self.act1 = Act(shape_or_features=[cd(3),  14, 10])
                self.dec2 = ConvUp(in_channels=cd(3),  out_channels=cd(2),  kernel_size=up_k, upsample=upsample); self.act2 = Act(shape_or_features=[cd(2),  28, 20])
                self.dec3 = ConvUp(in_channels=cd(2),  out_channels=cd(1),  kernel_size=up_k, upsample=upsample); self.act3 = Act(shape_or_features=[cd(1),  56, 40])
                self.dec4 = ConvUp(in_channels=cd(1),  out_channels=cd(0),  kernel_size=up_k, upsample=upsample); self.act4 = Act(shape_or_features=[cd(0), 112, 80])
                self.dec5 = ConvUp(in_channels=cd(0),  out_channels=cd(-1), kernel_size=up_k, upsample=upsample); self.act5 = Act(shape_or_features=None)  # we don't want normalisation here
                self.out  =   Conv(in_channels=cd(-1), out_channels=3,      kernel_size=3)
                # skip_connections starting at x0
                self.s0_1 = nn.Sequential(Upsample(scale_factor=2),  SingleConv(c_repr, cd(3)))
                self.s0_2 = nn.Sequential(Upsample(scale_factor=4),  SingleConv(c_repr, cd(2)))
                self.s0_3 = nn.Sequential(Upsample(scale_factor=8),  SingleConv(c_repr, cd(1)))
                self.s0_4 = nn.Sequential(Upsample(scale_factor=16), SingleConv(c_repr, cd(0)))
                self.s0_5 = nn.Sequential(Upsample(scale_factor=32), SingleConv(c_repr, cd(-1)))
                # skip_connections starting at x1
                self.s1_2 = nn.Sequential(Upsample(scale_factor=2),  SingleConv(cd(3), cd(2)))
                self.s1_3 = nn.Sequential(Upsample(scale_factor=4),  SingleConv(cd(3), cd(1)))
                self.s1_4 = nn.Sequential(Upsample(scale_factor=8),  SingleConv(cd(3), cd(0)))
                self.s1_5 = nn.Sequential(Upsample(scale_factor=16),  SingleConv(cd(3), cd(-1)))
                # skip_connections starting at x2
                self.s2_3 = nn.Sequential(Upsample(scale_factor=2), SingleConv(cd(2), cd(1)))
                self.s2_4 = nn.Sequential(Upsample(scale_factor=4), SingleConv(cd(2), cd(0)))
                self.s2_5 = nn.Sequential(Upsample(scale_factor=8), SingleConv(cd(2), cd(-1)))
                # skip_connections starting at x3
                self.s3_4 = nn.Sequential(Upsample(scale_factor=2), SingleConv(cd(1), cd(0)))
                self.s3_5 = nn.Sequential(Upsample(scale_factor=4), SingleConv(cd(1), cd(-1)))

            if skip_mode == 'all':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0) + self.s0_1(z0))
                    z2 = self.act2(self.dec2(z1) + self.s0_2(z0) + self.s1_2(z1))
                    z3 = self.act3(self.dec3(z2) + self.s0_3(z0) + self.s1_3(z1) + self.s2_3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s0_4(z0) + self.s1_4(z1) + self.s2_4(z2) + self.s3_4(z3))
                    z5 = self.act5(self.dec5(z4))
                    return self.out(z5)
            elif skip_mode == 'none':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0))
                    z2 = self.act2(self.dec2(z1))
                    z3 = self.act3(self.dec3(z2))
                    z4 = self.act4(self.dec4(z3))
                    z5 = self.act5(self.dec5(z4))
                    return self.out(z5)
            elif skip_mode == 'inner':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0))
                    z2 = self.act2(self.dec2(z1))
                    z3 = self.act3(self.dec3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s0_4(z0) + self.s1_4(z1) + self.s2_4(z2) + self.s3_4(z3))
                    z5 = self.act5(self.dec5(z4))
                    return self.out(z5)
            elif skip_mode == 'inner_some':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0))
                    z2 = self.act2(self.dec2(z1))
                    z3 = self.act3(self.dec3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s0_4(z0) + self.s1_4(z1) + self.s2_4(z2))
                    z5 = self.act5(self.dec5(z4))
                    return self.out(z5)

        # BASE MODES

        self.__enc = nn.Sequential(
            _EncSkips(),
            ReprDown(in_shape=[c_repr, 7, 5], hidden_size=repr_hidden_size, out_size=z_size * 2),
        )

        self.__dec = nn.Sequential(
            ReprUp(in_size=z_size, hidden_size=repr_hidden_size, out_shape=[c_repr, 7, 5]),
            _DecSkips(),
            *([nn.Sigmoid()] if sigmoid_out else []),
        )

        # Weight Scale The Model
        if weight_scale is not None:
            import nfnets
            gamma = compute_gamma(Act(None))
            print('GAMMA:', gamma)
            replace_conv(self, conv_class={
                'scaled_std': lambda *args, **kwargs: nfnets.ScaledStdConv2d(*args, **kwargs, gamma=gamma),
                'ws': nfnets.WSConv2d,
            }[weight_scale])

        # initialise weights
        if weight_init is not None:
            init_model_weights(self, mode=weight_init)

    @property
    def _enc(self):
        return self.__enc

    @property
    def _dec(self):
        return self.__dec


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
