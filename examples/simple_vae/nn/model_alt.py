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

from typing import Optional
import torch.nn as nn

from examples.simple_vae.nn.helper import Activation
from examples.simple_vae.nn.model import BaseAutoEncoder
from examples.simple_vae.nn.model import ReprDown
from examples.simple_vae.nn.model import ReprUp


def SingleConv(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

def Conv(in_channels: int, out_channels: int, kernel_size: int = 3):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

def ConvDown(in_channels: int, out_channels: int, kernel_size: int = 4):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)

def ConvUp(in_channels: int, out_channels: int, kernel_size: int = 4):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)


# ========================================================================= #
# AE                                                                        #
# ========================================================================= #


class AutoEncoderSkips(BaseAutoEncoder):

    def __init__(self, z_size: int = 128, repr_channels: int = 16, repr_hidden_size: Optional[int] = None, channel_mul=1.5, channel_start=16, skip_mode='some', smooth_upsample=True, smooth_downsample=True, sigmoid_out=False):
        super().__init__()

        def c(i: int):
            return int(channel_start * (channel_mul**i))

        Pool = nn.AvgPool2d if smooth_downsample else nn.MaxPool2d

        class _EncSkips(nn.Module):
            def __init__(self):
                super().__init__()
                # encoder skips
                self.enc1 = ConvDown(in_channels=3,    out_channels=c(0));          self.act1 = Activation(norm_features=c(0))
                self.enc2 = ConvDown(in_channels=c(0), out_channels=c(1));          self.act2 = Activation(norm_features=c(1))
                self.enc3 = ConvDown(in_channels=c(1), out_channels=c(2));          self.act3 = Activation(norm_features=c(2))
                self.enc4 = ConvDown(in_channels=c(2), out_channels=c(3));          self.act4 = Activation(norm_features=c(3))
                self.enc5 = ConvDown(in_channels=c(3), out_channels=repr_channels); self.act5 = Activation(norm_features=repr_channels)
                # skip_connections starting at x0
                self.s0_1 = nn.Sequential(Pool(kernel_size=2),  SingleConv(3, c(0)))
                self.s0_2 = nn.Sequential(Pool(kernel_size=4),  SingleConv(3, c(1)))
                self.s0_3 = nn.Sequential(Pool(kernel_size=8),  SingleConv(3, c(2)))
                self.s0_4 = nn.Sequential(Pool(kernel_size=16), SingleConv(3, c(3)))
                self.s0_5 = nn.Sequential(Pool(kernel_size=32), SingleConv(3, repr_channels))
                # skip_connections starting at x1
                self.s1_2 = nn.Sequential(Pool(kernel_size=2),  SingleConv(c(0), c(1)))
                self.s1_3 = nn.Sequential(Pool(kernel_size=4),  SingleConv(c(0), c(2)))
                self.s1_4 = nn.Sequential(Pool(kernel_size=8),  SingleConv(c(0), c(3)))
                self.s1_5 = nn.Sequential(Pool(kernel_size=16), SingleConv(c(0), repr_channels))
                # skip_connections starting at x2
                self.s2_3 = nn.Sequential(Pool(kernel_size=2), SingleConv(c(1), c(2)))
                self.s2_4 = nn.Sequential(Pool(kernel_size=4), SingleConv(c(1), c(3)))
                self.s2_5 = nn.Sequential(Pool(kernel_size=8), SingleConv(c(1), repr_channels))
                # skip_connections starting at x3
                self.s3_4 = nn.Sequential(Pool(kernel_size=2), SingleConv(c(2), c(3)))
                self.s3_5 = nn.Sequential(Pool(kernel_size=4), SingleConv(c(2), repr_channels))
                # skip_connections starting at x4
                self.s4_5 = nn.Sequential(Pool(kernel_size=2), SingleConv(c(3), repr_channels))

            if skip_mode == 'all':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0) + self.s0_1(x0))
                    x2 = self.act2(self.enc2(x1) + self.s0_2(x0) + self.s1_2(x1))
                    x3 = self.act3(self.enc3(x2) + self.s0_3(x0) + self.s1_3(x1) + self.s2_3(x2))
                    x4 = self.act4(self.enc4(x3) + self.s0_4(x0) + self.s1_4(x1) + self.s2_4(x2) + self.s3_4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s0_5(x0) + self.s1_5(x1) + self.s2_5(x2) + self.s3_5(x3) + self.s4_5(x4))
                    return x5 # TODO: skip connections might be better here
            elif skip_mode == 'all_not_end':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1) + self.s1_2(x1))
                    x3 = self.act3(self.enc3(x2) + self.s1_3(x1) + self.s2_3(x2))
                    x4 = self.act4(self.enc4(x3) + self.s1_4(x1) + self.s2_4(x2) + self.s3_4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s1_5(x1) + self.s2_5(x2) + self.s3_5(x3) + self.s4_5(x4))
                    return x5 # TODO: skip connections might be better here
            elif skip_mode == 'next_all':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0) + self.s0_1(x0))
                    x2 = self.act2(self.enc2(x1) + self.s1_2(x1))
                    x3 = self.act3(self.enc3(x2) + self.s2_3(x2))
                    x4 = self.act4(self.enc4(x3) + self.s3_4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s4_5(x4))
                    return x5
            elif skip_mode == 'next_mid':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2) + self.s1_3(x1))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s3_5(x3))
                    return x5 # TODO: skip connections might be better here
            elif skip_mode == 'none':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4))
                    return x5 # TODO: skip connections might be better here
            elif skip_mode == 'inner':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s0_5(x0) + self.s1_5(x1) + self.s2_5(x2) + self.s3_5(x3) + self.s4_5(x4))
                    return x5 # TODO: skip connections might be better here
            elif skip_mode == 'inner_mid':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3))
                    x5 = self.act5(self.enc5(x4) + self.s1_5(x1) + self.s3_5(x3))
                    return x5  # TODO: skip connections might be better here
            elif skip_mode == 'inner_alt':
                def forward(self, x0):
                    x1 = self.act1(self.enc1(x0))
                    x2 = self.act2(self.enc2(x1))
                    x3 = self.act3(self.enc3(x2))
                    x4 = self.act4(self.enc4(x3) + self.s0_4(x0) + self.s1_4(x1) + self.s2_4(x2) + self.s3_4(x3))
                    x5 = self.act5(self.enc5(x4))
                    return x5

        Upsampler = nn.UpsamplingBilinear2d if smooth_upsample else nn.UpsamplingNearest2d

        class _DecSkips(nn.Module):
            def __init__(self):
                super().__init__()
                # decoder skips
                self.dec1 = ConvUp(in_channels=repr_channels, out_channels=c(3)); self.act1 = Activation(norm_features=c(3))
                self.dec2 = ConvUp(in_channels=c(3), out_channels=c(2));          self.act2 = Activation(norm_features=c(2))
                self.dec3 = ConvUp(in_channels=c(2), out_channels=c(1));          self.act3 = Activation(norm_features=c(1))
                self.dec4 = ConvUp(in_channels=c(1), out_channels=c(0));          self.act4 = Activation(norm_features=c(0))
                self.dec5 = ConvUp(in_channels=c(0), out_channels=c(-1));         self.act5 = Activation(norm_features=None)
                self.out = Conv(in_channels=c(-1), out_channels=3, kernel_size=3)
                # skip_connections starting at x0
                self.s0_1 = nn.Sequential(Upsampler(scale_factor=2),  SingleConv(repr_channels, c(3)))
                self.s0_2 = nn.Sequential(Upsampler(scale_factor=4),  SingleConv(repr_channels, c(2)))
                self.s0_3 = nn.Sequential(Upsampler(scale_factor=8),  SingleConv(repr_channels, c(1)))
                self.s0_4 = nn.Sequential(Upsampler(scale_factor=16), SingleConv(repr_channels, c(0)))
                self.s0_5 = nn.Sequential(Upsampler(scale_factor=32), SingleConv(repr_channels, c(-1)))
                # skip_connections starting at x1
                self.s1_2 = nn.Sequential(Upsampler(scale_factor=2),  SingleConv(c(3), c(2)))
                self.s1_3 = nn.Sequential(Upsampler(scale_factor=4),  SingleConv(c(3), c(1)))
                self.s1_4 = nn.Sequential(Upsampler(scale_factor=8),  SingleConv(c(3), c(0)))
                self.s1_5 = nn.Sequential(Upsampler(scale_factor=16), SingleConv(c(3), c(-1)))
                # skip_connections starting at x2
                self.s2_3 = nn.Sequential(Upsampler(scale_factor=2), SingleConv(c(2), c(1)))
                self.s2_4 = nn.Sequential(Upsampler(scale_factor=4), SingleConv(c(2), c(0)))
                self.s2_5 = nn.Sequential(Upsampler(scale_factor=8), SingleConv(c(2), c(-1)))
                # skip_connections starting at x3
                self.s3_4 = nn.Sequential(Upsampler(scale_factor=2), SingleConv(c(1), c(0)))
                self.s3_5 = nn.Sequential(Upsampler(scale_factor=4), SingleConv(c(1), c(-1)))
                # skip_connections starting at x4
                self.s4_5 = nn.Sequential(Upsampler(scale_factor=2), SingleConv(c(0), c(-1)))

            if skip_mode == 'all':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0) + self.s0_1(z0))
                    z2 = self.act2(self.dec2(z1) + self.s0_2(z0) + self.s1_2(z1))
                    z3 = self.act3(self.dec3(z2) + self.s0_3(z0) + self.s1_3(z1) + self.s2_3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s0_4(z0) + self.s1_4(z1) + self.s2_4(z2) + self.s3_4(z3))
                    z5 = self.act5(self.dec5(z4) + self.s0_5(z0) + self.s1_5(z1) + self.s2_5(z2) + self.s3_5(z3) + self.s4_5(z4))
                    return self.out(z5)
            elif skip_mode == 'all_not_end':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0) + self.s0_1(z0))
                    z2 = self.act2(self.dec2(z1) + self.s0_2(z0) + self.s1_2(z1))
                    z3 = self.act3(self.dec3(z2) + self.s0_3(z0) + self.s1_3(z1) + self.s2_3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s0_4(z0) + self.s1_4(z1) + self.s2_4(z2) + self.s3_4(z3))
                    z5 = self.act5(self.dec5(z4))
                    return self.out(z5)
            elif skip_mode == 'next_all':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0) + self.s0_1(z0))
                    z2 = self.act2(self.dec2(z1) + self.s1_2(z1))
                    z3 = self.act3(self.dec3(z2) + self.s2_3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s3_4(z3))
                    z5 = self.act5(self.dec5(z4) + self.s4_5(z4))
                    return self.out(z5)
            elif skip_mode == 'next_mid':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0))
                    z2 = self.act2(self.dec2(z1))
                    z3 = self.act3(self.dec3(z2) + self.s1_3(z1))
                    z4 = self.act4(self.dec4(z3))
                    z5 = self.act5(self.dec5(z4) + self.s3_5(z3))
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
                    z4 = self.act4(self.dec4(z3))
                    z5 = self.act5(self.dec5(z4) + self.s0_5(z0) + self.s1_5(z1) + self.s2_5(z2) + self.s3_5(z3) + self.s4_5(z4)) # TODO: THIS SHOULD BE ONE EARLIER?
                    return self.out(z5)
            elif skip_mode == 'inner_mid':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0))
                    z2 = self.act2(self.dec2(z1))
                    z3 = self.act3(self.dec3(z2))
                    z4 = self.act4(self.dec4(z3))
                    z5 = self.act5(self.dec5(z4) + self.s1_5(z1) + self.s3_5(z3))  # TODO: THIS SHOULD BE ONE EARLIER?
                    return self.out(z5)
            elif skip_mode == 'inner_alt':
                def forward(self, z0):
                    z1 = self.act1(self.dec1(z0))
                    z2 = self.act2(self.dec2(z1))
                    z3 = self.act3(self.dec3(z2))
                    z4 = self.act4(self.dec4(z3) + self.s0_4(z0) + self.s1_4(z1) + self.s2_4(z2) + self.s3_4(z3))
                    z5 = self.act5(self.dec5(z4))
                    return self.out(z5)

        # BASE MODES

        self.__enc = nn.Sequential(
            _EncSkips(),
            ReprDown(in_size=repr_channels * 7 * 5, hidden_size=repr_hidden_size, out_size=z_size * 2),
        )

        self.__dec = nn.Sequential(
            ReprUp(in_size=z_size, hidden_size=repr_hidden_size, out_shape=[repr_channels, 7, 5]),
            _DecSkips(),
            *([nn.Sigmoid()] if sigmoid_out else []),
        )

    @property
    def _enc(self):
        return self.__enc

    @property
    def _dec(self):
        return self.__dec


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
