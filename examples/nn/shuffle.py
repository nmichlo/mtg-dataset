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

from typing import Sequence

import numpy as np
import torch
from torch import nn as nn

from examples.nn.helper import Activation
from examples.nn.model import AutoEncoder
from examples.nn.helper import Bias
from examples.nn.model import ConvDown
from examples.nn.model import ConvUp
from examples.nn.helper import NormalDist
from examples.nn.helper import Weight


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


class DeterministicShuffle(nn.Module):
    def __init__(self, input_shape: Sequence[int]):
        super().__init__()
        size = int(np.prod(input_shape))
        # model
        indices = np.arange(size)
        np.random.shuffle(indices)
        self._indices = nn.Parameter(torch.from_numpy(indices), requires_grad=False)
        self._flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        shape = x.shape
        x = self._flatten(x)
        x = x[:, self._indices]
        x = x.resize(*shape)
        return x


class DeterministicShuffleMulti(nn.Module):

    def __init__(self, n: int, input_shape):
        super().__init__()
        assert n > 0
        self._weights = nn.ModuleList([
            nn.Sequential(
                DeterministicShuffle(input_shape),
                Weight(input_shape),
            )
            for i in range(n)
        ])
        self._bias = Bias(input_shape)

    def forward(self, x):
        output = self._weights[0](x)
        for shuffler in self._weights[1:]:
            output += shuffler(x)
        return output * (1/len(self._weights)) + self._bias._bias


# ========================================================================= #
# MODULES                                                                   #
# ========================================================================= #


def DownShuffle(input_shape, output_shape, n=4):
    return nn.Sequential(
        DeterministicShuffleMulti(n=n, input_shape=input_shape),
        nn.AvgPool2d(kernel_size=2),
        # Weight(output_shape),
        Activation(),
    )


def UpShuffle(input_shape, output_shape, n=4, last_activation=True):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        DeterministicShuffleMulti(n=n, input_shape=output_shape),
        # Weight(output_shape),
        *([Activation()] if last_activation else []),
    )


# ========================================================================= #
# AE                                                                        #
# ========================================================================= #


class ShuffleAutoEncoder(AutoEncoder):

    def __init__(self, z_size, n=16):
        super().__init__()
        self._encoder = nn.Sequential(
            ConvDown(in_channels=3, out_channels=3),
            ConvDown(in_channels=3, out_channels=3),
            # DownShuffle(input_shape=[3, 224, 160], output_shape=[3, 112,  80]),
            # DownShuffle(input_shape=[3, 112,  80], output_shape=[3,  56,  40]),
            DownShuffle(input_shape=[3,  56,  40], output_shape=[3,  28,  20], n=n),
            DownShuffle(input_shape=[3,  28,  20], output_shape=[3,  14,  10], n=n),
            DownShuffle(input_shape=[3,  14,  10], output_shape=[3,   7,   5], n=n),
            # /\ encoder
            nn.Flatten(),
            nn.Linear(3*7*5, z_size*2),
            NormalDist(),
        )
        self._decoder = nn.Sequential(
            nn.Linear(z_size, 3*7*5),
            nn.Unflatten(1, [3, 7, 5]),
            # \/ decoder
            UpShuffle(input_shape=[3,   7,   5], output_shape=[3,  14,  10], n=n),
            UpShuffle(input_shape=[3,  14,  10], output_shape=[3,  28,  20], n=n),
            UpShuffle(input_shape=[3,  28,  20], output_shape=[3,  56,  40], n=n),
            # UpShuffle(input_shape=[3,  56,  40], output_shape=[3, 112,  80]),
            # UpShuffle(input_shape=[3, 112,  80], output_shape=[3, 224, 160], last_activation=False),
            ConvUp(in_channels=3, out_channels=3),
            ConvUp(in_channels=3, out_channels=3, last_activation=False),
        )


def DeterministicShuffleAvePool(input_shape: Sequence[int], kernel_size: int = 2):
    return nn.Sequential(
        DeterministicShuffle(input_shape=input_shape),
        nn.AvgPool2d(kernel_size=kernel_size),
    )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
