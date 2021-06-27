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
from typing import Sequence
from typing import Union

import torch
from torch import nn as nn
from torch.distributions import Normal


# ========================================================================= #
# Weights & Biases                                                          #
# ========================================================================= #


class BiasWeight(nn.Module):

    def __init__(self, input_shape: Sequence[int]):
        super().__init__()
        self._bias = nn.Parameter(torch.zeros(*input_shape, dtype=torch.float32), requires_grad=True)
        self._weight = nn.Parameter(torch.ones(*input_shape, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return self._weight[None, ...] * x + self._bias[None, ...]


class Weight(nn.Module):

    def __init__(self, input_shape: Sequence[int]):
        super().__init__()
        self._weight = nn.Parameter(torch.ones(*input_shape, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return self._weight[None, ...] * x


class Bias(nn.Module):

    def __init__(self, input_shape: Sequence[int]):
        super().__init__()
        self._bias = nn.Parameter(torch.zeros(*input_shape, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x + self._bias[None, ...]


# ========================================================================= #
# Debug                                                                     #
# ========================================================================= #


class PrintLayer(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        if self.name is None:
            print(x.shape)
        else:
            print(self.name, x.shape)
        return x


# ========================================================================= #
# Activations                                                               #
# ========================================================================= #


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def Activation(shape_or_features: Optional[Union[Sequence[int], int]] = None, activation: Optional[str] = 'leaky_relu', norm: Optional[str] = 'instance'):
    layers = []
    # make norm layer
    if (norm is not None) and (shape_or_features is not None):
        # get components
        if isinstance(shape_or_features, int):
            C, H, W = shape_or_features, None, None
        else:
            C, H, W = shape_or_features
        # get norm layers
        if norm == 'instance':    layers.append(nn.InstanceNorm2d(num_features=C, momentum=0.05))
        elif norm == 'batch':     layers.append(nn.BatchNorm2d(num_features=C,    momentum=0.05))
        elif norm == 'layer_hw': layers.append(nn.LayerNorm(normalized_shape=[H, W]))
        elif norm == 'layer_chw': layers.append(nn.LayerNorm(normalized_shape=[C, H, W]))
        else:                     raise KeyError(f'invalid norm mode: {norm}')
    # make activation
    if activation is not None:
        if activation == 'swish':        layers.append(Swish())
        elif activation == 'leaky_relu': layers.append(nn.LeakyReLU(inplace=True))
        elif activation == 'relu':       layers.append(nn.ReLU(inplace=True))
        elif activation == 'relu66':     layers.append(nn.ReLU(inplace=True))
        else:                            raise KeyError(f'invalid activation mode: {activation}')
    # return model
    if layers: return nn.Sequential(*layers)
    else:      return nn.Identity()


# ========================================================================= #
# Conv                                                                      #
# ========================================================================= #


def SingleConv(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


# ========================================================================= #
# DISTS                                                                     #
# ========================================================================= #


class NormalDist(nn.Module):
    def forward(self, x):
        assert x.ndim == 2
        mu, log_var = x.chunk(2, dim=1)
        return Normal(loc=mu, scale=torch.exp(0.5 * log_var))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
