from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import seaborn as sns
from torch.nn import functional as F
from tqdm import tqdm

from examples.nn.components import Swish
from examples.util.iter import iter_pairs


def init_model_weights(model: nn.Module, mode='xavier_normal', verbose=False):
    # get default mode
    if mode is None:
        mode = 'default'
    # print function
    count = 0
    def _init(init: bool, m: nn.Module, attr: str, mode: str):
        w = getattr(m, attr)
        if init and init_weights(w, mode=mode):
            if verbose: print(f'| {count:03d} \033[92mINIT\033[0m: {m.__class__.__name__}:{attr:15s} | {mode:20s}: {repr(list(w.shape)):15s} | {w.mean()} {w.std()} {w.min()} {w.max()}')
        else:
            if verbose: print(f'| {count:03d} \033[91mSKIP\033[0m: {m.__class__.__name__}:{attr:15s} | {mode:20s}: {repr(list(w.shape)):15s} | {w.mean()} {w.std()} {w.min()} {w.max()}')
    # init function
    def apply_init(m):
        nonlocal count
        init, count = False, count + 1
        # actually initialise!
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            _init(True, m, 'weight', mode=mode)
            _init(True, m, 'bias', mode='zeros')
    # apply weights
    if verbose: print(f'Initialising Model Layers: {mode}')
    return model.apply(apply_init)


_ALLOWED = {
    # 'uniform_',
    # 'normal_',
    # 'trunc_normal_',
    # 'constant_',
    # 'ones_',
    'zeros_',
    # 'eye_',
    # 'dirac_',
    'xavier_uniform_',
    'xavier_normal_',
    'kaiming_uniform_',
    'kaiming_normal_',
    'orthogonal_',
    # 'sparse_',
}


def init_weights(tensor, mode=None):
    if mode in (None, 'default'):
        return False
    elif mode == 'custom':
        init_custom(tensor)
    elif f'{mode}_' in _ALLOWED:
        getattr(torch.nn.init, f'{mode}_')(tensor)
    else:
        raise KeyError(f'invalid weight init mode: {repr(mode)}')
    return True


def stat(x, name: str = 'layer'):
    return {
        'std': x.std().item(),
        'mean': x.mean().item(),
        'min': x.min().item(),
        'max': x.max().item(),
        'name': name,
    }


def linear_layers(sizes: Sequence[int]):
    return nn.ModuleList([nn.Linear(prev, next) for prev, next in iter_pairs(sizes)])


class SequentialStatTracker(nn.Module):

    def __init__(self, layers, activation=None, normalize=True, include_pre_act=False):
        super().__init__()
        self._layers = nn.ModuleList(layers)
        self._activation = activation
        self._norm = normalize
        self._include_pre_act = include_pre_act

    def forward(self, x, include_pre_act=False):
        stats = [stat(x, name='x0')]
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if self._include_pre_act:
                stats.append(stat(x, name=f'x{i+1}'))
            if self._activation is not None:
                if self._norm:
                    x = norm_activation(x, self._activation)
                else:
                    x = self._activation(x)
                stats.append(stat(x, name=f'x{i+1}_a'))
        return x, stats


_NORM_CONSTS = {}


def norm_activation(x, activation=None, samples=16384):
    if activation.__class__ not in _NORM_CONSTS:
        act = activation(torch.randn(samples))
        _NORM_CONSTS[activation.__class__] = (act.mean().item(), act.std().item())
    # normalise
    mean, std = _NORM_CONSTS[activation.__class__]
    return (activation(x) - mean) / std


def init_custom(tensor: torch.Tensor):
    assert tensor.ndim == 2
    O, I = tensor.shape
    with torch.no_grad():
        tensor[...] = torch.randn_like(tensor)
        tensor -= tensor.mean(dim=0, keepdim=True)
        tensor /= tensor.std(dim=0, keepdim=True)
        # tensor *=
        tensor *= np.sqrt(1 / I)


if __name__ == '__main__':

    def sample_forward_random(sizes, title: str, batch_size=128, n_samples=100, activation=None, norm=False, include_pre_act=False):
        # generate values
        samples = []
        for mode in tqdm([
            'xavier_uniform',
            'xavier_normal',
            'custom',
            # 'kaiming_normal',
        ], desc=title):
            for _ in range(n_samples):
                model = SequentialStatTracker(layers=linear_layers(sizes), activation=activation, normalize=norm, include_pre_act=include_pre_act)
                model = init_model_weights(model, mode=mode)
                x = torch.randn(batch_size, sizes[0])
                y, y_stats = model(x)
                for y_stat in y_stats:
                    y_stat['mode'] = mode
                samples.extend(y_stats)

        # aggregate
        return [{'title': title, 'name': s['name'], 'mode': s['mode'], 'stat': k, 'value': s[k]} for s in samples for k in {'mean', 'std'}]

    def plot_activations(sizes, batch_size=128, n_samples=50, norm=False, include_pre_act=True):
        suffix = ':norm' if norm else ''

        samples = []
        samples.extend(sample_forward_random(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=norm, include_pre_act=include_pre_act, activation=None,      title='linear' + suffix))
        # samples.extend(sample_forward_random(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=norm, include_pre_act=include_pre_act, activation=nn.Sigmoid(), title='sigmoid' + suffix))
        samples.extend(sample_forward_random(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=norm, include_pre_act=include_pre_act, activation=nn.ReLU(), title='relu' + suffix))
        # samples.extend(sample_forward_random(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=norm, include_pre_act=include_pre_act, activation=nn.ReLU6(),   title='relu6' + suffix))
        # samples.extend(sample_forward_random(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=norm, include_pre_act=include_pre_act, activation=nn.ELU(),     title='elu' + suffix))
        samples.extend(sample_forward_random(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=norm, include_pre_act=include_pre_act, activation=Swish(),   title='swish' + suffix))

        df = pd.DataFrame(samples).groupby(['title', 'mode', 'name', 'stat']).mean().reset_index()
        df = df.sort_values(['name', 'title', 'mode', 'stat']).reset_index()

        # plt.title(f'mean -- {sizes}')
        # sns.lineplot(x="name", y="value", hue="title", style='mode', data=df[df['stat'] == 'mean'])
        # plt.show()

        plt.title(f'std -- {sizes}')
        sns.lineplot(x="name", y="value", hue="title", style='mode', data=df[df['stat'] == 'std'])
        plt.show()

        return samples


    def __main__():
        batch_size = 1024
        n_samples = 100
        sizes = [4, 16, 128, 16, 4]
        plot_activations(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=False)
        plot_activations(batch_size=batch_size, n_samples=n_samples, sizes=sizes, norm=True)


    __main__()




# import numpy as np
# import torch
#
#
# if __name__ == '__main__':
#
#     inp, out = 32, 4
#     # inp, out = 4, 32
#
#
#     samples_inp, samples_out = [], []
#     for i in range(256):
#         # model
#         m = torch.nn.Linear(inp, out)
#
#         # init
#         with torch.no_grad():
#             m.bias[...] = 0
#             m.weight[...] = torch.randn_like(m.weight)
#             m.weight -= m.weight.mean(dim=-1, keepdim=True)
#             m.weight /= m.weight.std(dim=-1, keepdim=True)
#             # normalize
#             m.weight *= np.sqrt(1 / inp)
#
#         # feed forward
#         x = torch.randn(256, inp)
#         samples_inp.append([x.mean().item(), x.std().item(), x.var().item()])
#         x = m(x)
#         samples_out.append([x.mean().item(), x.std().item(), x.var().item()])
#
#     # combine samples
#     samples_inp = torch.as_tensor(samples_inp, dtype=torch.float32).mean(dim=0)
#     samples_out = torch.as_tensor(samples_out, dtype=torch.float32).mean(dim=0)
#
#     print(', '.join(f'{v:8.4f}' for v in samples_inp.numpy()))
#     print(', '.join(f'{v:8.4f}' for v in samples_out.numpy()))
