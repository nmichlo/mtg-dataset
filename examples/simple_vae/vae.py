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

import logging
import os
import warnings
import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from disent.nn.functional import get_kernel_size
from disent.nn.functional import torch_conv2d_channel_wise
from disent.nn.functional import torch_conv2d_channel_wise_fft
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from examples.common import Hdf5DataModule
from examples.common import VisualiseCallback
from examples.simple_vae.nn.model_alt import AutoEncoderSkips


def torch_laplace_kernel2d(size, sigma=1.0, normalise=True):
    # make kernel
    pos = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y = torch.meshgrid(pos, pos)
    # compute values
    norm = (x**2 + y**2) / (2 * sigma ** 2)
    # compute kernel
    kernel = - (2 - 2 * norm) * torch.exp(-norm) / (2 * np.pi * sigma ** 2)
    if normalise:
        kernel -= kernel.mean()
        # kernel /= torch.abs(kernel).sum()
    # return kernel
    return kernel[None, None, :, :]


class MseLoss(nn.Module):
    def forward(self, x, target, reduction='mean'):
        return F.mse_loss(x, target, reduction=reduction)


class SpatialFreqLoss(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/1806.02336.pdf
    """

    def __init__(self, sigmas=(0.8, 1.6, 3.2), truncate=3, fft=True):
        super().__init__()
        assert len(sigmas) > 0
        self._kernels = nn.ParameterList([
            nn.Parameter(torch_laplace_kernel2d(get_kernel_size(sigma=sigma, truncate=truncate), sigma=sigma, normalise=True), requires_grad=False)
            for sigma in sigmas
        ])
        print([k.shape for k in self._kernels])
        self._n = len(self._kernels)
        self._conv_fn = torch_conv2d_channel_wise_fft if fft else torch_conv2d_channel_wise

    def forward(self, x, target, reduction='mean'):
        loss_orig = F.mse_loss(x, target, reduction=reduction)
        loss_freq = 0
        for kernel in self._kernels:
            loss_freq += F.mse_loss(self._conv_fn(x, kernel), self._conv_fn(target, kernel), reduction=reduction)
        return (loss_orig + loss_freq) / (self._n + 1)


class LaplaceMseLoss(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/1806.02336.pdf
    """

    def __init__(self, freq_ratio=0.5):
        super().__init__()
        self._ratio = freq_ratio
        self._kernel = nn.Parameter(torch.as_tensor([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0],
        ], dtype=torch.float32), requires_grad=False)

    def forward(self, x, target, reduction='mean'):
        x_conv = torch_conv2d_channel_wise(x, self._kernel)
        t_conv = torch_conv2d_channel_wise(target, self._kernel)
        loss_orig = F.mse_loss(x, target, reduction=reduction)
        loss_freq = F.mse_loss(x_conv, t_conv, reduction=reduction)
        return (1 - self._ratio) * loss_orig + self._ratio * loss_freq


# ========================================================================= #
# Loss Reduction                                                            #
# ========================================================================= #


_REDUCE_DIMS = {
    'none': None,
    'chn': (   -3,       ),
    'pix': (       -2, -1),
    'obs': (   -3, -2, -1),
    'bch': (0,           ),
}

_REDUCE_FNS = {
    'mean': torch.mean,
    'var': torch.var,
    'std': torch.std,
}


def mean_weighted_loss(unreduced_loss, weight_mode: str = 'none', weight_reduce='obs', weight_shift=True, weight_power=1):
    # exit early
    if weight_mode == 'none':
        if weight_reduce != 'none':
            warnings.warn('`weight_reduce` has no effect when `weight_mode` == "none"')
        return unreduced_loss.mean()
    # generate weights
    with torch.no_grad():
        reduce_dims = _REDUCE_DIMS[weight_reduce]
        reduce_fn = _REDUCE_FNS[weight_mode]
        # get weights
        if reduce_dims is None:
            weights = unreduced_loss
        else:
            weights = reduce_fn(unreduced_loss, keepdim=True, dim=reduce_dims)
        # shift
        if weight_shift:
            weights = weights - torch.min(weights)
        if weight_power != 1:
            weights = weights ** weight_power
        # normalise weights
        weights = (weights / weights.mean()).detach()
    # compute version
    return (unreduced_loss * weights).mean()


# ========================================================================= #
# System                                                                    #
# ========================================================================= #


class MtgVaeSystem(pl.LightningModule):

    def get_progress_bar_dict(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def __init__(
        self,
        lr=1e-3,
        alpha=1.0,
        beta=0.003,
        # model options
        z_size=128,
        repr_hidden_size=None,
        repr_channels=16,
        channel_mul=1.5,
        channel_start=16,
        model_skip_mode='next',
        model_smooth_upsample=False,
        model_smooth_downsample=False,
        # training options
        is_vae=True,
        # loss options
        recon_loss='mse',
        recon_weight_reduce='none',
        recon_weight_mode='none',
        recon_weight_shift=True,
        recon_weight_power=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        # make model
        self.model = AutoEncoderSkips(
            z_size=self.hparams.z_size,
            repr_hidden_size=self.hparams.repr_hidden_size,
            repr_channels=self.hparams.repr_channels,
            channel_mul=self.hparams.channel_mul,
            channel_start=self.hparams.channel_start,
            #
            skip_mode=self.hparams.model_skip_mode,
            smooth_upsample=self.hparams.model_smooth_upsample,
            smooth_downsample=self.hparams.model_smooth_downsample,
            sigmoid_out=False,
        )
        # get loss
        if self.hparams.recon_loss == 'mse':
            self._loss = MseLoss()
        elif self.hparams.recon_loss == 'mse_laplace':
            self._loss = LaplaceMseLoss()
        elif self.hparams.recon_loss == 'mse_freq':
            self._loss = SpatialFreqLoss()
        else:
            raise KeyError(f'invalid recon_loss: {self.hparams.recon_loss}')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.25,
            patience=10,
            verbose=True,
            min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "recon",
        }

    def forward(self, x, deterministic=True, return_dists=False):
        return self.model.forward(x, deterministic=deterministic, return_dists=return_dists)

    def training_step(self, batch, batch_idx):
        recon, posterior, prior = self.model.forward(batch, deterministic=not self.hparams.is_vae, return_dists=True)
        # compute recon loss
        loss_recon = self.hparams.alpha * mean_weighted_loss(
            unreduced_loss=self._loss(recon, batch, reduction='none'),
            weight_reduce=self.hparams.recon_weight_reduce,
            weight_mode=self.hparams.recon_weight_mode,
            weight_shift=self.hparams.recon_weight_shift,
            weight_power=self.hparams.recon_weight_power,
        )
        # compute kl divergence
        if self.hparams.is_vae:
            loss_kl = self.hparams.beta * torch.distributions.kl_divergence(posterior, prior).mean()
        else:
            loss_kl = 0
        # combined loss
        loss = loss_recon + loss_kl
        # return loss
        self.log('kl',    loss_kl,    on_step=True, prog_bar=True)
        self.log('recon', loss_recon, on_step=True)
        self.log('loss',  loss,       on_step=True)
        return loss


class WandbContextManagerCallback(pl.Callback):

    def __init__(self, keys_values):
        self._keys_values = keys_values

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        print()
        for k, v in self._keys_values.items():
            setattr(wandb.config, f'model/{k}', v)
            print(f'model/{k}:', repr(v))
        print()

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        wandb.finish()


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main():

        # settings:
        # 5926MiB / 5932MiB (RTX2060)

        datamodule = Hdf5DataModule(
            'data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
            val_ratio=0,
            batch_size=32,
        )

        system = MtgVaeSystem(
            lr=3e-4,
            alpha=100,
            beta=0.01,
            # model options
            z_size=1024,
            repr_hidden_size=1024+128,  # 1536,
            repr_channels=128,  # 32*5*7 = 1120, 44*5*7 = 1540, 56*5*7 = 1960
            channel_mul=1.19,
            channel_start=160,
            model_skip_mode='inner',  # all, all_not_end, next_all, next_mid, none
            model_smooth_downsample=True,
            model_smooth_upsample=True,
            # training options
            is_vae=True,
            recon_loss='mse_laplace',
            recon_weight_mode='none',
            recon_weight_reduce='none',
            recon_weight_power=1,
            recon_weight_shift=False,
        )

        # initialise model trainer
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=500,
            # checkpoint_callback=False,
            resume_from_checkpoint='checkpoint_border/2021-06-09_23:14:25/epoch=37-step=69999.ckpt',
            logger=WandbLogger(
                name=f'mtg-vae__resume-69999:{system.hparams.model_skip_mode}:{system.hparams.model_smooth_downsample}:{system.hparams.model_smooth_upsample}|{datamodule._batch_size}|{system.hparams.recon_loss}:{system.hparams.recon_weight_reduce}:{system.hparams.recon_weight_mode}',
                project='MTG',
            ),
            callbacks=[
                WandbContextManagerCallback({**system.hparams, 'batch_size': datamodule._batch_size}),
                VisualiseCallback(every_n_steps=500, log_wandb=True, log_local=False),
                ModelCheckpoint(
                    dirpath=os.path.join('checkpoint_border__resume-69999', datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
                    monitor='recon',
                    every_n_train_steps=2500,
                    verbose=True,
                    save_top_k=5,
                ),
            ]
        )

        # start training model
        trainer.fit(system, datamodule)

    # ENTRYPOINT
    logging.basicConfig(level=logging.INFO)
    main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
