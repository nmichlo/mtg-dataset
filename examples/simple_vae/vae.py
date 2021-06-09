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

import datetime
import os

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

    def __init__(self):
        super().__init__()
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
        return (loss_orig + loss_freq) / 2


# ========================================================================= #
# Loss Reduction                                                            #
# ========================================================================= #


def weight_loss(unreduced_loss, weight_mode: str = 'distscale2'):
    with torch.no_grad():
        m = torch.min(unreduced_loss, dim=0, keepdim=True).values
        M = torch.max(unreduced_loss, dim=0, keepdim=True).values
        # reduce loss
        if weight_mode == 'distscale':
            weights = (unreduced_loss - m) / (M - m)
        elif weight_mode == 'distscale2':
            weights = (unreduced_loss - m) / (M - m)
            weights = weights**2
        elif weight_mode == 'lenscale':
            weights = unreduced_loss / M
        elif weight_mode == 'lenscale2':
            weights = unreduced_loss / M
            weights = weights ** 2
        else:
            raise KeyError(f'invalid weight_mode: {weight_mode}')
        # scale weights
        weights = weights * (len(unreduced_loss) / weights.sum(dim=0, keepdim=True))
    # scale loss
    return unreduced_loss * weights


def mean_weighted_loss(unreduced_loss, weight_mode: str = 'mean', weight_reduce='obs'):
    if weight_mode == 'none':
        return unreduced_loss.mean()
    # observation weighted losses
    if weight_reduce == 'none':
        reduced_loss = unreduced_loss
    elif weight_reduce == 'chn':
        reduced_loss = unreduced_loss.mean(keepdim=True, dim=(-3,       ))
    elif weight_reduce == 'pix':
        reduced_loss = unreduced_loss.mean(keepdim=True, dim=(    -2, -1))
    elif weight_reduce == 'obs':
        reduced_loss = unreduced_loss.mean(keepdim=True, dim=(-3, -2, -1))
    else:
        raise KeyError(f'invalid weight_reduce: {weight_reduce}')
    # compute version
    return weight_loss(reduced_loss, weight_mode=weight_mode).mean()


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
        )
        # compute kl divergence
        if self.hparams.is_vae:
            loss_kl = self.hparams.beta * torch.distributions.kl_divergence(posterior, prior).mean()
        else:
            loss_kl = 0
        # combined loss
        loss = loss_recon + loss_kl
        # return loss
        self.log('kl', loss_kl, prog_bar=True)
        self.log('recon', loss_recon)
        self.log('loss', loss)
        return loss


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(use_wandb=False):

        datamodule = Hdf5DataModule(
            'data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
            val_ratio=0,
            batch_size=64-32,
        )

        system = MtgVaeSystem(
            lr=1e-3,
            alpha=100,
            beta=0.0001,
            # model options
            z_size=1024,
            repr_hidden_size=None,  # 1536,
            repr_channels=64,  # 32*5*7 = 1120, 44*5*7 = 1540, 56*5*7 = 1960
            channel_mul=1.2,
            channel_start=80,
            model_skip_mode='inner',  # all, all_not_end, next_all, next_mid, none
            model_smooth_downsample=True,
            model_smooth_upsample=False,
            # training options
            is_vae=True,
            recon_loss='mse_laplace',
            recon_weight_reduce='mean',
            recon_weight_mode='none',
        )

        if use_wandb:
            print()
            for k, v in system.hparams.items():
                setattr(wandb.config, f'model/{k}', v)
                print(f'model/{k}:', repr(v))
            print()

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=500,
            # checkpoint_callback=False,
            logger=WandbLogger(
                name=f'mtg-vae:{system.hparams.model_skip_mode}:{system.hparams.model_smooth_downsample}:{system.hparams.model_smooth_upsample}|{system.hparams.recon_loss}:{system.hparams.recon_weight_reduce}:{system.hparams.recon_weight_mode}',
                project='MTG'
            ) if use_wandb else False,
            callbacks=[
                VisualiseCallback(every_n_steps=500, use_wandb=use_wandb),
                ModelCheckpoint(
                    dirpath=os.path.join('checkpoint_border', datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
                    monitor='recon',
                    every_n_train_steps=2500,
                    verbose=True,
                    save_top_k=5,
                ),
            ]
        )

        trainer.fit(system, datamodule)

    # RUN
    main(use_wandb=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
