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
from disent.visualize.visualize_util import make_image_grid
from disent.nn.functional import torch_conv2d_channel_wise, torch_conv2d_channel_wise_fft, get_kernel_size
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from examples.simple_vae.nn.kornia import hsv_to_rgb
from examples.simple_vae.nn.kornia import rgb_to_hsv
from examples.simple_vae.nn.model_alt import AutoEncoderSkips
from mtgml.util.hdf5 import H5pyDataset
import wandb


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


def to_hsv_for_loss(x, t, augment_hsv: bool):
    if augment_hsv:
        s = torch.as_tensor([3*0.4, 3*0.1, 3*0.5], dtype=torch.float32, device=x.device)[None, :, None, None]
        x = torch.cat([x, s * rgb_to_hsv(torch.clip(x, 0, 1))], dim=1)
        t = torch.cat([t, s * rgb_to_hsv(torch.clip(t, 0, 1))], dim=1)
    return x, t


class MseLoss(nn.Module):
    def __init__(self, augment_hsv=False):
        super().__init__()
        self._augment_hsv = augment_hsv

    def forward(self, x, target, reduction='mean'):
        x, target = to_hsv_for_loss(x, target, augment_hsv=self._augment_hsv)
        return F.mse_loss(x, target, reduction=reduction)


class SpatialFreqLoss(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/1806.02336.pdf
    """

    def __init__(self, sigmas=(0.8, 1.6, 3.2), truncate=3, fft=True, augment_hsv=False):
        super().__init__()
        assert len(sigmas) > 0
        self._kernels = nn.ParameterList([
            nn.Parameter(torch_laplace_kernel2d(get_kernel_size(sigma=sigma, truncate=truncate), sigma=sigma, normalise=True), requires_grad=False)
            for sigma in sigmas
        ])
        self._augment_hsv = augment_hsv
        print([k.shape for k in self._kernels])
        self._n = len(self._kernels)
        self._conv_fn = torch_conv2d_channel_wise_fft if fft else torch_conv2d_channel_wise

    def forward(self, x, target, reduction='mean'):
        x, target = to_hsv_for_loss(x, target, augment_hsv=self._augment_hsv)
        loss_orig = F.mse_loss(x, target, reduction=reduction)
        loss_freq = 0
        for kernel in self._kernels:
            loss_freq += F.mse_loss(self._conv_fn(x, kernel), self._conv_fn(target, kernel), reduction=reduction)
        return (loss_orig + loss_freq) / (self._n + 1)


class SpatialFreqMseLoss(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/1806.02336.pdf
    """

    def __init__(self, augment_hsv=False):
        super().__init__()
        self._kernel = nn.Parameter(torch.as_tensor([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0],
        ], dtype=torch.float32), requires_grad=False)
        self._augment_hsv = augment_hsv

    def forward(self, x, target, reduction='mean'):
        x, target = to_hsv_for_loss(x, target, augment_hsv=self._augment_hsv)
        x_conv = torch_conv2d_channel_wise(x, self._kernel)
        t_conv = torch_conv2d_channel_wise(target, self._kernel)
        loss_orig = F.mse_loss(x, target, reduction=reduction)
        loss_freq = F.mse_loss(x_conv, t_conv, reduction=reduction)
        return (loss_orig + loss_freq) / 2


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


class ToTensor(object):
    def __init__(self, move_channels=True, to_hsv=False):
        self._move_channels = move_channels
        self._to_hsv = to_hsv

    def __call__(self, img):
        if self._move_channels:
            img = np.moveaxis(img, -1, -3)
        img = img.astype('float32') / 255
        img = torch.from_numpy(img)
        if self._to_hsv:
            img = rgb_to_hsv(img, scale_h=False)
        return img


class VisualiseCallback(pl.Callback):

    def __init__(self, every_n_steps=1000, use_wandb=False, is_hsv=False):
        self._count = 0
        self._every_n_steps = every_n_steps
        self._wandb = use_wandb
        self._is_hsv = is_hsv

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: STEP_OUTPUT, batch: torch.Tensor, batch_idx: int, dataloader_idx: int) -> None:
        self._count += 1
        # skip
        if self._count % self._every_n_steps != 0:
            return
        # feed forward
        with torch.no_grad():
            # generate images
            data = trainer.train_dataloader.dataset.datasets
            xs = torch.stack([data[i] for i in [3466, 18757, 20000, 40000, 22038, 20541, 1100]])
            rs = pl_module.forward(xs.to(pl_module.device), deterministic=True)
            # clip images
            xs = torch.clip(xs, 0, 1)
            rs = torch.clip(rs, 0, 1)
            # convert from hsv
            if self._is_hsv:
                xs = hsv_to_rgb(xs, scale_h=False)
                rs = hsv_to_rgb(rs, scale_h=False)
            # convert to uint8
            xs = torch.moveaxis(torch.clip(xs * 255, 0, 255).to(torch.uint8).detach().cpu(), 1, -1).numpy()
            rs = torch.moveaxis(torch.clip(rs * 255, 0, 255).to(torch.uint8).detach().cpu(), 1, -1).numpy()
            # make grid
            img = make_image_grid(np.concatenate([xs, rs]), num_cols=len(xs), pad=4)
        # plot
        if self._wandb:
            wandb.log({'mtg-recons': wandb.Image(img)})
        else:
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_axis_off()
            fig.tight_layout()
            plt.show()


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


class MtgSystem(pl.LightningModule):

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
        model_all_skips=False,
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
            all_skips=self.hparams.model_all_skips,
            smooth_upsample=self.hparams.model_smooth_upsample,
            smooth_downsample=self.hparams.model_smooth_downsample,
            sigmoid_out=False,
        )
        # get loss
        if self.hparams.recon_loss == 'mse':
            self._loss = MseLoss(augment_hsv=False)
        elif self.hparams.recon_loss == 'mse_freq':
            self._loss = SpatialFreqLoss(augment_hsv=False)
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

        dataloader = DataLoader(
            dataset=H5pyDataset(
                h5_path='data/mtg-default_cards-normal-224x160x3.h5',
                h5_dataset_name='data',
                transform=ToTensor(move_channels=True),
            ),
            num_workers=os.cpu_count(),
            batch_size=64,
            shuffle=True,
        )

        model = MtgSystem(
            lr=1e-3,
            beta=0.0001,
            # model options
            z_size=1024,
            repr_hidden_size=None, # 1536,
            repr_channels=64,  # 32*5*7 = 1120, 44*5*7 = 1540, 56*5*7 = 1960
            channel_mul=1.2,
            channel_start=80,
            model_all_skips=False,
            model_smooth_upsample=False,
            model_smooth_downsample=False,
            # training options
            is_vae=True,
            recon_loss='mse_freq',
            recon_weight_reduce='obs',
            recon_weight_mode='distscale2',
        )

        if use_wandb:
            print()
            for k, v in model.hparams.items():
                setattr(wandb.config, f'model/{k}', v)
                print(f'model/{k}:', repr(v))
            print()

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=500,
            # checkpoint_callback=False,
            logger=WandbLogger(name=f'mtg-vae|{model.hparams.recon_loss}:{model.hparams.recon_weight_reduce}:{model.hparams.recon_weight_mode}', project='MTG') if use_wandb else False,
            callbacks=[
                VisualiseCallback(every_n_steps=500, use_wandb=use_wandb),
                ModelCheckpoint(
                    dirpath=os.path.join('checkp', datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
                    monitor='recon',
                    every_n_train_steps=2500,
                    verbose=True,
                    save_top_k=5,
                ),
            ]
        )

        trainer.fit(model, dataloader)

    # RUN
    main(use_wandb=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
