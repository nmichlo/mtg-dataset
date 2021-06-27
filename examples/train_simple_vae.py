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
from typing import Callable
from typing import Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from examples.common import BaseLightningModule
from examples.common import make_mtg_datamodule
from examples.common import make_mtg_trainer
from examples.nn.loss import LaplaceMseLoss
from examples.nn.loss import MseLoss
from examples.nn.loss import SpatialFreqLoss
from examples.nn.model_alt import AutoEncoderSkips


# ========================================================================= #
# System                                                                    #
# ========================================================================= #


class MtgVaeSystem(BaseLightningModule):

    def __init__(
        self,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 0.003,
        # model options
        z_size: int = 128,
        repr_hidden_size: Optional[int] = None,
        repr_channels: int = 16,
        channel_mul: float = 1.5,
        channel_start: int = 16,
        channel_dec_mul: float = 1.0,
        model_activation: str = 'leaky_relu',
        model_norm: Optional[str] = None,
        model_weight_init: Optional[str] = None,
        model_skip_mode='next',
        model_skip_downsample: str = 'ave',     # max, ave
        model_skip_upsample: str = 'bilinear',  # nearest, bilinear
        model_downsample: str = 'stride',       # stride, max, ave
        model_upsample: str = 'stride',         # stride, nearest, bilinear
        # loss options
        recon_loss: str = 'mse',
    ):
        super().__init__()
        self.save_hyperparameters()
        # make model
        self.model = AutoEncoderSkips(
            # sizes
            z_size=self.hparams.z_size,
            repr_hidden_size=self.hparams.repr_hidden_size,
            c_repr=self.hparams.repr_channels,
            channel_mul=self.hparams.channel_mul,
            channel_start=self.hparams.channel_start,
            channel_dec_mul=self.hparams.channel_dec_mul,
            # layers
            weight_init=self.hparams.model_weight_init,
            activation=self.hparams.model_activation,
            norm=self.hparams.model_norm,
            skip_mode=self.hparams.model_skip_mode,
            skip_upsample=self.hparams.model_skip_upsample,
            skip_downsample=self.hparams.model_skip_downsample,
            upsample=self.hparams.model_upsample,
            downsample=self.hparams.model_downsample,
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        recon, posterior, prior = self.model.forward_train(batch)
        # compute recon loss
        loss_recon = self.hparams.alpha * self._loss(recon, batch, reduction='mean')
        # compute kl divergence
        loss_kl = self.hparams.beta * torch.distributions.kl_divergence(posterior, prior).mean()
        # combined loss
        loss = loss_recon + loss_kl
        # return loss
        self.log('kl',    loss_kl,    on_step=True, prog_bar=True)
        self.log('recon', loss_recon, on_step=True)
        self.log('loss',  loss,       on_step=True)
        return loss


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(data_path: str = None, resume_path: str = None):

        # settings:
        # 5926MiB / 5932MiB (RTX2060)

        system = MtgVaeSystem(
            # training options
            lr=3e-4,
            alpha=1,
            beta=0.001,
            recon_loss='mse',

            # MEDIUM MODEL: batch_size 32 -- 5898MiB -- 2.3it/s
            # z_size=768,
            # repr_hidden_size=None,  # 1024+128,
            # repr_channels=64,       # 64*7*5 == 2240
            # channel_mul=1.245,
            # channel_start=120,
            # channel_dec_mul=1.0,  # enc: 120->231, dec: 231->120

            # MEDIUM MODEL ALT: batch_size 32 -- 5898MiB -- 2.3it/s
            z_size=768,
            repr_hidden_size=None,  # 1024+128,
            repr_channels=64,       # 64*7*5 == 2240
            channel_mul=1.26,
            channel_start=96,
            channel_dec_mul=1.3334,  # enc: 96->192, dec: 256->128

            # SMALLER MODEL - batch_size=32 8.52it/s
            # z_size=256,
            # repr_hidden_size=None,  # 512,
            # repr_channels=64,  # 32*5*7 = 1120, 44*5*7 = 1540, 56*5*7 = 1960
            # channel_mul=1.25,
            # channel_start=32,

            # good model defaults
            model_weight_init=None,
            model_activation='swish',
            model_norm='batch',
            model_skip_mode='inner_some',  # inner_some, inner, all, none
            model_skip_downsample='ave',
            model_skip_upsample='bilinear',
            model_downsample='stride',
            model_upsample='stride',
        )

        # get dataset & visualise images
        mean_std = (0.5, 0.5)  # TODO: compute actual mean
        datamodule = make_mtg_datamodule(batch_size=32, load_path=data_path, mean_std=mean_std, in_memory=False)
        vis_imgs = torch.stack([datamodule.data[i] for i in [3466, 18757, 20000, 40000, 21586, 20541, 1100]])

        # start training model
        h = system.hparams
        trainer = make_mtg_trainer(
            train_epochs=500,
            resume_from_checkpoint=resume_path,
            visualize_input={'recons': (vis_imgs, mean_std)},
            wandb_project='MTG-VAE',
            wandb_name=f'mtg-vae|{h.z_size}:{h.repr_hidden_size}:{h.repr_channels}:{h.channel_start}:{h.channel_mul}:{h.channel_dec_mul}',
            wandb=True
        )
        trainer.fit(system, datamodule)

    # ENTRYPOINT
    logging.basicConfig(level=logging.INFO)
    main(
        data_path='data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
        resume_path=None,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
