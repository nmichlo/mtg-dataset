"""
port and customised version of:
https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/dcgan.py
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning.core import LightningModule


logger = logging.getLogger(__name__)


# ========================================================================= #
# Generator                                                                 #
# ========================================================================= #


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, final_activation='tanh'):
        super().__init__()
        self.img_shape = img_shape

        # size of downscaled image
        C, H, W = img_shape
        scale = 2**2
        assert H % scale == 0
        assert W % scale == 0

        final_act = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity(),
        }[final_activation]

        self._generator = nn.Sequential(
            # LINEAR NET
            nn.Linear(in_features=latent_dim, out_features=128 * (H // scale) * (W // scale)),
            nn.Unflatten(dim=-1, unflattened_size=[128, H // scale, W // scale]),
            # CONV NET
                nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
                final_act,
        )

    def forward(self, z):
        return self._generator(z)


# ========================================================================= #
# Discriminator                                                             #
# ========================================================================= #


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        # size of downscaled image
        C, H, W = img_shape
        scale = 2**4
        assert H % scale == 0
        assert W % scale == 0

        self._discriminator = nn.Sequential(
            # CONV LAYERS
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            # LINEAR LAYERS
            nn.Flatten(),
            nn.Linear(128 * (H // scale) * (W // scale), 1),
            # this used to have a sigmoid output layer, but replaced with `F.binary_cross_entropy_with_logits`
        )

    def forward(self, img, activate=False):
        out = self._discriminator(img)
        if activate:
            raise RuntimeError('sigmoid output has been replaced with `F.binary_cross_entropy_with_logits`')
        return out


# ========================================================================= #
# GAN                                                                       #
# ========================================================================= #


class DCGAN(LightningModule):

    def __init__(
        self,
        latent_dim: int = 128,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        obs_shape: Tuple[int, int, int] = (1, 32, 32),
        final_activation: str = 'sigmoid',  # originally tanh
    ):
        super().__init__()
        self.save_hyperparameters()
        # networks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=self.hparams.obs_shape, final_activation=self.hparams.final_activation)
        self.discriminator = Discriminator(img_shape=self.hparams.obs_shape)
        # checks
        assert self.dtype in (torch.float32, torch.float16)

    def forward(self, z):
        return self.generator(z)

    @torch.no_grad()
    def sample_z(self, batch_size: int):
        return torch.randn(batch_size, self.hparams.latent_dim, dtype=self.dtype, device=self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # sample noise
        z = self.sample_z(batch_size=batch.shape[0])

        # improve the generator to fool the discriminator TODO: I don't think the discriminator should be updated here?
        if optimizer_idx == 0:
            loss_gen = self.adversarial_loss(self.discriminator(self.generator(z)), is_real=True)
            self.log('loss_gen', loss_gen, prog_bar=True)
            return loss_gen

        # improve the discriminator to correctly identify the generator TODO: I don't think the generator should be updated here?
        elif optimizer_idx == 1:
            loss_real = self.adversarial_loss(self.discriminator(batch), is_real=True)
            loss_fake = self.adversarial_loss(self.discriminator(self.generator(z).detach()), is_real=False)
            loss_dsc = 0.5 * loss_real + 0.5 * loss_fake
            self.log('loss_real', loss_real, prog_bar=False)
            self.log('loss_fake', loss_fake, prog_bar=False)
            self.log('loss_dsc', loss_dsc, prog_bar=True)
            return loss_dsc

    def adversarial_loss(self, y_logits, is_real: bool):
        # generate targets
        gen_fn = (torch.ones if is_real else torch.zeros)
        y_targ = gen_fn(len(y_logits), 1, device=self.device, dtype=y_logits.dtype)
        # compute loss
        return F.binary_cross_entropy_with_logits(y_logits, y_targ)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []

    @torch.no_grad()
    def on_epoch_end(self):
        # log sampled images
        grid = torchvision.utils.make_grid(self.generator(self.validation_z))
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(data_path: str = None, resume_path: str = None, wandb: bool = False):
        from examples.common import make_mtg_datamodule
        from examples.common import make_mtg_trainer

        # settings:
        # 5878MiB / 5932MiB (RTX2060)
        system = DCGAN(obs_shape=(3, 224, 160))
        vis_input = system.sample_z(8)

        # start training model
        datamodule = make_mtg_datamodule(batch_size=32, load_path=data_path)
        trainer = make_mtg_trainer(train_epochs=500, visualize_period=500, resume_from_checkpoint=resume_path, visualize_input={'samples': vis_input}, wandb=wandb, wandb_project='MTG-GAN', wandb_name='MTG-GAN')
        trainer.fit(system, datamodule)

    # ENTRYPOINT
    logging.basicConfig(level=logging.INFO)
    main(
        data_path='data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
        resume_path=None,
        wandb=True,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
