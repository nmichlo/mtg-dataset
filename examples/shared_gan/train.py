"""
port and customised version of:
https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/dcgan.py
"""

import logging
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.simple_vae.nn.model import BaseAutoEncoder


logger = logging.getLogger(__name__)


# ========================================================================= #
# Generator                                                                 #
# ========================================================================= #


class Generator(nn.Module):
    def __init__(self, z_size: int, img_shape, final_activation='tanh'):
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
            nn.Linear(in_features=z_size, out_features=128 * (H // scale) * (W // scale)),
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
            nn.Conv2d(64, C, kernel_size=3, stride=1, padding=1),
                final_act,
        )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._generator(z)


# ========================================================================= #
# Discriminator                                                             #
# ========================================================================= #


class DiscriminatorBody(nn.Module):
    def __init__(self, z_size, img_shape):
        super().__init__()

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

        self._discriminator_body = nn.Sequential(
            # CONV LAYERS
            *discriminator_block(C, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            # LINEAR LAYERS
            nn.Flatten(),
            nn.Linear(128 * (H // scale) * (W // scale), z_size),
        )

    def forward(self, x):
        assert x.ndim == 4, f'erroneous x.shape: {x.shape}'
        return self._discriminator_body(x)


class DiscriminatorHead(nn.Module):

    def __init__(self, z_size: int):
        super().__init__()
        self._discriminator_head = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_size, 1)
        )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        # this should be used with F.binary_cross_entropy_with_logits(y_logits, y_targ)
        logit = self._discriminator_head(z)
        return logit


class SharedGanAutoEncoder(BaseAutoEncoder):

    def _enc(self, x):
        assert x.ndim == 4, f'erroneous x.shape: {x.shape}'
        return self._discriminator_body(x)

    def _dec(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._generator(z)

    def __init__(self, generator, discriminator_body, discriminator_head):
        super().__init__()
        self._generator = generator
        self._discriminator_body = discriminator_body
        self._discriminator_head = discriminator_head

    def discriminate(self, x):
        # we discard the variance values, we should probably include these!
        return self._discriminator_head(self.encode(x).mean)

    def generate(self, z):
        return self.decode(z)

    def generate_discriminate(self, z):
        return self.discriminate(self.generate(z))


# ========================================================================= #
# GAN                                                                       #
# ========================================================================= #


class SharedGAN(pl.LightningModule):

    def __init__(
        self,
        z_size: int = 128,
        obs_shape: Tuple[int, int, int] = (1, 32, 32),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        alpha=1.0,
        beta=0.003,
    ):
        super().__init__()
        self.save_hyperparameters()
        # checks
        assert len(self.hparams.obs_shape) == 3
        # combined network
        self.sgan = SharedGanAutoEncoder(
            generator=Generator(z_size=self.hparams.z_size, img_shape=self.hparams.obs_shape, final_activation='none'),
            discriminator_body=DiscriminatorBody(z_size=self.hparams.z_size*2, img_shape=self.hparams.obs_shape),
            discriminator_head=DiscriminatorHead(z_size=self.hparams.z_size),
        )
        # checks
        assert self.dtype in (torch.float32, torch.float16)

    @torch.no_grad()
    def sample_z(self, batch_size: int):
        return torch.randn(batch_size, self.hparams.z_size, dtype=self.dtype, device=self.device)

    def forward(self, x_or_z):
        assert x_or_z.ndim in (2, 4)
        # auto-encoder forward
        if x_or_z.ndim == 4:
            return self.sgan.forward(x_or_z)
        # generator forward
        elif x_or_z.ndim == 2:
            return self.sgan.generate(x_or_z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # sample noise
        z = self.sample_z(batch_size=batch.shape[0])

        # improve the generator to fool the discriminator
        if optimizer_idx == 0:
            # compute vae loss
            recon, vae_loss, vae_loss_recon, vae_loss_kl = self._train_vae_forward(batch)
            self.log('vae_loss_recon', vae_loss_recon, prog_bar=False)
            self.log('vae_loss_kl',    vae_loss_kl,    prog_bar=False)
            self.log('vae_loss',       vae_loss,       prog_bar=True)
            # compute gan loss
            loss_gen_sample = self.adversarial_loss(self.sgan.generate_discriminate(z), is_real=True)
            loss_gen_recons = self.adversarial_loss(self.sgan.discriminate(recon),      is_real=True)  # self.sgan.encode_generate_discriminate_train(batch)
            loss_gen = 0.5 * (loss_gen_sample + loss_gen_recons)
            self.log('loss_gen_sample', loss_gen_sample, prog_bar=False)
            self.log('loss_gen_recons', loss_gen_recons, prog_bar=False)
            self.log('loss_gen',        loss_gen,        prog_bar=True)
            return loss_gen

        # improve the discriminator to correctly identify the generator
        elif optimizer_idx == 1:
            # compute vae forward
            with torch.no_grad():
                recon, _, _ = self.sgan.forward_train(batch)
            # discriminate random samples
            loss_fake_sample = self.adversarial_loss(self.sgan.generate_discriminate(z), is_real=False)
            loss_fake_recons = self.adversarial_loss(self.sgan.discriminate(recon),      is_real=False)  # self.sgan.encode_generate_discriminate_train(batch)
            loss_real_batch  = self.adversarial_loss(self.sgan.discriminate(batch),      is_real=True)
            loss_dsc = 0.25 * (loss_fake_sample + loss_fake_recons) + 0.5 * loss_real_batch
            self.log('loss_fake_sample', loss_fake_sample, prog_bar=False)
            self.log('loss_fake_recons', loss_fake_recons, prog_bar=False)
            self.log('loss_real_batch',  loss_real_batch,  prog_bar=False)
            self.log('loss_dsc',         loss_dsc,         prog_bar=True)
            return loss_dsc

    def _train_vae_forward(self, batch):
        recon, posterior, prior = self.sgan.forward_train(batch)
        # compute recon loss
        loss_recon = self.hparams.alpha * F.mse_loss(recon, batch, reduction='mean')
        # compute kl divergence
        loss_kl = self.hparams.beta * torch.distributions.kl_divergence(posterior, prior).mean()
        # combined loss
        loss = loss_recon + loss_kl
        # return losses
        return recon, loss, loss_recon, loss_kl

    def adversarial_loss(self, y_logits, is_real: bool):
        # get targets
        gen_fn = (torch.ones if is_real else torch.zeros)
        y_targ = gen_fn(len(y_logits), 1, device=self.device, dtype=y_logits.dtype)
        # compute loss
        return F.binary_cross_entropy_with_logits(y_logits, y_targ)

    def configure_optimizers(self):
        g_params = self.sgan._generator.parameters()
        d_params = nn.ModuleList([self.sgan._discriminator_head, self.sgan._discriminator_body]).parameters()
        # make optimizers
        opt_g = torch.optim.Adam(g_params, lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = torch.optim.Adam(d_params, lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(data_path: str = None, resume_path: str = None, wandb: bool = True):
        from examples.common import make_mtg_datamodule
        from examples.common import make_mtg_trainer

        # settings:
        # 5878MiB / 5932MiB (RTX2060)
        system = SharedGAN(obs_shape=(3, 224, 160))
        vis_input = system.sample_z(8)

        # start training model
        datamodule = make_mtg_datamodule(batch_size=32, load_path=data_path)
        trainer = make_mtg_trainer(train_epochs=500, visualize_period=500, resume_from_checkpoint=resume_path, visualize_input=vis_input, visualize_input_is_images=False, wandb=wandb, wandb_project='MTG-GAN', wandb_name='MTG-GAN')
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
