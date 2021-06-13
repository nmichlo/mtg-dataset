"""
Just a quick hack I wanted to try, that trains a VAE like a DCGAN, but
shares the encoder with the discriminator (discriminator is the encoder
with a small FNC on top).

EDIT:
-- I just skimmed over the paper, but this seems similar to the
   ideas from "IntroVAE: Introspective Variational Autoencoders"
   https://arxiv.org/pdf/1807.06358.pdf
"""

import logging
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.simple_vae.nn.loss import LaplaceMseLoss
from examples.simple_vae.nn.loss import MseLoss


logger = logging.getLogger(__name__)


# ========================================================================= #
# Generator                                                                 #
# ========================================================================= #

class Generator(nn.Module):
    def __init__(self, z_size: int, img_shape, in_features=(256, 256, 192, 128, 96), pix_in_features=64, final_activation='none'):
        super().__init__()
        self.img_shape = img_shape

        # size of downscaled image
        C, H, W = img_shape
        scale = 2 ** len(in_features)
        assert H % scale == 0
        assert W % scale == 0

        final_act = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity(),
        }[final_activation]

        def upsample_block(in_feat, out_feat, bn=True):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
                    *([nn.BatchNorm2d(out_feat)] if bn else []),
                    nn.LeakyReLU(1e-3, inplace=True),
            )

        # add pix_in_features to in_features list
        in_features = [*in_features, pix_in_features]

        self._generator = nn.Sequential(
            # LINEAR NET
            nn.Linear(in_features=z_size, out_features=in_features[0] * (H // scale) * (W // scale)),
            nn.Unflatten(dim=-1, unflattened_size=[in_features[0], H // scale, W // scale]),
            # CONV NET
            nn.BatchNorm2d(in_features[0]),
            *(upsample_block(inp, out, bn=i < len(in_features) - 1) for i, (inp, out) in enumerate(zip(in_features[:-1], in_features[1:]))),
            # PIXELS
            nn.Conv2d(pix_in_features, C, kernel_size=3, stride=1, padding=1),
            final_act,
        )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._generator(z)


# ========================================================================= #
# Discriminator                                                             #
# ========================================================================= #


class DiscriminatorBody(nn.Module):
    def __init__(self, img_shape, out_features=(16, 32, 64, 128, 192)):
        super().__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(1e-3, inplace=True),
                # nn.Dropout2d(0.25),
                *([nn.BatchNorm2d(out_feat)] if bn else [])
            )

        # size of downscaled image
        C, H, W = img_shape
        scale = 2 ** len(out_features)
        assert H % scale == 0
        assert W % scale == 0

        self._discriminator_body = nn.Sequential(
            # CONV LAYERS
            discriminator_block(C, out_features[0], bn=False),
            *(discriminator_block(inp, out) for inp, out in zip(out_features[:-1], out_features[1:])),
            nn.Flatten(),
        )

        self.out_orig_shape = (out_features[-1], (H // scale), (W // scale))
        self.out_size = out_features[-1] * (H // scale) * (W // scale)

    def forward(self, x):
        assert x.ndim == 4, f'erroneous x.shape: {x.shape}'
        return self._discriminator_body(x)


class DiscriminatorHead(nn.Module):

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self._discriminator_head = nn.Sequential(
            nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(1e-3, inplace=True),
                nn.Dropout(0.25),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._discriminator_head(z)


class SharedGanAutoEncoder(nn.Module):

    # AE: BaseAutoEncoder

    def encode(self, x):
        return self._discriminator_head_ae(self._discriminator_body(x))

    def decode(self, z):
        return self._generator(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    # INIT:

    def __init__(self, z_size: int, hidden_size: int, obs_shape: Tuple[int, int, int] = (1, 32, 32)):
        super().__init__()
        assert len(obs_shape) == 3
        self._generator              = Generator(z_size=z_size, img_shape=obs_shape, final_activation='none')
        self._discriminator_body     = DiscriminatorBody(img_shape=obs_shape)
        self._discriminator_head_ae  = DiscriminatorHead(in_size=self._discriminator_body.out_size, hidden_size=hidden_size, out_size=z_size)
        self._discriminator_head_dsc = DiscriminatorHead(in_size=self._discriminator_body.out_size, hidden_size=hidden_size, out_size=1)

    # GAN:

    def discriminate(self, x):
        return self._discriminator_head_dsc(self._discriminator_body(x))

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
        z_size: int = 256,
        hidden_size: int = 512,
        obs_shape: Tuple[int, int, int] = (3, 224, 160),
        lr: float = 0.0002,
        adam_betas: float = (0.5, 0.999),
        ae_alpha=25.0,
        ae_beta=0.003,
        recon_loss='mse_laplace',
    ):
        super().__init__()
        self.save_hyperparameters()
        # checks
        assert len(self.hparams.obs_shape) == 3
        # combined network
        self.sgan = SharedGanAutoEncoder(z_size=self.hparams.z_size, hidden_size=self.hparams.hidden_size, obs_shape=self.hparams.obs_shape)
        # checks
        assert self.dtype in (torch.float32, torch.float16)
        # loss function
        if self.hparams.recon_loss == 'mse':
            self._loss = MseLoss()
        elif self.hparams.recon_loss == 'mse_laplace':
            self._loss = LaplaceMseLoss(freq_ratio=0.25)
        else:
            raise KeyError(f'invalid recon_loss: {self.hparams.recon_loss}')

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

        # improve the AE
        if optimizer_idx == 0:
            # compute ae loss
            recon, ae_loss, ae_loss_rec, ae_loss_reg = self._train_stochastic_ae_forward(batch)
            self.log('ae_loss_rec', ae_loss_rec, prog_bar=False)
            self.log('ae_loss_reg', ae_loss_reg, prog_bar=False)
            self.log('ae_loss',     ae_loss,     prog_bar=True)
            return ae_loss

        # improve the generator to fool the discriminator
        elif optimizer_idx == 1:
            # compute gan loss
            loss_gen_sample = self.adversarial_loss(self.sgan.generate_discriminate(z), is_real=True)
            # loss_gen_recons = self.adversarial_loss(self.sgan.discriminate(recon),      is_real=True)  # self.sgan.encode_generate_discriminate_train(batch)
            # loss_gen = 0.5 * (loss_gen_sample + loss_gen_recons)
            # self.log('loss_gen_recons', loss_gen_recons, prog_bar=False)
            loss_gen = loss_gen_sample
            self.log('loss_gen_sample', loss_gen_sample, prog_bar=False)
            self.log('loss_gen',        loss_gen,        prog_bar=True)
            return loss_gen

        # improve the discriminator to correctly identify the generator
        elif optimizer_idx == 2:
            # compute ae forward
            with torch.no_grad():
                recon = self.sgan.forward(batch)
            # discriminate random samples
            loss_fake_sample = self.adversarial_loss(self.sgan.generate_discriminate(z), is_real=False)
            loss_fake_recons = self.adversarial_loss(self.sgan.discriminate(recon),      is_real=False)  # self.sgan.encode_generate_discriminate_train(batch)
            loss_real_batch  = self.adversarial_loss(self.sgan.discriminate(batch),      is_real=True)
            loss_dsc = 0.25 * (loss_fake_sample + loss_fake_recons) + 0.5 * loss_real_batch
            self.log('loss_dsc_fake_sample', loss_fake_sample, prog_bar=False)
            self.log('loss_dsc_fake_recons', loss_fake_recons, prog_bar=False)
            self.log('loss_dsc_real_batch',  loss_real_batch,  prog_bar=False)
            self.log('loss_dsc',             loss_dsc,         prog_bar=True)
            return loss_dsc

    def _train_stochastic_ae_forward(self, batch):
        # feed forward with noise
        z = self.sgan.encode(batch)
        z + torch.randn_like(z)
        recon = self.sgan.decode(z)
        # compute recon loss & regularizer -- like KL for sigma = 1, MSE from zero
        loss_rec = self.hparams.ae_alpha * self._loss(recon, batch, reduction='mean')
        loss_reg = self.hparams.ae_beta * (z ** 2).mean()
        loss = loss_rec + loss_reg
        # return all values...
        return recon, loss, loss_rec, loss_reg

    def adversarial_loss(self, y_logits, is_real: bool):
        # get targets
        gen_fn = (torch.ones if is_real else torch.zeros)
        y_targ = gen_fn(len(y_logits), 1, device=self.device, dtype=y_logits.dtype)
        # compute loss
        return F.binary_cross_entropy_with_logits(y_logits, y_targ, reduction='mean')

    def configure_optimizers(self):
        params_ae  = nn.ModuleList([self.sgan._discriminator_body, self.sgan._discriminator_head_ae, self.sgan._generator]).parameters()
        params_gen = nn.ModuleList([self.sgan._generator]).parameters()
        params_dsc = nn.ModuleList([self.sgan._discriminator_body, self.sgan._discriminator_head_dsc]).parameters()
        # make optimizers
        opt_ae  = torch.optim.Adam(params_ae,  lr=self.hparams.lr, betas=self.hparams.adam_betas)
        opt_gen = torch.optim.Adam(params_gen, lr=self.hparams.lr, betas=self.hparams.adam_betas)
        opt_dsc = torch.optim.Adam(params_dsc, lr=self.hparams.lr, betas=self.hparams.adam_betas)
        return [opt_ae, opt_gen, opt_dsc], []


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(data_path: str = None, resume_path: str = None, wandb: bool = True):
        from examples.common import make_mtg_datamodule
        from examples.common import make_mtg_trainer

        # settings:
        # 5878MiB / 5932MiB (RTX2060)
        system = SharedGAN()
        vis_input = system.sample_z(8)

        # start training model
        datamodule = make_mtg_datamodule(batch_size=24, load_path=data_path)
        trainer = make_mtg_trainer(
            train_epochs=500, visualize_period=500, resume_from_checkpoint=resume_path,
            visualize_input=vis_input, visualize_input_is_images=False,
            wandb=wandb, wandb_project='MTG-GAN', wandb_name='MTG-GAN',
            checkpoint_monitor='ae_loss_rec',
        )
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
