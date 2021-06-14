"""
Just a quick hack I wanted to try, that trains a VAE like a DCGAN, but
shares the encoder with the discriminator (discriminator is the encoder
with a small FNC on top).

EDIT:
-- I just skimmed over the paper, but this seems similar to the
   ideas from "IntroVAE: Introspective Variational Autoencoders"
   https://arxiv.org/pdf/1807.06358.pdf

TODO: look at these
     - https://arxiv.org/pdf/2004.04467.pdf
     - https://arxiv.org/pdf/1912.10321.pdf
     - https://arxiv.org/pdf/2012.13736.pdf
     - https://arxiv.org/abs/2012.13375
     - https://arxiv.org/pdf/2012.11879.pdf
"""

import logging
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from examples.common import BaseLightningModule
from examples.common import count_params
from examples.common import make_features
from examples.nn.loss import LaplaceMseLoss
from examples.nn.loss import MseLoss
from examples.util.iter import is_last_iter
from examples.util.iter import iter_pairs


logger = logging.getLogger(__name__)

# ========================================================================= #
# global layers                                                             #
# ========================================================================= #


def activation():
    # return Swish()
    return nn.LeakyReLU(inplace=True)


def norm_dsc(feat, bn=True):
    if bn:
        return nn.BatchNorm2d(feat, momentum=0.05)
    else:
        return nn.Identity()


def norm_gen(feat, bn=True):
    if bn:
        return nn.BatchNorm2d(feat, momentum=0.05)
    else:
        return nn.Identity()


def weight_norm_dsc(module):
    return module
    # return torch.nn.utils.weight_norm(module)


def weight_norm_gen(module):
    return module
    # return torch.nn.utils.weight_norm(module)


def dropout():
    return nn.Dropout(0.25)


def dropout2d():
    # return nn.Dropout2d(0.05)
    return nn.Identity()


# ========================================================================= #
# Generator                                                                 #
# ========================================================================= #


class GeneratorBody(nn.Module):
    def __init__(self, img_shape, features=(256, 256, 192, 128, 96), pix_in_features=64):
        super().__init__()
        self.img_shape = img_shape

        def upsample_block(in_feat, out_feat, bn=True):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                weight_norm_gen(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)),
                    norm_gen(out_feat, bn=bn),
                    activation(),
            )

        # size of downscaled image
        C, H, W = img_shape
        scale = 2 ** len(features)
        assert H % scale == 0
        assert W % scale == 0

        # add pix_in_features to features list
        features = [*features, pix_in_features]
        self.in_size = features[0] * (H // scale) * (W // scale)

        self._generator = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=[features[0], H // scale, W // scale]),
            norm_gen(features[0]),
            # CONV NET
            *(upsample_block(inp, out, bn=not is_last) for is_last, (inp, out) in is_last_iter(iter_pairs(features))),
            # PIXELS
            nn.Conv2d(pix_in_features, C, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._generator(z)


# ========================================================================= #
# Discriminator                                                             #
# ========================================================================= #


class DiscriminatorBody(nn.Module):
    def __init__(self, img_shape, features=(16, 32, 64, 128, 192)):
        super().__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            return nn.Sequential(
                weight_norm_dsc(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)),
                    norm_dsc(out_feat, bn=bn),
                    activation(),
                    dropout2d(),
            )

        # size of downscaled image
        C, H, W = img_shape
        scale = 2 ** len(features)
        assert H % scale == 0
        assert W % scale == 0

        self.out_orig_shape = (features[-1], (H // scale), (W // scale))
        self.out_size = features[-1] * (H // scale) * (W // scale)

        self._discriminator_body = nn.Sequential(
            # CONV LAYERS
            discriminator_block(C, features[0], bn=False),
            *(discriminator_block(inp, out) for inp, out in iter_pairs(features)),
            nn.Flatten(),
        )

    def forward(self, x):
        assert x.ndim == 4, f'erroneous x.shape: {x.shape}'
        return self._discriminator_body(x)


class DiscriminatorHead(nn.Module):

    def __init__(self, in_size: int, hidden_size: Optional[int], out_size: int):
        super().__init__()
        if hidden_size is None:
            self._discriminator_head = nn.Linear(in_size, out_size)
        else:
            self._discriminator_head = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                    activation(),
                    dropout(),
                nn.Linear(hidden_size, out_size),
            )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._discriminator_head(z)


# ========================================================================= #
# Discriminator                                                             #
# ========================================================================= #


class SharedGanAutoEncoder(nn.Module):

    # AE: BaseAutoEncoder

    def encode(self, x):
        z = self._encoder_head(self._encoder_body(x))
        if self.is_vae:
            mu, log_var = z.chunk(2, dim=-1)
            z = Normal(loc=mu, scale=torch.exp(0.5 * log_var))
        return z

    def decode(self, z):
        return self._generator_body(self._generator_head(z))

    def forward(self, x):
        # deterministic
        z = self.encode(x)
        if self.is_vae:
            z = z.mean
        return self.decode(z)

    # PARAMETERS:

    @property
    def params_ae(self):
        return nn.ModuleList([self._encoder_body, self._encoder_head, self._generator_head, self._generator_body]).parameters()

    @property
    def params_gen(self):
        return self.params_ae if self._params_enc_in_gen_step else nn.ModuleList([self._generator_head, self._generator_body]).parameters()

    @property
    def params_dsc(self):
        return nn.ModuleList([self._discriminator_body, self._discriminator_head]).parameters()

    # INIT:

    def __init__(
        self, z_size:
        int, hidden_size:
        int, obs_shape: Tuple[int, int, int] = (1, 32, 32),
        # features
        dsc_features: Tuple[int, ...] = (16, 32, 64, 128, 192),
        gen_features: Tuple[int, ...] = (256, 256, 192, 128, 96),
        gen_features_pix: int = 64,
        params_enc_in_gen_step: bool = False,
        share_enc: bool = False,
        # vae
        is_vae: bool = False,
    ):
        super().__init__()
        assert params_enc_in_gen_step is False
        assert len(obs_shape) == 3
        self.is_vae = is_vae
        self._params_enc_in_gen_step = params_enc_in_gen_step
        # models
        self._generator_body     = GeneratorBody(img_shape=obs_shape, features=gen_features, pix_in_features=gen_features_pix)
        self._generator_head     = DiscriminatorHead(in_size=z_size, hidden_size=None, out_size=self._generator_body.in_size)
        self._discriminator_body = DiscriminatorBody(img_shape=obs_shape, features=dsc_features)
        self._encoder_body       = DiscriminatorBody(img_shape=obs_shape, features=dsc_features) if not share_enc else self._discriminator_body
        self._encoder_head       = DiscriminatorHead(in_size=self._discriminator_body.out_size, hidden_size=hidden_size, out_size=(z_size*2) if self.is_vae else z_size)
        self._discriminator_head = DiscriminatorHead(in_size=self._discriminator_body.out_size, hidden_size=hidden_size, out_size=1)
        # count params
        print(f'\n- params: generator_body     | trainable = {count_params(self._generator_body, True)} | non-trainable = {count_params(self._generator_body, False)}')
        print(f'- params: generator_head     | trainable = {count_params(self._generator_head, True)} | non-trainable = {count_params(self._generator_head, False)}')
        print(f'- params: encoder_body       | trainable = {count_params(self._encoder_body, True)} | non-trainable = {count_params(self._encoder_body, False)}')
        print(f'- params: encoder_head       | trainable = {count_params(self._encoder_head, True)} | non-trainable = {count_params(self._encoder_head, False)}')
        print(f'- params: discriminator_body | trainable = {count_params(self._discriminator_body, True)} | non-trainable = {count_params(self._discriminator_body, False)}')
        print(f'- params: discriminator_head | trainable = {count_params(self._discriminator_head, True)} | non-trainable = {count_params(self._discriminator_head, False)}\n')

    # GAN

    def discriminate(self, x):
        return self._discriminator_head(self._discriminator_body(x))

    def generate(self, z):
        return self.decode(z)


# ========================================================================= #
# GAN                                                                       #
# ========================================================================= #


class SimpleVaeGan(BaseLightningModule):

    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (3, 224, 160),
        # optimizer
        lr: float = 0.00025,
        adam_betas: float = (0.5, 0.999),
        # features
        z_size: int = 256,                                                 # 256,                      # 256,
        hidden_size: int = 384,                                            # 512,                      # 384,
        dsc_features: Tuple[int, ...] = make_features(16, 128, num=5),     # (16, 32, 64, 128, 192),   # features(16, 128, num=5),
        gen_features: Tuple[int, ...] = make_features(128, 48, num=5),     # (256, 256, 192, 128, 96), # features(128, 48, num=5),
        gen_features_pix: int = 32,                                        # 64,                       # 32,
        # auto-encoder loss
        share_enc: bool = False,
        train_ae: bool = True,
        ae_recon_loss: str ='mse',
        beta: float = 0.1,
        ae_is_vae: bool = True,
        # loss components
        adversarial_sample: bool = True,
        adversarial_recons: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        # checks
        assert len(self.hparams.obs_shape) == 3
        # combined network
        self.sgan = SharedGanAutoEncoder(
            is_vae=self.hparams.ae_is_vae,
            z_size=self.hparams.z_size,
            hidden_size=self.hparams.hidden_size,
            obs_shape=self.hparams.obs_shape,
            dsc_features=self.hparams.dsc_features,
            gen_features=self.hparams.gen_features,
            gen_features_pix=self.hparams.gen_features_pix,
            params_enc_in_gen_step=False,  # self.hparams.adversarial_recons_train_enc and self.hparams.adversarial_recons,
            share_enc=self.hparams.share_enc,
        )
        # checks
        assert self.dtype in (torch.float32, torch.float16)
        # loss function
        if self.hparams.ae_recon_loss == 'mse':
            self._loss = MseLoss()
        elif self.hparams.ae_recon_loss == 'mse_laplace':
            self._loss = LaplaceMseLoss(freq_ratio=0.25)
        else:
            raise KeyError(f'invalid ae_recon_loss: {self.hparams.ae_recon_loss}')
        # number of loss components
        self._loss_count = (1 if self.hparams.adversarial_sample else 0) + (1 if self.hparams.adversarial_recons else 0)
        assert self._loss_count > 0, 'both `adversarial_sample` and `adversarial_recons` cannot be False'

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
        B, _, _, _ = batch.shape

        # improve auto-encoder
        if optimizer_idx == 0:
            # skip the step
            if not self.hparams.train_ae:
                return None
            # auto-encoder loss
            recon, ae_loss, ae_loss_rec, ae_loss_reg = self.ae_forward(batch, True)
            self.log('ae_loss_rec',     ae_loss_rec,     prog_bar=False)
            self.log('ae_loss_reg',     ae_loss_reg,     prog_bar=False)
            self.log('ae_loss',         ae_loss,         prog_bar=False)
            return ae_loss

        # improve the generator to fool the discriminator
        elif optimizer_idx >= 2:
            # compute gan loss over generator & auto-encoder
            loss_gen_sample = 0 if not self.hparams.adversarial_sample else self.discriminate_loss(self.sgan.generate(self.sample_z(batch_size=B)), is_real=True)
            loss_gen_recons = 0 if not self.hparams.adversarial_recons else self.discriminate_loss(self.ae_forward(batch, False), is_real=True)
            loss_gen = (loss_gen_sample + loss_gen_recons) / self._loss_count
            # return loss
            self.log('loss_gen_recons', loss_gen_recons, prog_bar=False)
            self.log('loss_gen_sample', loss_gen_sample, prog_bar=False)
            self.log('loss_gen',        loss_gen,        prog_bar=False)
            return loss_gen

        # improve the discriminator to correctly identify the generator
        elif optimizer_idx == 1:
            # compute dsc loss over generator & batch
            loss_real_batch  = self.adversarial_loss(self.sgan.discriminate(batch), is_real=True)
            loss_fake_sample = 0 if not self.hparams.adversarial_sample else self.discriminate_loss(self.sgan.generate(self.sample_z(batch_size=B)), is_real=False)
            loss_fake_recons = 0 if not self.hparams.adversarial_recons else self.discriminate_loss(self.ae_forward_no_grad(batch), is_real=False)
            loss_dsc = 0.5 * (loss_real_batch + (loss_fake_sample + loss_fake_recons) / self._loss_count)
            # return loss
            self.log('loss_dsc_fake_sample', loss_fake_sample, prog_bar=False)
            self.log('loss_dsc_fake_recons', loss_fake_recons, prog_bar=False)
            self.log('loss_dsc_real_batch',  loss_real_batch,  prog_bar=False)
            self.log('loss_dsc',             loss_dsc,         prog_bar=False)
            return loss_dsc

    # AE LOSS

    def ae_forward(self, x, compute_loss: bool = True):
        # feed forward with noise
        posterior = self.sgan.encode(x)
        recon = self.sgan.decode(posterior.rsample() if self.sgan.is_vae else posterior)
        # compute recon loss & regularizer -- like KL for sigma = 1, MSE from zero
        if compute_loss:
            loss_rec = self._loss(recon, x, reduction='mean')
            # get regularizer
            loss_reg = 0
            if self.sgan.is_vae:
                loss_reg = self.hparams.beta * torch.distributions.kl_divergence(posterior, Normal(torch.zeros_like(posterior.loc), torch.ones_like(posterior.scale))).mean()
            # compute final loss
            loss = loss_rec + loss_reg
            # return all values...
            return recon, loss, loss_rec, loss_reg
        return recon

    @torch.no_grad()
    def ae_forward_no_grad(self, x):
        return self.ae_forward(x, False)

    # GAN LOSS

    def discriminate_loss(self, x, is_real: bool):
        return self.adversarial_loss(self.sgan.discriminate(x), is_real=is_real)

    def adversarial_loss(self, y_logits, is_real: bool):
        # get targets
        gen_fn = (torch.ones if is_real else torch.zeros)
        y_targ = gen_fn(len(y_logits), 1, device=self.device, dtype=y_logits.dtype)
        # compute loss
        return F.binary_cross_entropy_with_logits(y_logits, y_targ, reduction='mean')

    # OPTIM

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(self.sgan.params_ae, lr=self.hparams.lr * 2, betas=self.hparams.adam_betas)
        opt_dsc = torch.optim.Adam(self.sgan.params_dsc, lr=self.hparams.lr, betas=self.hparams.adam_betas)
        opt_gen = torch.optim.Adam(self.sgan.params_gen, lr=self.hparams.lr / 2, betas=self.hparams.adam_betas)
        return [opt_ae, opt_dsc, opt_gen], []


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(data_path: str = None, resume_path: str = None, wandb: bool = True):
        from examples.common import make_mtg_datamodule
        from examples.common import make_mtg_trainer

        # get dataset & visualise images
        datamodule = make_mtg_datamodule(
            batch_size=32,
            load_path=data_path,
        )

        system = SimpleVaeGan(
            # MEDIUM | batch_size=64 gives ~4070MiB at ~2.47it/s
            # z_size=256,
            # hidden_size=384,
            # dsc_features=make_features(16, 128, num=5),
            # gen_features=make_features(128, 48, num=5),
            # gen_features_pix=32,

            # LARGE | batch_size=32 gives ~5530MiB at ~2.08it/s
            z_size=384,                              # 256
            hidden_size=512,                         # 512
            dsc_features=(64, 96, 128, 192, 224),    # (16, 32, 64, 128, 192)
            gen_features=(256, 224, 192, 128, 96),   # (256, 256, 192, 128, 96)
            gen_features_pix=64,                     # 64

            # GENERAL
            share_enc=True,
            ae_recon_loss='mse_laplace',
        )

        # start training model
        trainer = make_mtg_trainer(
            train_epochs=500,
            visualize_period=500,
            visualize_input={
                'samples': system.sample_z(8),
                'recons': torch.stack([datamodule.data[i] for i in [3466, 18757, 20000, 40000, 21586, 20541, 1100]]),
            },
            wandb=wandb,
            wandb_project='MTG-GAN',
            wandb_name='MTG-GAN',
            wandb_kwargs=dict(tags=['large']),
            checkpoint_monitor='ae_loss_rec',
            resume_from_checkpoint=resume_path,
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
