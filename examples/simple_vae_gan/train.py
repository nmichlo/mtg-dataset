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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.common import BaseLightningModule
from examples.simple_vae.nn.loss import LaplaceMseLoss
from examples.simple_vae.nn.loss import MseLoss


logger = logging.getLogger(__name__)


# ========================================================================= #
# global layers                                                             #
# ========================================================================= #


def activation():
    # return Swish()
    return nn.LeakyReLU(1e-3, inplace=True)


def norm(feat, bn=True):
    if bn:
        # (1-momentum) ** steps = remaining_ratio
        # 1 - remaining_ratio ** (1/steps)
        # 1 - 0.01 ** (1/100) ~= 0.045
        # 1 - 0.01 ** (1/250) ~= 0,01825
        return nn.BatchNorm2d(feat, momentum=0.045)
    else:
        return nn.Identity()


def dropout():
    # return nn.Dropout(0.05)
    return nn.Identity()


def dropout2d():
    # return nn.Dropout2d(0.05)
    return nn.Identity()


# ========================================================================= #
# Generator                                                                 #
# ========================================================================= #


class GeneratorBody(nn.Module):
    def __init__(self, img_shape, in_features=(256, 256, 192, 128, 96), pix_in_features=64, final_activation='none'):
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
                    # norm(out_feat, bn=bn),
                    activation(),
            )

        # add pix_in_features to in_features list
        in_features = [*in_features, pix_in_features]
        self.in_size = in_features[0] * (H // scale) * (W // scale)

        self._generator = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=[in_features[0], H // scale, W // scale]),
            # CONV NET
            norm(in_features[0]),
            *(upsample_block(inp, out, bn=i < len(in_features) - 2) for i, (inp, out) in enumerate(zip(in_features[:-1], in_features[1:]))),
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
                    norm(out_feat, bn=bn),
                    activation(),
                    dropout2d(),
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
                activation(),
                dropout(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, z):
        assert z.ndim == 2, f'erroneous z.shape: {z.shape}'
        return self._discriminator_head(z)


class SharedGanAutoEncoder(nn.Module):

    # AE: BaseAutoEncoder

    def encode(self, x):
        return self._encoder_head(self._encoder_body(x))

    def decode(self, z):
        return self._generator_body(self._generator_head(z))

    def forward(self, x):
        return self.decode(self.encode(x))

    def forward_stochastic(self, x):
        z = self.encode(x)
        z += torch.randn_like(z)
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

    ):
        super().__init__()
        assert params_enc_in_gen_step is False
        assert len(obs_shape) == 3
        self._params_enc_in_gen_step = params_enc_in_gen_step
        # models
        self._generator_body     = GeneratorBody(img_shape=obs_shape, final_activation='none', in_features=gen_features, pix_in_features=gen_features_pix)
        self._generator_head     = DiscriminatorHead(in_size=z_size, hidden_size=hidden_size, out_size=self._generator_body.in_size)
        self._discriminator_body = DiscriminatorBody(img_shape=obs_shape, out_features=dsc_features)
        self._encoder_body       = DiscriminatorBody(img_shape=obs_shape, out_features=dsc_features)  # TODO: shared, if this is replaced with the discriminator_body
        self._encoder_head       = DiscriminatorHead(in_size=self._discriminator_body.out_size, hidden_size=hidden_size, out_size=z_size)
        self._discriminator_head = DiscriminatorHead(in_size=self._discriminator_body.out_size, hidden_size=hidden_size, out_size=1)
        # count params
        print(f'\n- params: generator_body     | trainable = {count_params(self._generator_body, True)} | non-trainable = {count_params(self._generator_body, False)}')
        print(f'- params: generator_head     | trainable = {count_params(self._generator_head, True)} | non-trainable = {count_params(self._generator_head, False)}')
        print(f'- params: encoder_body       | trainable = {count_params(self._encoder_body, True)} | non-trainable = {count_params(self._encoder_body, False)}')
        print(f'- params: encoder_head       | trainable = {count_params(self._encoder_head, True)} | non-trainable = {count_params(self._encoder_head, False)}')
        print(f'- params: discriminator_body | trainable = {count_params(self._discriminator_body, True)} | non-trainable = {count_params(self._discriminator_body, False)}')
        print(f'- params: discriminator_head | trainable = {count_params(self._discriminator_head, True)} | non-trainable = {count_params(self._discriminator_head, False)}\n')

    def discriminate(self, x):
        return self._discriminator_head(self._discriminator_body(x))

    def generate(self, z):
        return self.decode(z)


# ========================================================================= #
# GAN                                                                       #
# ========================================================================= #


def features(start, end, num):
    import numpy as np
    mul = (end / start) ** (1 / (num-1))
    sequence = start * mul ** np.arange(num)
    return tuple(int(v) for v in np.round(sequence))


def _count_params(model, trainable=None):
    if model is None:
        return 0
    if trainable is None:
        return sum(p.numel() for p in model.parameters())
    elif trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def count_params(model, trainable=None):
    p = _count_params(model, trainable)
    pow = 0 if (p == 0) else int(np.log(p) / np.log(1000))
    mul = 1000 ** pow
    symbol = {0: '', 1: 'K', 2: 'M', 3: 'B', 4: 'T', 5: 'P', 6: 'E'}[pow]
    return f'{p/mul:5.1f}{symbol}'


class SimpleVaeGan(BaseLightningModule):

    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (3, 224, 160),
        # optimizer
        lr: float = 0.0005,
        adam_betas: float = (0.5, 0.999),
        # features
        z_size: int = 256,
        hidden_size: int = 384,
        dsc_features: Tuple[int, ...] = features(16, 128, num=5),
        gen_features: Tuple[int, ...] = features(128, 48, num=5),
        gen_features_pix: int = 32,
        # auto-encoder loss
        train_ae: bool = True,
        ae_recon_loss='mse',
        ae_beta=0.003,
        # loss components
        adversarial_sample: bool = True,
        adversarial_recons: bool = False,
        adversarial_recons_train_enc: bool = False,  # only applies when adversarial_recons=True, adversarial_recons=True
    ):
        super().__init__()
        self.save_hyperparameters()
        # checks
        assert adversarial_recons_train_enc is False
        assert len(self.hparams.obs_shape) == 3
        # combined network
        self.sgan = SharedGanAutoEncoder(
            z_size=self.hparams.z_size,
            hidden_size=self.hparams.hidden_size,
            obs_shape=self.hparams.obs_shape,
            dsc_features=self.hparams.dsc_features,
            gen_features=self.hparams.gen_features,
            gen_features_pix=self.hparams.gen_features_pix,
            params_enc_in_gen_step=self.hparams.adversarial_recons_train_enc and self.hparams.adversarial_recons,
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
            recon, ae_loss, ae_loss_rec, ae_loss_reg = self._train_stochastic_ae_forward(batch, True)
            self.log('ae_loss_rec',     ae_loss_rec,     prog_bar=False)
            self.log('ae_loss_reg',     ae_loss_reg,     prog_bar=False)
            self.log('ae_loss',         ae_loss,         prog_bar=False)
            return ae_loss

        # improve the generator to fool the discriminator
        elif optimizer_idx == 2:
            # compute gan loss over generator & auto-encoder
            loss_gen_sample = 0 if not self.hparams.adversarial_sample else self.adversarial_loss(self.sgan.discriminate(self.sgan.generate(self.sample_z(batch_size=B))), is_real=True)
            loss_gen_recons = 0 if not self.hparams.adversarial_recons else self.adversarial_loss(self.sgan.discriminate(self._train_stochastic_ae_forward(batch, False)), is_real=True)
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
            loss_fake_sample = 0 if not self.hparams.adversarial_sample else self.adversarial_loss(self.sgan.discriminate(self.sgan.generate(self.sample_z(batch_size=B))), is_real=False)
            loss_fake_recons = 0 if not self.hparams.adversarial_recons else self.adversarial_loss(self.sgan.discriminate(self._no_grad_stochastic_ae_forward(batch, False)), is_real=False)
            loss_dsc = 0.5 * (loss_real_batch + (loss_fake_sample + loss_fake_recons) / self._loss_count)
            # return loss
            self.log('loss_dsc_fake_sample', loss_fake_sample, prog_bar=False)
            self.log('loss_dsc_fake_recons', loss_fake_recons, prog_bar=False)
            self.log('loss_dsc_real_batch',  loss_real_batch,  prog_bar=False)
            self.log('loss_dsc',             loss_dsc,         prog_bar=False)
            return loss_dsc

    def _train_stochastic_ae_forward(self, batch, compute_loss: bool = True):
        # feed forward with noise
        z = self.sgan.encode(batch)
        # z += torch.randn_like(z)
        recon = self.sgan.decode(z)
        # compute recon loss & regularizer -- like KL for sigma = 1, MSE from zero
        if compute_loss:
            loss_rec = self._loss(recon, batch, reduction='mean')
            loss_reg = self.hparams.ae_beta * (z ** 2).mean()
            loss = loss_rec + loss_reg
            # return all values...
            return recon, loss, loss_rec, loss_reg
        return recon

    @torch.no_grad()
    def _no_grad_stochastic_ae_forward(self, batch, compute_loss: bool = True):
        return self._train_stochastic_ae_forward(batch, compute_loss=compute_loss)

    def adversarial_loss(self, y_logits, is_real: bool):
        # get targets
        gen_fn = (torch.ones if is_real else torch.zeros)
        y_targ = gen_fn(len(y_logits), 1, device=self.device, dtype=y_logits.dtype)
        # compute loss
        return F.binary_cross_entropy_with_logits(y_logits, y_targ, reduction='mean')

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(self.sgan.params_ae, lr=self.hparams.lr, betas=self.hparams.adam_betas)
        opt_dsc = torch.optim.Adam(self.sgan.params_dsc, lr=self.hparams.lr, betas=self.hparams.adam_betas)
        opt_gen = torch.optim.Adam(self.sgan.params_gen, lr=self.hparams.lr, betas=self.hparams.adam_betas)
        return [opt_ae, opt_dsc, opt_gen], []


# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main(data_path: str = None, resume_path: str = None, wandb: bool = True):
        from examples.common import make_mtg_datamodule
        from examples.common import make_mtg_trainer

        # settings:
        # 5878MiB / 5932MiB (RTX2060)
        system = SimpleVaeGan(
            # z_size=384,
            # hidden_size=768,
            # dsc_features = features(32, 256, num=5),
            # gen_features = features(256, 128, num=5),
            # gen_features_pix = 96,

            z_size=256,
            hidden_size=384,
            dsc_features=features(32, 160, num=5),
            gen_features=features(160, 48, num=5),
            gen_features_pix=32,
        )

        # get dataset & visualise images
        datamodule = make_mtg_datamodule(
            batch_size=96+16+4,
            load_path=data_path,
        )
        vis_input = system.sample_z(8)
        vis_imgs = torch.stack([datamodule.data[i] for i in [3466, 18757, 20000, 40000, 21586, 20541, 1100]])

        # start training model
        trainer = make_mtg_trainer(
            train_epochs=500,
            visualize_period=500, visualize_input={'samples': vis_input, 'recons': vis_imgs},
            wandb=wandb, wandb_project='MTG-GAN', wandb_name='MTG-GAN', wandb_kwargs=dict(tags=['medium']),
            checkpoint_monitor=None, resume_from_checkpoint=resume_path,
        )
        trainer.fit(system, datamodule)

    # ENTRYPOINT
    logging.basicConfig(level=logging.INFO)
    main(
        data_path='data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
        resume_path=None, #'/home/nmichlo/workspace/playground/mtg-dataset/checkpoints/2021-06-13_14:43:56/epoch=13-step=17499.ckpt',
        wandb=True,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
