"""
FROM: https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/dcgan.py
"""
import os
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from examples.common import ToTensor
from examples.common import VisualiseCallback
from mtgdata.util import H5pyDataset


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_shape[1] // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class DCGAN(LightningModule):

    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # networks
        img_shape = (1, 32, 32)
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.validation_z = torch.randn(8, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict(
                {
                    'loss': g_loss,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                }
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict(
                {
                    'loss': d_loss,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                }
            )
            return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


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
            repr_hidden_size=None,  # 1536,
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
            logger=WandbLogger(
                name=f'mtg-gan|{model.hparams.recon_loss}:{model.hparams.recon_weight_reduce}:{model.hparams.recon_weight_mode}',
                project='MTG'
            ) if use_wandb else False,
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
