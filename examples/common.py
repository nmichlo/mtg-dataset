import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from mtgdata.util import Hdf5Dataset


logger = logging.getLogger(__name__)


# ========================================================================= #
# Transform                                                                 #
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
        return img


# ========================================================================= #
# Visualise                                                                 #
# ========================================================================= #


class VisualiseCallback(pl.Callback):

    def __init__(self, every_n_steps=1000, use_wandb=False, is_hsv=False):
        self._count = 0
        self._every_n_steps = every_n_steps
        self._wandb = use_wandb
        self._is_hsv = is_hsv

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch: torch.Tensor, batch_idx: int, dataloader_idx: int) -> None:
        # counter
        self._count += 1
        # skip
        if self._count % self._every_n_steps != 0:
            return
        # import everything
        import wandb
        from disent.visualize.visualize_util import make_image_grid
        # feed forward
        with torch.no_grad():
            # generate images
            assert isinstance(trainer.datamodule, Hdf5DataModule), f'trainer.datamodule is not an instance of {Hdf5DataModule.__name__}, got: {type(trainer.datamodule)}'
            data = trainer.datamodule.data
            xs = torch.stack([data[i] for i in [3466, 18757, 20000, 40000, 21586, 20541, 1100]])
            rs = pl_module.forward(xs.to(pl_module.device), deterministic=True)
            # convert to uint8
            xs = torch.moveaxis(torch.clip(xs * 255, 0, 255).to(torch.uint8).detach().cpu(), 1, -1).numpy()
            rs = torch.moveaxis(torch.clip(rs * 255, 0, 255).to(torch.uint8).detach().cpu(), 1, -1).numpy()
            # make grid
            img = make_image_grid(np.concatenate([xs, rs]), num_cols=len(xs), pad=4)
        # plot
        if self._wandb:
            wandb.log({'mtg-recons': wandb.Image(img)})
            logger.info('logged wandb model visualisation')
        else:
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_axis_off()
            fig.tight_layout()
            plt.show()
            logger.info('shown matplotlib model visualisation')


# ========================================================================= #
# Hdf5 Data Module                                                          #
# ========================================================================= #


class Hdf5DataModule(pl.LightningDataModule):

    def __init__(self, h5_path: str, h5_dataset_name: str = 'data', batch_size: int = 64, val_ratio: float = 0.1, num_workers: int = os.cpu_count()):
        super().__init__()
        self._batch_size = batch_size
        self._val_ratio = val_ratio
        self._num_workers = num_workers
        # load h5py data
        self._data = Hdf5Dataset(
            h5_path=h5_path,
            h5_dataset_name=h5_dataset_name,
            transform=ToTensor(move_channels=True),
        )
        # self.dims is returned when you call dm.size()
        self.dims = self._data.shape[1:]

    @property
    def data(self):
        return self._data

    def setup(self, stage: Optional[str] = None):
        self._data_trn, self._data_val = random_split(
            dataset=self._data,
            lengths=[
                int(np.floor(len(self._data) * (1 - self._val_ratio))),
                int(np.ceil(len(self._data) * self._val_ratio)),
            ],
        )

    def train_dataloader(self):
        return DataLoader(dataset=self._data_trn, num_workers=self._num_workers, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self._data_val, num_workers=self._num_workers, batch_size=self._batch_size, shuffle=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

