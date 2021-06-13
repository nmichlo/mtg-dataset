import logging
import os
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
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


class BaseLightningModule(pl.LightningModule):

    def get_progress_bar_dict(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


# ========================================================================= #
# Visualise                                                                 #
# ========================================================================= #


@lru_cache()
def _fn_has_param(fn, param: str):
    import inspect
    return param in inspect.signature(fn).parameters


class VisualiseCallback(pl.Callback):
    """
    Takes in an input batch, if the ndim == 4, then it is assumed to be a batch of images (B, C, H, W).
    Feeds the input batch through the model every `period` steps, and obtains the output which now must
    be a batch of images (B, C, H, W).
    """

    def __init__(self, name: str, input_batch: torch.Tensor, every_n_steps=1000, log_local=True, log_wandb=False, is_hsv=False, figwidth=15):
        assert isinstance(input_batch, torch.Tensor)
        assert log_wandb or log_local
        assert isinstance(name, str) and name.strip()
        self._name = name
        self._count = 0
        self._every_n_steps = every_n_steps
        self._wandb = log_wandb
        self._local = log_local
        self._figwidth = figwidth
        self._is_hsv = is_hsv
        self._input_batch = input_batch
        self._input_batch_is_images = (input_batch.ndim == 4)

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch: torch.Tensor, batch_idx: int, dataloader_idx: int) -> None:
        self._count += 1
        if self._count % self._every_n_steps != 0:
            return
        # import everything
        import wandb
        from disent.visualize.visualize_util import make_image_grid
        # feed forward
        with torch.no_grad(), evaluate_context(pl_module) as eval_module:
            xs = self._input_batch.to(eval_module.device)
            rs = eval_module.forward(xs)
            # convert to uint8
            xs = torch.moveaxis(torch.clip(xs * 255, 0, 255).to(torch.uint8), 1, -1).detach().cpu().numpy()
            rs = torch.moveaxis(torch.clip(rs * 255, 0, 255).to(torch.uint8), 1, -1).detach().cpu().numpy()
            # make grid
            img = make_image_grid(np.concatenate([xs, rs]) if self._input_batch_is_images else rs, num_cols=len(xs), pad=4)
        # plot
        if self._wandb:
            wandb.log({self._name: wandb.Image(img)})
            logger.info('logged wandb model visualisation')

        if self._local:
            w, h = img.shape[:2]
            fig, ax = plt.subplots(figsize=(self._figwidth/w*h, self._figwidth))
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
# Wandb Setup & Finish Callback                                             #
# ========================================================================= #


class WandbContextManagerCallback(pl.Callback):

    def __init__(self, extra_entries: dict = None):
        self._extra_entries = {} if (extra_entries is None) else extra_entries

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        import wandb

        # get initial keys and values
        keys_values = {
            **pl_module.hparams,
        }
        # get batch size from datamodule
        if getattr(getattr(trainer, 'datamodule', None), 'batch_size', None):
            keys_values['batch_size'] = trainer.datamodule.batch_size
        # overwrite keys
        keys_values.update(self._extra_entries)

        print()
        for k, v in keys_values.items():
            setattr(wandb.config, k, v)
            print(f'{k}: {repr(v)}')
        print()

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        import wandb
        wandb.finish()


# ========================================================================= #
# makers                                                                    #
# ========================================================================= #


@contextmanager
def evaluate_context(module: torch.nn.Module, train: bool = False):
    """
    Temporarily switch a model to evaluation
    mode, and restore the mode afterwards!
    """
    was_training = module.training
    try:
        module.train(mode=train)
        yield module
    finally:
        module.train(mode=was_training)


# ========================================================================= #
# makers                                                                    #
# ========================================================================= #


def make_mtg_datamodule(
    batch_size: int = 32,
    num_workers: int = os.cpu_count(),
    val_ratio: float = 0,
    # convert options
    load_path: str = None,
    data_root: Optional[str] = None,
    convert_kwargs: Dict[str, Any] = None,
):
    from mtgdata.scryfall_convert import generate_converted_dataset

    # generate training set
    if load_path is None:
        if convert_kwargs is None:
            convert_kwargs = {}
        h5_path, meta_path = generate_converted_dataset(save_root=data_root, data_root=data_root, **convert_kwargs)
    else:
        assert not convert_kwargs, '`convert_kwargs` cannot be set if `data_path` is specified'
        assert not data_root, '`data_root` cannot be set if `data_path` is specified'
        h5_path = load_path

    return Hdf5DataModule(
        h5_path,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
    )


def make_mtg_trainer(
    # training
    train_epochs: int = None,
    train_steps: int = None,
    cuda: bool = torch.cuda.is_available(),
    # visualise
    visualize_period: int = 500,
    visualize_input: Dict[str, torch.Tensor] = None,
    # utils
    checkpoint_period: int = 2500,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_monitor: Optional[str] = 'loss',
    resume_from_checkpoint: str = None,
    # logging
    wandb=False,
    wandb_name: str = None,
    wandb_project: str = None,
    wandb_kwargs: dict = None,
):
    time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # initialise callbacks
    callbacks = []
    if wandb:
        callbacks.append(WandbContextManagerCallback())
    if visualize_period and (visualize_input is not None):
        for k, v in visualize_input.items():
            callbacks.append(VisualiseCallback(name=k, input_batch=v, every_n_steps=visualize_period, log_wandb=wandb, log_local=not wandb))

    if checkpoint_period:
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, time_str),
            monitor=checkpoint_monitor,
            every_n_train_steps=checkpoint_period,
            verbose=True,
            save_top_k=None if (checkpoint_monitor is None) else 5,
        ))

    # initialise logger
    logger = True
    if wandb:
        assert isinstance(wandb_name, str) and wandb_name, f'`wandb_name` must be a non-empty str, got: {repr(wandb_name)}'
        assert isinstance(wandb_project, str) and wandb_project, f'`wandb_project` must be a non-empty str, got: {repr(wandb_project)}'
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(name=f'{time_str}:{wandb_name}', project=wandb_project, **(wandb_kwargs if (wandb_kwargs is not None) else {}))

    # initialise model trainer
    return pl.Trainer(
        gpus=1 if cuda else 0,
        max_epochs=train_epochs,
        max_steps=train_steps,
        # checkpoint_callback=False,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        callbacks=callbacks
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

