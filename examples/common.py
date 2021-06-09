import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt


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


class VisualiseCallback(pl.Callback):

    def __init__(self, every_n_steps=1000, use_wandb=False, is_hsv=False):
        self._count = 0
        self._every_n_steps = every_n_steps
        self._wandb = use_wandb
        self._is_hsv = is_hsv

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch: torch.Tensor, batch_idx: int, dataloader_idx: int) -> None:
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
            # convert to uint8
            xs = torch.moveaxis(torch.clip(xs * 255, 0, 255).to(torch.uint8).detach().cpu(), 1, -1).numpy()
            rs = torch.moveaxis(torch.clip(rs * 255, 0, 255).to(torch.uint8).detach().cpu(), 1, -1).numpy()
            # make grid
            img = make_image_grid(np.concatenate([xs, rs]), num_cols=len(xs), pad=4)
        # plot
        if self._wandb:
            import wandb
            wandb.log({'mtg-recons': wandb.Image(img)})
        else:
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_axis_off()
            fig.tight_layout()
            plt.show()



import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms


class MtgDataModule(pl.LightningDataModule):

    def __init__(self, img_type: str, bulk_type: str):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):


        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)