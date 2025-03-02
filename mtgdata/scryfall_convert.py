#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
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

import os
import warnings
import logging
from pathlib import Path
from typing import Optional, TypeVar
from typing import Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from mtgdata.scryfall import (
    ScryfallBulkType,
    ScryfallImageType,
    ScryfallDataset,
    ScryfallCardFace,
)
from mtgdata.util import Hdf5Dataset


logger = logging.getLogger(__name__)


# ========================================================================= #
# Transform                                                                 #
# ========================================================================= #


class PilResizeNumpyTransform(object):
    def __init__(
        self,
        resize: Tuple[int, int] = None,
        assert_shape: Tuple[int, int, int] | None = None,
        assert_dtype: np.dtype | None = None,
        transpose: bool = True,
        pad_to_square: bool = False,
    ):
        self._resize = resize  # W, H
        self._assert_shape = assert_shape  # H, W, C
        self._assert_dtype = assert_dtype
        self._transpose = transpose
        self._pad_to_square = pad_to_square

    def __call__(self, img: "Image.Image"):
        # resize
        if self._resize is not None:
            if img.size != self._resize:
                img = img.resize(self._resize)
        # convert to numpy
        img = np.asarray(img)
        # check RGB
        assert img.ndim == 3, f"expected 3 dimensions, got: {img.ndim}"
        assert img.shape[-1] == 3, f"expected 3 channels, got: {img.shape[-1]}"
        # check shapes
        if self._assert_shape is not None:
            assert img.shape == self._assert_shape, (
                f"expected shape: {self._assert_shape}, got: {img.shape}"
            )
        if self._assert_dtype is not None:
            assert img.dtype == self._assert_dtype, (
                f"expected dtype: {self._assert_dtype}, got: {img.dtype}"
            )
        # pad to a square
        if self._pad_to_square:
            H, W, C = img.shape
            pad_h = (max(H, W) - H) / 2
            pad_w = (max(H, W) - W) / 2
            img = np.pad(
                img,
                [
                    (int(np.floor(pad_h)), int(np.ceil(pad_h))),
                    (int(np.floor(pad_w)), int(np.ceil(pad_w))),
                    (0, 0),
                ],
            )
            assert img.shape[0] == img.shape[1] == max(H, W)
        # transpose
        if self._transpose:
            img = np.moveaxis(img, -1, -3)
        # return values
        return img


# ========================================================================= #
# RESAVE                                                                    #
# ========================================================================= #


T = TypeVar("T")


def dataset_save_as_hdf5(
    dataset: ScryfallDataset,
    *,
    obs_shape: Tuple[int, int, int],  # (H, W, C) usually
    save_path: Path | str,
    batch_size: int = 64,
    num_workers: int = os.cpu_count(),
    overwrite: bool = False,
    compression_lvl: int = 4,
):
    """
    Re-save the given Scryfall dataset as an HDF5 file.
    - the hdf5 file will have the key `data`
    """
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError(
            "torch is not installed. Please install it via `pip install torch`"
        )

    save_path = Path(save_path)
    # defaults & checks
    if not save_path.name.endswith(".h5"):
        raise ValueError('save_path must end with the ".h5" extension')
    # skip if exists
    if not overwrite:
        if os.path.exists(save_path):
            logger.info(
                f"dataset already exists and overwriting is not enabled, skipping: {repr(save_path)}"
            )
            return
    # open file
    with h5py.File(save_path, "w", libver="earliest") as f:
        # create new dataset
        d = f.create_dataset(
            name="data",
            shape=(len(dataset), *obs_shape),
            dtype="uint8",
            chunks=(1, *obs_shape),
            compression="gzip",
            compression_opts=compression_lvl,
            # non-deterministic time stamps are added to the file if this is not
            # disabled, resulting in different hash sums when the file is re-generated!
            # https://github.com/h5py/h5py/issues/225
            track_times=False,
        )
        # dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        # save data -- TODO: this is not atomic! if conversion fails the file is leftover!
        logger.info(f"Starting dataset conversion: {repr(save_path)}")
        with tqdm(desc="Converting:", total=len(dataset), dynamic_ncols=True) as p:
            for i, batch in enumerate(loader):
                d[i * batch_size : (i + 1) * batch_size] = batch
                p.update(len(batch))
        logger.info(f"Finished dataset conversion: {repr(save_path)}")


# LEGACY, TODO: rather just use duckdb over the JSON bulk json file, and export reduced data
# def dataset_save_meta(
#     data: ScryfallDataset,
#     save_path: str,
#     data_root: Optional[str] = None,
#     overwrite: bool = False,
# ):
#     if not save_path.endswith('.json'):
#         raise ValueError('save_path must end with the ".json" extension')
#     # check exists
#     if not overwrite:
#         if os.path.exists(save_path):
#             logger.warning(f'cards list already exists, overwriting not enabled, skipping: {repr(save_path)}')
#             return
#     # collect card information
#     card_info = tqdm(ScryfallAPI.card_face_info_iter(img_type=data.img_type, bulk_type=data.bulk_type, data_root=data_root), desc='Loading Cards List')
#     card_info = [{'idx': i, **info.__dict__} for i, info in enumerate(sorted(card_info, key=lambda x: x.img_file))]
#     # check that card_info corresponds to the dataset
#     # and that everything is sorted correctly!
#     assert len(card_info) == len(data)
#     assert [info['img_file'] for info in card_info] == [os.path.join(os.path.basename(os.path.dirname(sample[0])), os.path.basename(sample[0])) for sample in data.samples]
#     # save!
#     with open(save_path, 'w') as f:
#         json.dump(card_info, f, sort_keys=False)
#     logger.info(f'saved dataset meta: {repr(save_path)}')


# ========================================================================= #
# RESAVE                                                                    #
# ========================================================================= #


def _speed_test(desc, dat):
    # speed test helper function
    import time

    with tqdm(desc=desc) as p:
        t = time.time()
        while time.time() - t < 5:
            _ = dat[np.random.randint(0, len(dat))]
            p.update(n=1)


# sane modes that won't use too much disk space
SANE_MODES = {
    ("all_cards", "small"),  # ~243280 cards ~3GB
    ("default_cards", "small"),  # ~60459 cards ~0.75GB
    ("default_cards", "normal"),  # ~60459 cards
    ("default_cards", "border_crop"),  # ~60459 cards
}

# ORIGINAL ASPECT RATIOS:
# 'small':       (204,  146, 3),    # ASPECT: 1,397260274  | PRIME FACTORS: 2*2*3*17, 2*73         | GCF: 2
# 'border_crop': (680,  480, 3),    # ASPECT: 1,4166666667 | PRIME FACTORS: 2**3*5*17, 2**5*3*5    | GCF: 2**4
# 'normal':      (680,  488, 3),    # ASPECT: 1,393442623  | PRIME FACTORS: 2**3*5*17, 2**3*61     | GCF: 2**3
# 'large':       (936,  672, 3),    # ASPECT: 1,3928571429 | PRIME FACTORS: 2**3*3**2*13, 2**5*3*7 | GCF: 2**3*3
# 'png':         (1040, 745, 3),    # ASPECT: 1,3959731544 | PRIME FACTORS: 2**4*5*13, 5*149       | GCF: 5

# RESIZED ASPECT RATIOS
# (192, 128)  # ASPECT: 1.5 | PRIME FACTORS: 3*2**6, 2*2**6 | GCF: 2**6 | AREA:  24576
# (112, 80)   # ASPECT: 1.4 | PRIME FACTORS: 7*2**4, 5*2**4 | GCF: 2**4 | AREA:   8960
# (224, 160)  # ASPECT: 1.4 | PRIME FACTORS: 7*2**5, 5*2**5 | GCF: 2**5 | AREA:  35840
# (448, 320)  # ASPECT: 1.4 | PRIME FACTORS: 7*2**6, 5*2**6 | GCF: 2**6 | AREA: 143360

# SPEED TESTS:
# default_small  - inp: 599.10it/s  ~0.9GB
# default_small  - out: 1528.33it/s ~5.7GB
# default_normal - inp: 81.19it/s   ~7.1GB
# default_normal - out: 1769.92it/s ~5.2GB


class CardTransform:
    def __init__(self, transform: PilResizeNumpyTransform):
        self.transform = transform

    def __call__(self, card: ScryfallCardFace) -> np.ndarray:
        img = card.dl_and_open_im_resized()
        return self.transform(img)


def generate_converted_dataset(
    # output settings
    out_img_type: ScryfallImageType = ScryfallImageType.border_crop,
    out_bulk_type: ScryfallBulkType = ScryfallBulkType.default_cards,
    out_obs_compression_lvl: int = 4,
    out_obs_size_wh: Optional[
        Tuple[int | None, int | None]
    ] = None,  # (W, H) -- None is auto computed based on default size.
    out_obs_channels_first: bool = False,
    out_obs_pad_to_square: bool = False,
    # save options
    save_root: Optional[str | Path] = None,
    save_overwrite: bool = True,
    # image download settings
    imgs_force_update: bool = False,
    imgs_download_threads: int = os.cpu_count() * 2,
    # conversion settings
    convert_batch_size: int = 128,
    convert_num_workers: int = os.cpu_count() // 2,
    convert_speed_test: bool = False,
) -> Tuple[Path, Path]:
    if out_obs_size_wh is None:
        out_obs_size_wh = (None, None)

    # scale the size to within out_obs_size_wh, keeping the aspect ratio
    width, height = out_img_type.get_scaled_size(
        width=out_obs_size_wh[0],
        height=out_obs_size_wh[1],
    )

    # create the transform
    transform = CardTransform(
        transform=PilResizeNumpyTransform(
            resize=(width, height),
            assert_dtype=np.uint8,
            assert_shape=(height, width, 3),
            transpose=out_obs_channels_first,
            pad_to_square=out_obs_pad_to_square,
        )
    )

    # download the dataset
    dataset = ScryfallDataset(
        img_type=out_img_type,
        bulk_type=out_bulk_type,
        force_update=imgs_force_update,
        download_mode="now",
        transform=transform,
    )

    if height / width != 1.4:
        warnings.warn(
            f"Aspect ratio of height/width is not 1.4, given: {height}x{width} which gives {height / width}"
        )
    if (out_bulk_type, out_img_type) not in SANE_MODES:
        warnings.warn(
            f"Current combination of bulk and image types might generate a lot of data: {(out_bulk_type, out_img_type)} consider instead one of: {sorted(SANE_MODES)}"
        )
    if (height > out_img_type.height) or (width > out_img_type.width):
        warnings.warn(
            f"images are being unscaled from input size of: {out_img_type.size} to: {(width, height)}"
        )

    # get the shape of the images in the dataset (without padding)
    data_shape = (
        (len(dataset), 3, height, width)
        if out_obs_channels_first
        else (len(dataset), height, width, 3)
    )
    data_shape_str = "x".join(str(d) for d in data_shape)

    # get the output observation shape (with padding)
    if not out_obs_pad_to_square:
        obs_shape = data_shape[1:]
    else:
        size = max(width, height)
        obs_shape = (3, size, size) if out_obs_channels_first else (size, size, 3)
        data_shape_str = f"{data_shape_str}sqr"

    # get paths & make sure parent folder exists
    save_root = Path(save_root) if save_root else dataset.ds.ds_dir / "converted"
    save_root.mkdir(parents=True, exist_ok=True)
    bt, it = out_bulk_type.replace("_", "-"), out_img_type.replace("_", "-")
    path_data = (
        save_root
        / f"mtg_{bt}-{dataset.ds.bulk_date}_{it}_{data_shape_str}_c{out_obs_compression_lvl}.h5"
    )
    path_meta = (
        save_root
        / f"mtg_{bt}-{dataset.ds.bulk_date}_{it}_{data_shape_str}_c{out_obs_compression_lvl}_meta.json"
    )

    # check paths
    do_save = True
    if not save_overwrite:
        if os.path.exists(path_data) or os.path.exists(path_meta):
            logger.warning(
                f"converted dataset or meta files already exist: {repr(path_data)} or {repr(path_meta)}"
            )
            do_save = False

    # convert the dataset
    if do_save:
        dataset_save_as_hdf5(
            dataset=dataset,
            save_path=path_data,
            overwrite=save_overwrite,
            obs_shape=obs_shape,
            compression_lvl=out_obs_compression_lvl,
            batch_size=convert_batch_size,
            num_workers=convert_num_workers,
        )
        # save dataset meta
        warnings.warn("saving dataset meta is not currently supported")
        # TODO: dataset_save_meta(dataset, save_path=path_meta, overwrite=save_overwrite)

    # test the datasets
    if convert_speed_test:
        _speed_test("raw images speed test", dataset)
        hdat = Hdf5Dataset(path_data, "data")
        _speed_test("converted hdf5 speed test", hdat)

    # done!
    return path_data, path_meta


# ========================================================================= #
# ENTRY POINT - HELPERS                                                     #
# ========================================================================= #


def _make_parser_scryfall_convert(parser=None):
    # make default parser
    if parser is None:
        import argparse

        parser = argparse.ArgumentParser()
    # add arguments from scryfall.py
    from mtgdata.scryfall import _make_parser_scryfall_prepare

    _make_parser_scryfall_prepare(parser)
    # extra args
    parser.add_argument(
        "-o", "--out-root", type=str, default=None, help="output folder"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        default="default",
        help="resized image shape: `<HEIGHT|?>x<WIDTH|?>` | `default` eg. --size=224x160, --size=512x? or --size==default",
    )
    parser.add_argument(
        "-c",
        "--channels-first",
        action="store_true",
        help="if specified, saves image channels first with the shape: (C, H, W) instead of: (H, W, C)",
    )
    parser.add_argument(
        "-p",
        "--pad-to-square",
        action="store_true",
        help="if specified, zero pad the resized observations (H, W) to squares (max(H, W), max(H, W))",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(os.cpu_count() // 2, 1),
        help="number of workers to use when processing the dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="number of images to load in every batch when processing the dataset",
    )
    parser.add_argument(
        "--skip-speed-test",
        action="store_true",
        help="if specified, disabled testing the before and after dataset speeds",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="string to add to the end of the file name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing generated dataset files",
    )
    parser.add_argument(
        "--compression-lvl",
        type=int,
        default=4,
        help="the compression level of the h5py file (0 to 9)",
    )
    # return the parser
    return parser


def _run_scryfall_convert(args):
    # update args
    if args.size == "default":
        obs_size_wh = None
    else:
        try:
            width, height = (
                None if (v == "?") else int(v) for v in args.size.split("x")
            )
            obs_size_wh = (width, height)
        except Exception:
            raise ValueError(
                f'invalid size argument: {repr(args.size)}, must be of format: "<width|?>x<height|?>" or "default", eg. "--size=160x224", "--size=?x224" or "--size=default"'
            )

    # convert dataset
    generate_converted_dataset(
        # output settings
        out_img_type=args.img_type,
        out_bulk_type=args.bulk_type,
        out_obs_compression_lvl=args.compression_lvl,
        out_obs_size_wh=obs_size_wh,
        out_obs_channels_first=args.channels_first,
        out_obs_pad_to_square=args.pad_to_square,
        # save options
        save_root=args.out_root,
        save_overwrite=args.overwrite,
        # image download settings
        # data_root=args.data_root,
        imgs_force_update=args.force_update,
        imgs_download_threads=args.download_threads,
        # imgs_clean_invalid=args.clean_invalid_images,
        # conversion settings
        convert_batch_size=args.batch_size,
        convert_num_workers=args.num_workers,
        convert_speed_test=not args.skip_speed_test,
    )


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


if __name__ == "__main__":
    # initialise logging
    logging.basicConfig(level=logging.INFO)
    # run application
    _run_scryfall_convert(_make_parser_scryfall_convert().parse_args())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
