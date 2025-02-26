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

__all__ = [
    "ScryfallDataset",
    "ScryfallCardFace",
    "ScryfallImageType",
    "ScryfallBulkType",
]

import dataclasses
import json
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal, TYPE_CHECKING
from typing import Tuple

import duckdb
import pytz
import requests

from doorway import AtomicOpen, io_download
from doorway.x import ProxyDownloader

if TYPE_CHECKING:
    from PIL import Image


logger = logging.getLogger(__name__)


# ========================================================================= #
# Dataset - Vars                                                            #
# ========================================================================= #


CACHE_STALE_AFTER = timedelta(days=365)


# ========================================================================= #
# Scryfall API Helper                                                       #
# ========================================================================= #


class ScryfallBulkType(str, Enum):
    oracle_cards = 'oracle_cards'
    unique_artwork = 'unique_artwork'
    default_cards = 'default_cards'
    all_cards = 'all_cards'
    rulings = 'rulings'


class ScryfallImageType(str, Enum):
    small = 'small'
    border_crop = 'border_crop'
    normal = 'normal'
    large = 'large'
    png = 'png'
    art_crop = 'art_crop'

    @property
    def extension(self) -> Literal['jpg', 'png']:
        return _IMG_TYPE_EXTENSIONS[self]

    @property
    def hwc(self) -> Tuple[int, int, int] | None:
        h, w, c = _IMG_TYPE_SIZES_HWC[self]
        return h, w, c

    @property
    def size(self) -> Tuple[int, int] | None:
        h, w, c = _IMG_TYPE_SIZES_HWC[self]
        return w, h  # PIL uses (W, H)

    @property
    def height(self) -> int:
        h, w, c = _IMG_TYPE_SIZES_HWC[self]
        return h

    @property
    def width(self) -> int:
        h, w, c = _IMG_TYPE_SIZES_HWC[self]
        return w


_IMG_TYPE_EXTENSIONS: dict[ScryfallImageType, Literal['jpg', 'png']] = {
    ScryfallImageType.small:       'jpg',
    ScryfallImageType.border_crop: 'jpg',
    ScryfallImageType.normal:      'jpg',
    ScryfallImageType.large:       'jpg',
    ScryfallImageType.png:         'png',
    ScryfallImageType.art_crop:    'jpg',
}

_IMG_TYPE_SIZES_HWC: dict[ScryfallImageType, Tuple[int, int, int] | None] = {
    # aspect ratio ~= 7:5 (H = 1.4 W)
    ScryfallImageType.small:       (204,  146, 3),
    ScryfallImageType.border_crop: (680,  480, 3),
    ScryfallImageType.normal:      (680,  488, 3),
    ScryfallImageType.large:       (936,  672, 3),
    ScryfallImageType.png:         (1040, 745, 3),
    ScryfallImageType.art_crop:    None,
}

_IMG_SHAPES = None


# ========================================================================= #
# Scryfall Dataset                                                          #
# ========================================================================= #


@dataclasses.dataclass(frozen=True)
class ScryfallCardFace:
    # query
    id: str
    oracle_id: str
    name: str
    set_code: str
    set_name: str
    img_uri: str
    _img_type: ScryfallImageType
    _bulk_type: ScryfallBulkType
    # computed
    _sets_dir: Path
    _proxy: ProxyDownloader | None = None

    @property
    def img_type(self) -> ScryfallImageType:
        return ScryfallImageType(self._img_type)

    @property
    def bulk_type(self) -> ScryfallBulkType:
        return ScryfallBulkType(self._bulk_type)

    @property
    def img_path(self) -> Path:
        return self._sets_dir / f'{self.set_code}/{self.id}.{self.img_type.extension}'

    @property
    def url_path_pair(self) -> Tuple[str, str]:
        return self.img_uri, str(self.img_path)

    def download(self, *, verbose: bool = True) -> Path:
        if self._proxy is None:
            raise RuntimeError('proxy is not set!')
        self._proxy.download(self.img_uri, str(self.img_path), exists_mode='skip', verbose=verbose, make_dirs=True, attempts=128, timeout=8)
        return self.img_path

    def dl_and_open_im(self, *, verbose: bool = True) -> "Image.Image":
        try:
            from PIL import Image
        except ImportError:
            raise ImportError('PIL is not installed, please install it using: `pip install pillow`')

        return Image.open(self.download(verbose=verbose))

    def __repr__(self):
        return f'<ScryfallCardFace: "{self.name}" ({self.set_code}), {self.img_path}>'

    def dl_and_open_im_resized(
        self,
        channel_mode: Literal["rgb", "skip"] = "rgb",
        resize_mode: Literal["resize", "error", "skip"] = "resize",
        *,
        verbose: bool = True,
    ) -> "Image.Image":
        img = self.dl_and_open_im(verbose=verbose)
        if channel_mode == 'rgb':
            img = img.convert('RGB')
        if self.img_type.size != img.size:
            if resize_mode == 'resize':
                logger.warning(f'image shape mismatch: {img.size} != {self.img_type.size} {self}')
                img = img.resize(self.img_type.size)
            elif resize_mode == 'error':
                raise RuntimeError(f'image shape mismatch: {img.size} != {self.img_type.size} {self}')
            elif resize_mode == 'skip':
                pass
            else:
                raise KeyError(f'invalid mode: {resize_mode}')

        return img


@dataclasses.dataclass()
class _DatasetIndex:
    bulk_data: dict[str, Any]
    last_updated: datetime

    @property
    def time_since_last_updated(self) -> timedelta:
        return datetime.now(pytz.utc) - self.last_updated

    def is_stale(self) -> bool:
        return self.time_since_last_updated > CACHE_STALE_AFTER

    @classmethod
    def _query_bulk_data(cls, bulk_type: ScryfallBulkType) -> dict[str, Any]:
        response = requests.get(f'https://api.scryfall.com/bulk-data')
        response.raise_for_status()
        data = response.json()
        # find the correct bulk data
        for item in data['data']:
            if item['type'] == bulk_type:
                return item
        raise ValueError(
            f'bulk type {bulk_type} not found in bulk data list, valid types are: {[d["type"] for d in data["data"]]}'
        )

    @classmethod
    def from_query(cls, bulk_type: ScryfallBulkType) -> '_DatasetIndex':
        return cls(
            bulk_data=cls._query_bulk_data(bulk_type),
            last_updated=datetime.now(pytz.utc),
        )


class ScryfallDataset:
    """
    Scryfall Card Face Dataset.

    This class is used to download and cache card face images from the Scryfall API.
    1. Check the prior index file to see if the bulk data is stale.
        b. If needed, download the new bulk data file, to overwrite the old one.
        a. Then update the index file.
    2. Parse the bulk data file and iterate over the card faces.

    NOTE: we do NOT handle removing stale images. We assume that scryfall infrequently
          updates their images. Downstream systems should NOT glob the image directory
          rather they should use the iterator to get the latest images. This way
          invalid images will be skipped.
    """

    def __init__(
        self,
        img_type: ScryfallImageType = ScryfallImageType.normal,
        bulk_type: ScryfallBulkType = ScryfallBulkType.default_cards,
        *,
        ds_dir: Path | None = None,
    ):
        self._img_type = ScryfallImageType(img_type)
        self._bulk_type = ScryfallBulkType(bulk_type)
        # get root dir
        if ds_dir is None:
            data_root = os.environ.get('DATA_ROOT', 'data')
            ds_dir = Path(data_root) / 'scryfall' / bulk_type / img_type
        # check dirs
        ds_dir.mkdir(parents=True, exist_ok=True)
        if not ds_dir.is_dir():
            raise NotADirectoryError(f'path is not a directory: {ds_dir}')
        # initialise dataset
        self.__ds_dir = ds_dir
        self.__path_index = ds_dir / 'index.json'
        self.__path_bulk = ds_dir / 'bulk.json'

    @property
    def data_root(self) -> Path:
        return self.__ds_dir

    @property
    def img_type(self) -> ScryfallImageType:
        return self._img_type

    @property
    def bulk_type(self) -> ScryfallBulkType:
        return self._bulk_type

    @property
    def ds_dir(self) -> Path:
        return self.__ds_dir

    @property
    def _path_sets_dir(self) -> Path:
        return self.__ds_dir / 'sets'

    @property
    def bulk_datetime(self) -> datetime:
        index, _ = self._download_bulk_data()
        date = index.bulk_data['updated_at']
        return datetime.fromisoformat(date)

    @property
    def bulk_date(self) -> str:
        return self.bulk_datetime.strftime('%Y%m%d%H%M%S')  # same as in bulk filename

    # ~=~=~ prepare ~=~=~

    def prepare(
        self,
        *,
        force_update: bool = False,
        download_threads: int = 64,
    ) -> "ScryfallDataset":
        if force_update:
            self.invalidate_cache()
        self.download_all(
            proxy=None,
            threads=download_threads,
            verbose=True,
        )
        return self

    # ~=~=~ DB ~=~=~

    def __read_index(self) -> _DatasetIndex | None:
        if self.__path_index.exists():
            with AtomicOpen(self.__path_index, 'r') as fp:
                dat = json.load(fp)
                return _DatasetIndex(
                    bulk_data=dat['bulk_data'],
                    last_updated=datetime.fromisoformat(dat['last_updated']),
                )
        else:
            return None

    def __update_index(self, index: _DatasetIndex) -> None:
        with AtomicOpen(self.__path_index, 'w') as fp:
            dat = {
                'bulk_data': index.bulk_data,
                'last_updated': index.last_updated.isoformat(),
            }
            json.dump(dat, fp)

    # ~=~=~ bulk data ~=~=~

    def invalidate_cache(self) -> "ScryfallDataset":
        # call this to force a cache update
        self.__path_index.unlink(missing_ok=True)
        self.__path_bulk.unlink(missing_ok=True)
        return self

    # ~=~=~ bulk data ~=~=~

    def _download_bulk_data(self) -> Tuple[_DatasetIndex, Path]:
        index = self.__read_index()
        # stale / non-existent
        if index is None or index.is_stale():
            new_index = _DatasetIndex.from_query(self._bulk_type)
            # if bulk info is different, then download
            if (index is None) or (index.bulk_data != new_index.bulk_data):
                io_download(
                    src_url=new_index.bulk_data['download_uri'],
                    dst_path=str(self.__path_bulk),
                    overwrite_existing=True,
                )
            self.__update_index(new_index)
        # done
        assert index
        assert self.__path_bulk.exists()
        return index, self.__path_bulk

    # ~=~=~ yield card faces ~=~=~

    def __iter__(self) -> Iterator['ScryfallCardFace']:
        return self.yield_all()

    def yield_all(self, shared_proxy: ProxyDownloader | None = None) -> Iterator['ScryfallCardFace']:
        # fetch the bulk data
        _, path_bulk = self._download_bulk_data()
        img_type = self._img_type.value
        bulk_type = self._bulk_type.value
        query = f"""
            SELECT
                t.id,
                t.oracle_id,
                t.name,
                t.set AS set_code,
                t.set_name,
                img.image_uri,
                '{img_type}' AS img_type,
                '{bulk_type}' AS bulk_type
            FROM read_json('{path_bulk}') t
            CROSS JOIN LATERAL (
              VALUES
                (t.image_uris.{img_type}),
                (t.card_faces[1].image_uris.{img_type}),
                (t.card_faces[2].image_uris.{img_type})
            ) AS img(image_uri)
            WHERE img.image_uri IS NOT NULL;
        """
        cursor = duckdb.sql(query)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield ScryfallCardFace(*row, _sets_dir=self._path_sets_dir, _proxy=shared_proxy)

    def download_all(
        self,
        proxy: ProxyDownloader | None = None,
        threads: int = max(os.cpu_count() * 2, 8),
        verbose: bool = True,
    ) -> list['ScryfallCardFace']:
        cards_list = []

        def _itr():
            for item in self.yield_all(shared_proxy=proxy):
                yield item.url_path_pair
                cards_list.append(item)

        if not proxy:
            proxy = ProxyDownloader()

        failed = proxy.download_threaded(
            _itr(),
            exists_mode='skip',
            verbose=verbose,
            make_dirs=True,
            threads=threads,
            attempts=128,
            timeout=8,
        )
        assert not failed, f'failed to download: {failed}'
        return cards_list


# ========================================================================= #
# ENTRY POINT - HELPERS                                                     #
# ========================================================================= #


def _make_parser_scryfall_prepare(parser=None):
    # make default parser
    if parser is None:
        import argparse
        parser = argparse.ArgumentParser()
    # these should match scryfall_convert.py
    parser.add_argument('-b', '--bulk_type', type=str, default='default_cards',                     help="[default_cards|all_cards|oracle_cards|unique_artwork]. For more information, see: https://scryfall.com/docs/api/bulk-data")
    parser.add_argument('-i', '--img-type', type=str, default='normal',                             help="[png|border_crop|art_crop|large|normal|small]. For more information, see: https://scryfall.com/docs/api/images")
    parser.add_argument('-d', '--data-root', type=str, default=None,                                help="download and cache directory location")
    parser.add_argument('-f', '--force-update', action='store_true',                                help="overwrite existing files and ignore caches")
    parser.add_argument('-t', '--download_threads', type=int, default=max(os.cpu_count() * 2, 128), help="number of threads to use when downloading files")
    return parser


def _run_scryfall_prepare(args):
    if args.data_root is not None:
        os.environ['DATA_ROOT'] = args.data_root

    ds = ScryfallDataset(bulk_type=args.bulk_type, img_type=args.img_type)

    if args.force_update:
        logger.info('Forcing cache update...')
        ds.invalidate_cache()

    logger.info(f'Downloading images for: {repr(args.bulk_type)} {repr(args.img_type)}')
    all_cards = ds.download_all(threads=args.download_threads, verbose=True)
    logger.info(f'Finished downloading {len(all_cards)} images for: {repr(args.bulk_type)} {repr(args.img_type)}')


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


if __name__ == '__main__':
    # initialise logging
    logging.basicConfig(level=logging.INFO)
    # run application
    _run_scryfall_prepare(_make_parser_scryfall_prepare().parse_args())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
