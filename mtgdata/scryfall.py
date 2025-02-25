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

import dataclasses
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Literal
from typing import Tuple

import duckdb
import pydantic
import pytz
import requests
from PIL import Image

from doorway import AtomicOpen, io_download
from doorway.x import ProxyDownloader


logger = logging.getLogger(__name__)


# ========================================================================= #
# Dataset - Vars                                                            #
# ========================================================================= #


CACHE_STALE_AFTER = timedelta(days=365)


# ========================================================================= #
# Scryfall API Helper                                                       #
# ========================================================================= #


ImageType = Literal['small', 'border_crop', 'normal', 'large', 'png', 'art_crop']
BulkType = Literal['oracle_cards', 'unique_artwork', 'default_cards', 'all_cards', 'rulings']

_IMG_TYPES: dict[ImageType, str] = {
    'small':       'jpg',
    'border_crop': 'jpg',
    'normal':      'jpg',
    'large':       'jpg',
    'png':         'png',
    'art_crop':    'jpg',
}

_IMG_SHAPES: dict[ImageType, Tuple[int, int, int] | None] = {
    # aspect ratio ~= 7:5 (H = 1.4 W)
    'small':       (204,  146, 3),
    'border_crop': (680,  480, 3),
    'normal':      (680,  488, 3),
    'large':       (936,  672, 3),
    'png':         (1040, 745, 3),
    'art_crop':    None,
}


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
    img_type: ImageType
    bulk_type: BulkType
    # computed
    _sets_dir: Path
    _proxy: ProxyDownloader | None = None

    @property
    def img_path(self) -> Path:
        return self._sets_dir / f'{self.set_code}/{self.id}.{_IMG_TYPES[self.img_type]}'

    def download(self, verbose: bool = True) -> Path:
        if self._proxy is None:
            raise RuntimeError('proxy is not set!')
        self._proxy.download(self.img_uri, str(self.img_path), exists_mode='skip', verbose=verbose, make_dirs=True, attempts=128, timeout=8)
        return self.img_path

    def open_image(self, verbose: bool = True) -> Image.Image:
        return Image.open(self.download(verbose=verbose))

    @property
    def url_path_pair(self) -> Tuple[str, str]:
        return self.img_uri, str(self.img_path)


class _DatasetIndex(pydantic.BaseModel):
    bulk_data: dict[str, Any]
    last_updated: datetime

    @property
    def time_since_last_updated(self) -> timedelta:
        return datetime.now(pytz.utc) - self.last_updated

    def is_stale(self) -> bool:
        return self.time_since_last_updated > CACHE_STALE_AFTER

    @classmethod
    def _query_bulk_data(cls, bulk_type: BulkType) -> dict[str, Any]:
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
    def from_query(cls, bulk_type: BulkType) -> '_DatasetIndex':
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
        img_type: ImageType = 'normal',
        bulk_type: BulkType = 'default_cards',
        *,
        ds_dir: Path | None = None,
    ):
        self._img_type = img_type
        self._bulk_type = bulk_type
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
    def img_type(self) -> ImageType:
        return self._img_type

    @property
    def bulk_type(self) -> BulkType:
        return self._bulk_type

    @property
    def ds_dir(self) -> Path:
        return self.__ds_dir

    @property
    def _path_sets_dir(self) -> Path:
        return self.__ds_dir / 'sets'

    # ~=~=~ DB ~=~=~

    def _read_index(self) -> _DatasetIndex | None:
        if self.__path_index.exists():
            with AtomicOpen(self.__path_index, 'r') as fp:
                return _DatasetIndex.model_validate_json(fp.read())
        else:
            return None

    def _update_index(self, index: _DatasetIndex):
        with AtomicOpen(self.__path_index, 'w') as fp:
            fp.write(index.model_dump_json())

    # ~=~=~ bulk data ~=~=~

    def invalidate_cache(self):
        # call this to force a cache update
        self.__path_index.unlink(missing_ok=True)
        self.__path_bulk.unlink(missing_ok=True)
        return self

    # ~=~=~ bulk data ~=~=~

    def _download_bulk_data(self) -> Path:
        index = self._read_index()
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
            self._update_index(new_index)
        # done
        assert self.__path_bulk.exists()
        return self.__path_bulk

    # ~=~=~ yield card faces ~=~=~

    def __iter__(self) -> Iterator['ScryfallCardFace']:
        return self.yield_all()

    def yield_all(self, shared_proxy: ProxyDownloader | None = None) -> Iterator['ScryfallCardFace']:
        # fetch the bulk data
        path_bulk = self._download_bulk_data()
        query = f"""
            SELECT
                t.id,
                t.oracle_id,
                t.name,
                t.set AS set_code,
                t.set_name,
                img.image_uri,
                '{self._img_type}' AS img_type,
                '{self._bulk_type}' AS bulk_type
            FROM read_json('{path_bulk}') t
            CROSS JOIN LATERAL (
              VALUES
                (t.image_uris.{self._img_type}),
                (t.card_faces[1].image_uris.{self._img_type}),
                (t.card_faces[2].image_uris.{self._img_type})
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
    ) -> int:
        total_cards = 0

        def _itr():
            nonlocal total_cards
            for item in self.yield_all(shared_proxy=proxy):
                yield item.url_path_pair
                total_cards += 1

        if not proxy:
            proxy = ProxyDownloader()

        proxy.download_threaded(
            _itr(),
            exists_mode='skip',
            verbose=verbose,
            make_dirs=True,
            threads=threads,
            attempts=128,
            timeout=8,
        )
        return total_cards


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
    total_cards = ds.download_all(threads=args.download_threads, verbose=True)
    logger.info(f'Finished downloading {total_cards} images for: {repr(args.bulk_type)} {repr(args.img_type)}')


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
