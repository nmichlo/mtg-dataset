#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
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

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Literal, NamedTuple
from typing import Optional
from typing import Tuple

import duckdb
import pytz
from PIL import Image

from doorway import io_download
from doorway.x import ProxyDownloader
from mtgdata.util.inout import get_json


logger = logging.getLogger(__name__)


# ========================================================================= #
# Dataset - Vars                                                            #
# ========================================================================= #


CACHE_STALE_AFTER = timedelta(days=365)


def _data_dir(data_root: Optional[str], relative_path: Optional[str], env_key='DATA_ROOT') -> str:
    if data_root is None:
        data_root = os.environ.get(env_key, 'data')
    # check root exists
    if os.path.exists(data_root):
        if not os.path.isdir(data_root):
            raise NotADirectoryError(f'specified path for {repr(env_key)} is not a directory: {repr(data_root)} ({repr(os.path.abspath(data_root))})')
    else:
        os.makedirs(data_root, exist_ok=True)
        logger.warning(f'created missing directory for {repr(env_key)}: {repr(data_root)} ({repr(os.path.abspath(data_root))})')
    # return joined path
    if relative_path is None:
        return data_root
    return os.path.join(data_root, relative_path)

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


class ScryfallCardFace(NamedTuple):
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


class ScryfallCardFaceDataset:

    def __init__(
        self,
        img_type: ImageType = 'normal',
        bulk_type: BulkType = 'default_cards',
        *,
        root_dir: Path | None = None,
    ):
        self._img_type = img_type
        self._bulk_type = bulk_type
        # get root dir
        if root_dir is None:
            root_dir = os.environ.get('DATA_ROOT', 'data')
            root_dir = Path(root_dir) / 'scryfall' / bulk_type / img_type
        # check dirs
        root_dir.mkdir(parents=True, exist_ok=True)
        if not root_dir.is_dir():
            raise NotADirectoryError(f'path is not a directory: {root_dir}')
        # initialise dataset
        self._path_sets_dir = root_dir / 'sets'
        self._path_ddb = root_dir / 'index.ddb'
        self._path_bulk = root_dir / 'bulk.json'
        self._key_bulk_info = f'bulk_info'
        # initialize
        self._download_bulk_data()

    # ~=~=~ DB ~=~=~

    def _get_db_connection(self, conn: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
        if conn is not None:
            return conn
        # should cache
        conn = duckdb.connect(self._path_ddb)
        conn.execute("""CREATE TABLE IF NOT EXISTS cache (id STRING PRIMARY KEY NOT NULL, last_updated TIMESTAMPTZ NOT NULL, data JSON NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS card_last_updated (card_id STRING PRIMARY KEY NOT NULL, card_set_code STRING NOT NULL, last_updated TIMESTAMPTZ NOT NULL)""")
        return conn

    # -- cache -- #

    def _db_get_cache_value(self, id: str, *, conn: duckdb.DuckDBPyConnection | None = None) -> Tuple[dict | None, timedelta]:
        conn = self._get_db_connection(conn)
        cursor = conn.execute(f"""SELECT data, last_updated FROM cache WHERE id = ?""", [id])
        result = cursor.fetchone()
        if result is not None:
            return json.loads(result[0]), datetime.now(pytz.utc) - result[1]
        return None, datetime.now() - datetime.min

    def _db_put_cache_value(self, id: str, data: dict, *, conn: duckdb.DuckDBPyConnection | None = None):
        conn = self._get_db_connection(conn)
        cursor = conn.execute(f"""INSERT OR REPLACE INTO cache (id, last_updated, data) VALUES (?, now(), ?)""",[id, data])
        return cursor

    def _db_del_cache_value(self, id: str, *, conn: duckdb.DuckDBPyConnection | None = None):
        conn = self._get_db_connection(conn)
        cursor = conn.execute(f"""DELETE FROM cache WHERE id = ?""", [id])
        return cursor

    # ~=~=~ bulk data ~=~=~

    @classmethod
    def _get_bulk_data_info(cls, bulk_type: BulkType):
        bulk_data_list = get_json(f'https://api.scryfall.com/bulk-data')['data']
        for item in bulk_data_list:
            if item['type'] == bulk_type:
                return item
        raise ValueError(
            f'bulk type {bulk_type} not found in bulk data list, valid types are: {[d["type"] for d in bulk_data_list]}'
        )

    def _download_bulk_data(self, force_update: bool = False) -> Path:
        bulk_data_info, last_updated = self._db_get_cache_value(self._key_bulk_info)
        # stale / non-existent
        if bulk_data_info is None or last_updated > CACHE_STALE_AFTER or force_update:
            bulk_data_info = self._get_bulk_data_info(self._bulk_type)
            # now download bulk data
            self._db_del_cache_value(self._key_bulk_info)
            self._path_bulk.unlink(missing_ok=True)
            io_download(
                src_url=bulk_data_info['download_uri'],
                dst_path=str(self._path_bulk),
                overwrite_existing=True,
            )
            self._db_put_cache_value(self._key_bulk_info, bulk_data_info)
        # result
        return self._path_bulk

    # ~=~=~ yield card faces ~=~=~

    def yield_card_faces(self) -> Iterator['ScryfallCardFace']:
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
            FROM read_json('{self._path_bulk}') t
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
            yield ScryfallCardFace(*row, _sets_dir=self._path_sets_dir)





class ScryfallAPI:
    pass
class ScryfallDataset:
    pass

# ========================================================================= #
# ENTRY POINT - HELPERS                                                     #
# ========================================================================= #


def _make_parser_scryfall_prepare(parser=None):
    # make default parser
    if parser is None:
        import argparse
        parser = argparse.ArgumentParser()
    # these should match scryfall_convert.py
    parser.add_argument('-b', '--bulk_type', type=str, default='default_cards',                    help="[default_cards|all_cards|oracle_cards|unique_artwork]. For more information, see: https://scryfall.com/docs/api/bulk-data")
    parser.add_argument('-i', '--img-type', type=str, default='normal',                            help="[png|border_crop|art_crop|large|normal|small]. For more information, see: https://scryfall.com/docs/api/images")
    parser.add_argument('-d', '--data-root', type=str, default=_data_dir(None, None),              help="download and cache directory location")
    parser.add_argument('-f', '--force-update', action='store_true',                               help="overwrite existing files and ignore caches")
    parser.add_argument('-t', '--download_threads', type=int, default=max(os.cpu_count() * 2, 256), help="number of threads to use when downloading files")
    parser.add_argument('--clean-invalid-images', action='store_true',                             help="delete invalid image files")
    return parser


def _run_scryfall_prepare(args):
    ds = ScryfallCardFaceDataset(
        bulk_type=args.bulk_type,
        img_type=args.img_type,
        force_update=args.force_update,
    )
    for i, item in enumerate(ds.yield_card_faces_and_download()):
        print(item)

    # # download the dataset
    # data = ScryfallDataset(
    #     bulk_type=args.bulk_type,
    #     img_type=args.img_type,
    #     data_root=args.data_root,
    #     force_update=args.force_update,
    #     download_threads=args.download_threads,
    #     clean_invalid_images=args.clean_invalid_images,
    # )
    # # information
    # logger.info(f'Finished downloading {len(data)} images for: {repr(data.bulk_type)} {repr(data.img_type)}')


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
