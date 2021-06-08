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

import os
from collections import Counter
from datetime import timedelta
from glob import glob
from logging import getLogger
from types import SimpleNamespace
from typing import Optional
from typing import Tuple

import ijson
from cachier import cachier
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from mtgdata.util.inout import get_json
from mtgdata.util.inout import smart_download
from mtgdata.util.proxy import ProxyDownloader
from mtgdata.util.proxy import scrape_proxies


logger = getLogger(__name__)


# ========================================================================= #
# Dataset - Vars                                                            #
# ========================================================================= #


CACHE_STALE_AFTER = timedelta(days=365)


def _data_dir(data_root: Optional[str], relative_path: Optional[str]):
    if data_root is None:
        data_root = os.environ.get('DATA_ROOT', 'data')
    # check root exists
    if not os.path.isdir(data_root):
        raise NotADirectoryError(f'data_root should be an existing directory: {data_root}')
    # return joined path
    if relative_path is None:
        return data_root
    return os.path.join(data_root, relative_path)


# ========================================================================= #
# Scryfall API Helper                                                       #
# ========================================================================= #


class ScryfallAPI(object):

    IMG_TYPES = {
        'small':       'jpg',
        'border_crop': 'jpg',
        'normal':      'jpg',
        'large':       'jpg',
        'png':         'png',
        'art_crop':    'jpg',
    }

    IMG_SHAPES = {
        # aspect ratio ~= 7:5 (H = 1.4 W)
        'small':       (204,  146, 3),
        'border_crop': (680,  480, 3),
        'normal':      (680,  488, 3),
        'large':       (936,  672, 3),
        'png':         (1040, 745, 3),
        'art_crop':    None,
    }

    @classmethod
    def api_download(cls, endpoint):
        logger.info(f'[Scryfall]: {endpoint}')
        return get_json(os.path.join(f'https://api.scryfall.com', endpoint))['data']

    @classmethod
    def get_bulk_info(cls, data_root=None):
        @cachier(stale_after=CACHE_STALE_AFTER, cache_dir=_data_dir(data_root=data_root, relative_path='cache/scryfall'))
        def _get_bulk_info():
            return {data['type']: data for data in cls.api_download('bulk-data')}

        return _get_bulk_info()

    @classmethod
    def bulk_iter(cls, bulk_type='default_cards', data_root=None, overwrite=False, return_bulk_info=True):
        # query information
        bulk_info = cls.get_bulk_info(data_root=data_root)
        assert bulk_type in bulk_info, f"Invalid {bulk_type=}, must be one of: {list(bulk_info.keys())}"
        # download bulk data if needed
        download_uri = bulk_info[bulk_type]['download_uri']
        path = smart_download(download_uri, folder=_data_dir(data_root, 'scryfall/bulk'), overwrite=overwrite)
        bulk_name = os.path.basename(path)
        # open json efficiently - these files are large!!!
        with open(path, 'rb') as f:
            for bulk_idx, item in enumerate(ijson.items(f, 'item')):  # item is behavior keyword for ijson
                if return_bulk_info:
                    yield item, (bulk_idx, bulk_name)
                else:
                    yield item

    @classmethod
    def _make_face_item(cls, card, img_type: str, bulk_idx: int, bulk_name: str, f_idx: int = None):
        # handle the case where the card has a face or not
        if f_idx is None:
            face = card
            file_name = f"{card['id']}.{cls.IMG_TYPES[img_type]}"
        else:
            face = card['card_faces'][f_idx]
            file_name = f"{card['id']}_{f_idx}.{cls.IMG_TYPES[img_type]}"
        # make the object
        return SimpleNamespace(
            id=card['id'],
            oracle_id=card['oracle_id'],
            name=face['name'],
            set=card['set'],
            set_name=card['set_name'],
            img_uri=face['image_uris'][img_type],
            img_file=os.path.join(card['set'], file_name),
            bulk_idx=bulk_idx,
            bulk_name=bulk_name,
            face_idx=f_idx,
        )

    @classmethod
    def card_face_info_iter(cls, img_type='border_crop', bulk_type='default_cards', overwrite=False, data_root=None):
        # check image type
        assert img_type in cls.IMG_TYPES, f'Invalid image type {img_type=}, must be one of: {list(ScryfallAPI.IMG_TYPES.keys())}'
        # count number of skips
        count, skips = 0, 0
        # yield faces
        for card, (bulk_idx, bulk_name) in cls.bulk_iter(bulk_type=bulk_type, overwrite=overwrite, data_root=data_root, return_bulk_info=True):
            count += 1
            # skip cards with placeholder or missing images
            if card['image_status'] not in ('lowres', 'highres_scan'):
                if card['image_status'] not in ('placeholder', 'missing'):
                    logger.error(f'[SKIPPED] unknown card `image_status`: {card["image_status"]}')
                skips += 1
                continue
            # ANY CARD WITH (card_faces AND image_uris) WILL NOT HAVE (image_uris IN card_faces)
            # ie. if image_uris does not exist, check card_faces for data.
            # ALSO: any card without image_uris can be assumed not to have an illustration_id (not the other way around)
            # ie. the card_faces must be checked for these.
            if 'image_uris' not in card:
                if 'card_faces' not in card:
                    logger.error(f'[SKIPPED] Scryfall error, card with no `image_uris` also has no `card_faces`: {card}')
                    skips += 1
                    continue
                for f_idx, _ in enumerate(card['card_faces']):
                    yield cls._make_face_item(card, img_type=img_type, bulk_idx=bulk_idx, bulk_name=bulk_name, f_idx=f_idx)
            else:
                yield cls._make_face_item(card, img_type=img_type, bulk_idx=bulk_idx, bulk_name=bulk_name, f_idx=None)
        # done iterating over cards
        if skips > 0:
            logger.warning(f'[TOTAL SKIPS]: {skips} of {count} cards/faces')


# ========================================================================= #
# Scryfall Dataset                                                          #
# ========================================================================= #


class ScryfallDataset(ImageFolder):

    IMG_SHAPES = ScryfallAPI.IMG_SHAPES
    IMG_TYPES = ScryfallAPI.IMG_TYPES

    def __init__(self, transform=None, img_type='border_crop', bulk_type='default_cards', resize_incorrect=True, data_root: Optional[str] = None, force_update=False, download_threads: int = 64):
        self._img_type = img_type
        self._bulk_type = bulk_type
        # download missing files
        self.data_dir = self._download_missing(data_root=data_root, img_type=img_type, bulk_type=bulk_type, force_update=force_update, download_threads=download_threads)
        # initialise dataset
        super().__init__(self.data_dir, transform=None, target_transform=None, is_valid_file=None)
        # override transform function
        self.__transform = transform
        self.__resize_incorrect = resize_incorrect

    def __getitem__(self, idx):
        img, label = super(ScryfallDataset, self).__getitem__(idx)
        # make sure the item is the right size
        if self.__resize_incorrect:
            if img.size != self.img_size_pil:
                img = img.resize(self.img_size_pil)
        # transform if required
        if self.__transform is not None:
            img = self.__transform(img)
        # done!
        return img

    @property
    def img_type(self) -> str:
        return self._img_type

    @property
    def bulk_type(self) -> str:
        return self._bulk_type

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return self.IMG_SHAPES[self._img_type]

    @property
    def img_size_pil(self) -> Tuple[int, int]:
        return self.img_shape[1::-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self), *self.img_shape)

    @staticmethod
    def _get_tuples(img_type: str, bulk_type: str, data_root=None, force_update: bool = False):

        @cachier(stale_after=CACHE_STALE_AFTER, cache_dir=_data_dir(data_root=data_root, relative_path='cache/scryfall'))
        def __get_tuples(img_type: str, bulk_type: str):
            # get all card information
            url_file_tuples = []
            for face in tqdm(ScryfallAPI.card_face_info_iter(img_type=img_type, bulk_type=bulk_type, data_root=data_root), desc='Loading Image Info'):
                url_file_tuples.append((face.img_uri, face.img_file))
            # get duplicated
            uf_counts = {k: count for k, count in Counter(url_file_tuples).items() if (count > 1)}
            f_counts  = {k: count for k, count in Counter(file for url, file in url_file_tuples).items() if (count > 1)}
            u_counts  = {k: count for k, count in Counter(url for url, file in url_file_tuples).items() if (count > 1)}
            # duplicates are errors!
            if uf_counts: raise RuntimeError(f'duplicate files and urls found: {uf_counts}')
            if f_counts: raise RuntimeError(f'duplicate files found: {f_counts}')
            if u_counts: raise RuntimeError(f'duplicate urls found: {u_counts}')
            # return all the files
            return url_file_tuples

        return __get_tuples(img_type=img_type, bulk_type=bulk_type, overwrite_cache=force_update)

    @staticmethod
    def _download_missing(data_root: str, img_type: str, bulk_type: str, force_update: bool, download_threads: int = 64, download_attempts: int = 1024, download_timeout: int = 2):
        # get paths
        data_dir = _data_dir(data_root=data_root, relative_path=os.path.join('scryfall', bulk_type, img_type))
        proxy_dir = _data_dir(data_root=data_root, relative_path=os.path.join('cache/proxy'))
        # get url_file tuple information
        url_file_tuples = ScryfallDataset._get_tuples(img_type=img_type, bulk_type=bulk_type, force_update=force_update, data_root=data_root)  # TODO: wont properly recheck bulk, will only regenerate list of files
        # get existing files without root, ie. "<set>/<uuid>.<ext>"
        img_ext = ScryfallAPI.IMG_TYPES[img_type]
        strip_len = len(data_dir.rstrip('/') + '/')
        existing = set(path[strip_len:] for path in glob(os.path.join(data_dir, f'*/*.{img_ext}')))
        # obtain list of files that do not yet exist
        url_file_tuples = [(u, os.path.join(data_dir, f)) for u, f in url_file_tuples if f not in existing]
        # download missing images
        if url_file_tuples:
            proxy = ProxyDownloader(proxies=scrape_proxies(cache_dir=proxy_dir), req_min_remove_count=3)
            failed = proxy.download_threaded(url_file_tuples, exists_mode='skip', verbose=False, make_dirs=True, ignore_failures=True, threads=download_threads, attempts=download_attempts, timeout=download_timeout)
            if failed:
                raise FileNotFoundError(f'Failed to download {len(failed)} card images')
        # return
        return data_dir


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


if __name__ == '__main__':
    import argparse
    import logging

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bulk_type', type=str, default='default_cards')            # SEE: https://scryfall.com/docs/api/bulk-data
    parser.add_argument('-i', '--img-type', type=str, default='border_crop')               # SEE: https://scryfall.com/docs/api/images
    parser.add_argument('-d', '--data-root', type=str, default=_data_dir(None, None))      # download and cache directory location
    parser.add_argument('-f', '--force-download', action='store_true')                     # overwrite existing files and ignore caches
    parser.add_argument('-t', '--download_threads', type=int, default=os.cpu_count() * 2)  # number of threads to use when downloading files
    args = parser.parse_args()

    # download the dataset
    logging.basicConfig(level=logging.INFO)
    data = ScryfallDataset(
        bulk_type=args.bulk_type,
        img_type=args.img_type,
        data_root=args.data_root,
        force_update=args.force_download,
        download_threads=args.download_threads,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
