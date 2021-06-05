import os
from datetime import timedelta
from logging import getLogger
from typing import Optional
from typing import Tuple

from torchvision.datasets import ImageFolder


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
        'normal':      'jpg',
        'large':       'jpg',
        'png':         'png',
        'art_crop':    'jpg',
        'border_crop': 'jpg',
    }

    IMG_SHAPES = {
        'small':       (204,  146, 3),
        'border_crop': (680,  480, 3),
        'normal':      (680,  488, 3),
        'large':       (936,  672, 3),
        'png':         (1040, 745, 3),
        'art_crop':    None,
    }

    @staticmethod
    def api_download(endpoint):
        from mtgml.util.inout import get_json
        logger.info(f'[Scryfall]: {endpoint}')
        return get_json(os.path.join(f'https://api.scryfall.com', endpoint))['data']

    @staticmethod
    def get_bulk_info(data_root=None):
        from cachier import cachier

        @cachier(stale_after=CACHE_STALE_AFTER, cache_dir=_data_dir(data_root=data_root, relative_path='cache/scryfall'))
        def _get_bulk_info():
            return {data['type']: data for data in ScryfallAPI.api_download('bulk-data')}

        return _get_bulk_info()

    @staticmethod
    def bulk_iter(bulk_type='default_cards', data_root=None, overwrite=False, bulk_info=True):
        import ijson
        from mtgml.util.inout import smart_download
        # query information
        bulk_info = ScryfallAPI.get_bulk_info(data_root=data_root)
        assert bulk_type in bulk_info, f"Invalid {bulk_type=}, must be one of: {list(bulk_info.keys())}"
        # download bulk data if needed
        download_uri = bulk_info[bulk_type]['download_uri']
        path = smart_download(download_uri, folder=_data_dir(data_root, 'scryfall/bulk'), overwrite=overwrite)
        bulk_name = os.path.basename(path)
        # open json efficiently - these files are large!!!
        with open(path, 'rb') as f:
            for bulk_idx, item in enumerate(ijson.items(f, 'item')):  # item is behavior keyword for ijson
                if bulk_info:
                    yield bulk_idx, bulk_name, item
                else:
                    yield item

    @staticmethod
    def card_face_info_iter(img_type='small', bulk_type='default_cards', overwrite=False, data_root=None):
        from types import SimpleNamespace
        # check image type
        assert img_type in ScryfallAPI.IMG_TYPES, f'Invalid image type {img_type=}, must be one of: {list(ScryfallAPI.IMG_TYPES.keys())}'
        img_ext = ScryfallAPI.IMG_TYPES[img_type]
        # count number of skips
        count, skips = 0, 0
        # yield faces
        for bulk_idx, bulk_name, card in ScryfallAPI.bulk_iter(bulk_type=bulk_type, overwrite=overwrite, data_root=data_root):
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
                for i, face in enumerate(card['card_faces']):
                    yield SimpleNamespace(
                        id=card['id'],
                        set=card['set'],
                        name=face['name'],
                        image_uri=face['image_uris'][img_type],
                        image_file=os.path.join(card['set'], f"{card['id']}_{i}.{img_ext}"),
                        bulk_idx=bulk_idx,
                        bulk_name=bulk_name,
                    )
            else:
                yield SimpleNamespace(
                    id=card['id'],
                    set=card['set'],
                    name=card['name'],
                    image_uri=card['image_uris'][img_type],
                    image_file=os.path.join(card['set'], f"{card['id']}.{img_ext}"),
                    bulk_idx=bulk_idx,
                    bulk_name=bulk_name,
                )
        # done iterating over cards
        if skips > 0:
            logger.warning(f'[TOTAL SKIPS]: {skips} of {count} cards/faces')


# ========================================================================= #
# Scryfall Dataset                                                          #
# ========================================================================= #


class ScryfallDataset(ImageFolder):

    IMG_SHAPES = ScryfallAPI.IMG_SHAPES
    IMG_TYPES = ScryfallAPI.IMG_TYPES

    def __init__(self, transform=None, img_type='small', bulk_type='default_cards', resize_incorrect=True, data_root: Optional[str] = None, force_update=False, download_threads: int = 64):
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
        from cachier import cachier

        @cachier(stale_after=CACHE_STALE_AFTER, cache_dir=_data_dir(data_root=data_root, relative_path='cache/scryfall'))
        def __get_tuples(img_type, bulk_type):
            from collections import Counter
            from tqdm import tqdm
            # get all card information
            url_file_tuples = []
            for face in tqdm(ScryfallAPI.card_face_info_iter(img_type=img_type, bulk_type=bulk_type, data_root=data_root), desc='Loading Image Info'):
                url_file_tuples.append((face.image_uri, face.image_file))
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
        from glob import glob
        from mtgml.util.proxy import ProxyDownloader
        from mtgml.util.proxy import scrape_proxies
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
    parser.add_argument('-b', '--bulk_type', type=str, default='default_cards')
    parser.add_argument('-i', '--img-type', type=str, default='normal')
    parser.add_argument('-d', '--data-root', type=str, default=_data_dir(None, None))
    parser.add_argument('-f', '--force-download', action='store_true')
    parser.add_argument('-t', '--download_threads', type=int, default=os.cpu_count() * 2)
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
