import datetime
from glob import glob
from types import SimpleNamespace
import torchvision
import ijson as ijson
from cachier import cachier
import os
from tqdm import tqdm
import mtgml.util.inout
from mtgml.util.proxy import ProxyDownloader


# ========================================================================= #
# Dataset - Vars                                                            #
# ========================================================================= #

# get from environment
if 'DATASETS_ROOT' not in os.environ:
    DATASET_ROOT = './data/datasets'
    tqdm.write(f'[WARNING]: DATASETS_ROOT environment variable not set! Defaulting to: {DATASET_ROOT}')
else:
    DATASET_ROOT = os.environ['DATASETS_ROOT']

# check DATASET_ROOT
assert not os.path.exists(DATASET_ROOT) or os.path.isdir(DATASET_ROOT), f'{DATASET_ROOT=} should be a directory if it already exists!'

# cachier options
CACHIER_DIR = os.path.join(DATASET_ROOT, 'cachier')
CACHE_TIME = datetime.timedelta(days=3)

# where we store scryfall data and datasets
SCRYFALL_FOLDER = os.path.join(DATASET_ROOT, 'scryfall')
BULK_FOLDER = os.path.join(SCRYFALL_FOLDER, 'bulk')

# where we store the proxy files
PROXY_DIR = os.path.join(DATASET_ROOT, 'proxy')


# ========================================================================= #
# bulk                                                                      #
# ========================================================================= #


class ScryfallAPI(object):

    IMG_TYPES = {
        'small': 'jpg',
        'normal': 'jpg',
        'large': 'jpg',
        'png': 'png',
        'art_crop': 'jpg',
        'border_crop': 'jpg',
    }

    @staticmethod
    def get(endpoint):
        tqdm.write(f'[Scryfall]: {endpoint}')
        return mtgml.util.inout.get_json(os.path.join(f'https://api.scryfall.com', endpoint))['data']

    @staticmethod
    @cachier(stale_after=CACHE_TIME, cache_dir=CACHIER_DIR)
    def get_bulk_info():
        return {data['type']: data for data in ScryfallAPI.get('bulk-data')}

    @staticmethod
    def bulk_iter(bulk_type='default_cards', overwrite=False):
        # query information
        bulk_info = ScryfallAPI.get_bulk_info()
        assert bulk_type in bulk_info, f"Invalid {bulk_type=}, must be one of: {list(bulk_info.keys())}"
        # download bulk data if needed
        download_uri = bulk_info[bulk_type]['download_uri']
        path = mtgml.util.inout.smart_download(download_uri, folder=BULK_FOLDER, overwrite=overwrite)
        # open json efficiently - these files are large!!!
        with open(path, 'rb') as f:
            for item in ijson.items(f, 'item'):  # item is behavior keyword for ijson
                yield item

    @staticmethod
    def card_face_info_iter(img_type='small', bulk_type='default_cards', overwrite=False):
        # check image type
        assert img_type in ScryfallAPI.IMG_TYPES, f'Invalid image type {img_type=}, must be one of: {list(ScryfallAPI.IMG_TYPES.keys())}'
        img_ext = ScryfallAPI.IMG_TYPES[img_type]
        # yield faces
        for card in ScryfallAPI.bulk_iter(bulk_type=bulk_type, overwrite=overwrite):
            # ANY CARD WITH (card_faces AND image_uris) WILL NOT HAVE (image_uris IN card_faces)
            # ie. if image_uris does not exist, check card_faces for data.
            # ALSO: any card without image_uris can be assumed not to have an illustration_id (not the other way around)
            # ie. the card_faces must be checked for these.
            if 'image_uris' not in card:
                if 'card_faces' not in card:
                    tqdm.write(f'[SKIPPED] no image: {card}')
                    continue
                for i, face in enumerate(card['card_faces']):
                    yield SimpleNamespace(
                        id=card['id'],
                        set=card['set'],
                        name=face['name'],
                        image_uri=face['image_uris'][img_type],
                        image_file=os.path.join(card['set'], f"{card['id']}_{i}.{img_ext}"),
                    )
            else:
                yield SimpleNamespace(
                    id=card['id'],
                    set=card['set'],
                    name=card['name'],
                    image_uri=card['image_uris'][img_type],
                    image_file=os.path.join(card['set'], f"{card['id']}.{img_ext}"),
                )


class ScryfallDataset(torchvision.datasets.ImageFolder):

    def __init__(self, transform=None, img_type='small', bulk_type='default_cards', force_update=False):
        root_dir = os.path.join(SCRYFALL_FOLDER, bulk_type, img_type)

        # get url_file tuple information
        # TODO: wont properly recheck bulk, will only regenerate list of files
        url_file_tuples = ScryfallDataset._get_tuples(img_type=img_type, bulk_type=bulk_type, overwrite_cache=force_update)

        # get existing files without root, ie. <set>/<uuid>.<ext>
        # much faster than using os.path.relpath
        # TODO: maybe glob is just slow?
        img_ext = ScryfallAPI.IMG_TYPES[img_type]
        strip_len = len(root_dir.rstrip('/') + '/')
        existing = set(path[strip_len:] for path in glob(os.path.join(root_dir, f'*/*.{img_ext}')))
        # filter files that need downloading
        url_file_tuples = [(u, os.path.join(root_dir, f)) for u, f in url_file_tuples if f not in existing]

        # download missing images
        if url_file_tuples:
            proxy = ProxyDownloader(cache_folder=PROXY_DIR, default_threads=64, default_attempts=1024, default_timeout=2, req_min_remove_count=3, proxy_type='http')
            failed = proxy.download_threaded(url_file_tuples, skip_existing=True, verbose=False, make_dirs=True)
            assert not failed, f'Failed to download {len(failed)} card images'

        # initialise dataset
        super().__init__(root_dir, transform=transform, target_transform=None, is_valid_file=None)
        tqdm.write(f'[INITIALISED]: Scryfall {bulk_type=} {img_type=}')

    def __getitem__(self, index):
        # dont return the label
        return super(ScryfallDataset, self).__getitem__(index)[0]

    @staticmethod
    @cachier(stale_after=CACHE_TIME, cache_dir=CACHIER_DIR)
    def _get_tuples(img_type, bulk_type):
        url_file_tuples = []
        for face in tqdm(ScryfallAPI.card_face_info_iter(img_type=img_type, bulk_type=bulk_type), desc='Loading Image Info'):
            url_file_tuples.append((face.image_uri, face.image_file))

        assert len(url_file_tuples) == len(set(url_file_tuples))
        assert len(url_file_tuples) == len(set(file for url, file in url_file_tuples))
        assert len(url_file_tuples) == len(set(url for url, file in url_file_tuples))

        return url_file_tuples


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    ScryfallDataset(bulk_type='default_cards', img_type='normal')
    # ScryfallDataset(bulk_type='all_cards', img_type='normal')
