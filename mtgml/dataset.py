import os
from pathlib import Path
import torchvision.datasets
from mtgtools.MtgDB import MtgDB
from tqdm import tqdm

from mtgml.alt import proxy_download_all
from mtgml.json import JsonCache
from mtgml.proxy import ProxyDownloader


# ========================================================================= #
# Dataset - Vars                                                            #
# ========================================================================= #


# open path
DATASET_ROOT = Path(os.environ.get('DATASETS_ROOT', './data/datasets'))

# show warning
if 'DATASETS_ROOT' not in os.environ:
    print(f'[WARNING]: DATASETS_ROOT environment variable not set! Defaulting to: {DATASET_ROOT}')

# check that nothing is a file
assert not DATASET_ROOT.is_file(), f'{DATASET_ROOT=} should be a directory, not an existing file!'


# ========================================================================= #
# Dataset - Base                                                            #
# ========================================================================= #

# TODO: this is retarded
class DatasetFolder(object):
    """
    Helper class for the root dataset storage folder.
    """

    def __init__(self, *paths):
        self._relative = paths
        self._folder = DATASET_ROOT.joinpath(*paths)
        # check that nothing is a file
        assert not self._folder.is_file(), f'{self._folder=} should be a directory, not an existing file!'
        # make directories
        self._folder.mkdir(parents=True, exist_ok=True)

    def cd(self, *paths):
        return DatasetFolder(*self._relative, *paths)

    def to(self, *paths):
        return str(self._folder.joinpath(*paths).absolute())


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


class MtgHandler(object):
    def __init__(self, force_update=False):
        # create dataset folders
        dataset = DatasetFolder('mtgtools')
        # initialise and update dataset
        self.db = MtgDB(dataset.to('mtg.db'))
        if force_update or len(self.db.root.scryfall_sets) <= 0:
            self.db.scryfall_update(verbose=True)

    @property
    def sets(self):
        return self.db.root.scryfall_sets

    @property
    def cards(self):
        return self.db.root.scryfall_cards


class MtgDataset(torchvision.datasets.ImageFolder):

    IMG_TYPES = {
        'small': 'jpg',
        'normal': 'jpg',
        'large': 'jpg',
        'png': 'png',
        'art_crop': 'jpg',
        'border_crop': 'jpg',
    }

    def __init__(self, transform=None, img_type='small', force_update=False):
        assert img_type in MtgDataset.IMG_TYPES, f'Invalid image type {img_type=}, must be one of: {list(MtgDataset.IMG_TYPES.keys())=}'
        img_ext = MtgDataset.IMG_TYPES[img_type]

        # mtg data
        self.folder = DatasetFolder('mtg', img_type)
        self.db = MtgHandler(force_update=force_update)

        with JsonCache(self.folder.to('image_registry2.json')) as data:
            # load data if missing
            if ('images' not in data):
                url_file_tuples, sets = [], set()
                for card in tqdm(self.db.cards, desc='Finding Images'):
                    if card.image_uris and card.image_uris[img_type]:
                        uri = card.image_uris[img_type]
                        file = os.path.join(card.set, f'{card.id}.{img_ext}')  # {os.path.splitext(uri)[1].split("?")[0]}')
                        sets.add(card.set)
                        url_file_tuples.append((uri, file))
                # save data
                data['images'], data['sets'] = url_file_tuples, list(sets)

            # make directories
            for s in data['sets']:
                self.folder.cd(s)

            # get missing images (and add path prefix)
            existing = {str(path.relative_to(self.folder._folder)) for path in self.folder._folder.glob(f'**/*.{img_ext}')}
            url_file_tuples = [(uri, self.folder.to(file)) for uri, file in data['images'] if file not in existing]

        # download missing images
        if url_file_tuples:
            proxy = ProxyDownloader(cache_folder=DatasetFolder('proxy_alt').to(), default_threads=256, default_attempts=256, default_timeout=8)
            failed = proxy.download_threaded(url_file_tuples, skip_existing=True, verbose=False)
            assert not failed, f'Failed to download {len(failed)} card images'

        # initialise dataset
        super().__init__(self.folder.to(), transform=transform, target_transform=None, loader=None, is_valid_file=None)


    #
    #
    # def scryfall_cards_paths_uris(self, img_type='small', force=False):
    #     asrt_in(img_type, MtgHandler.IMG_TYPES)
    #     root = Dataset('mtg').cd(img_type, init=True)
    #     with JsonCache('./cache/mtg_uris_{}.json'.format(img_type), refresh=force) as uris:
    #         if 'resources' not in uris:
    #             resources = []
    #             for s in tqdm(self.scryfall_sets, desc='Building Image URI Cache [{}] ({})'.format(img_type, root)):
    #                 if s.api_type != 'scryfall':
    #                     continue
    #                 for card in s:
    #                     uri_file = MtgHandler._card_uri_to_file(card, folder=s.code, img_type=img_type)
    #                     if uri_file is not None:
    #                         resources.append(uri_file)
    #             print('URIS FOUND: {} of {}'.format(len(resources), len(self.scryfall_cards)))
    #             uris['resources'] = resources
    #         return [(u, root.to(f)) for u, f in uris['resources']]










    # def scryfall_card_from_uuid(self, uuid):
    #     if uuid in self.scryfall_uuids_to_index:
    #         return self.scryfall_cards[self.scryfall_uuids_to_index[uuid]]
    #     return None

    # IMG_TYPES = ['small', 'normal', 'large', 'png', 'art_crop', 'border_crop']
    #
    # def scryfall_cards_paths_uris(self, img_type='small', force=False):
    #     asrt_in(img_type, MtgHandler.IMG_TYPES)
    #     root = Dataset('mtg').cd(img_type, init=True)
    #     with JsonCache('./cache/mtg_uris_{}.json'.format(img_type), refresh=force) as uris:
    #         if 'resources' not in uris:
    #             resources = []
    #             for s in tqdm(self.scryfall_sets, desc='Building Image URI Cache [{}] ({})'.format(img_type, root)):
    #                 if s.api_type != 'scryfall':
    #                     continue
    #                 for card in s:
    #                     uri_file = MtgHandler._card_uri_to_file(card, folder=s.code, img_type=img_type)
    #                     if uri_file is not None:
    #                         resources.append(uri_file)
    #             print('URIS FOUND: {} of {}'.format(len(resources), len(self.scryfall_cards)))
    #             uris['resources'] = resources
    #         return [(u, root.to(f)) for u, f in uris['resources']]
    #
    # @staticmethod
    # def _card_uri_to_file(card, folder, img_type):
    #     if card.image_uris is None:
    #         return None
    #     uri = card.image_uris[img_type]
    #     if uri is None:
    #         return None
    #     # filename:
    #     ext = os.path.splitext(uri)[1].split('?')[0]
    #     name = re.sub('[^-a-z]', '', card.name.lower().replace(" ", "-"))
    #     file = os.path.join(folder, '{}__{}__{}__{}{}'.format(card.set, card.id, name, img_type, ext))
    #     return uri, file


# class MtgImages(util.LazyList):
#
#     def __init__(self, img_type='normal', predownload=False, handler=None):
#         if handler is None:
#             handler = MtgHandler()
#         resources = handler.scryfall_cards_paths_uris(img_type=img_type) #, force=predownload)
#
#         prox = Proxy('cache', default_threads=128, default_attempts=10, logger=tqdm.write)
#         if predownload:
#             download = [(u, f) for u, f in resources if not os.path.exists(f)]
#             print('PREDOWNLOADING DOWNLOADING: {} of {}'.format(len(download), len(resources)))
#             dirs = [util.init_dir(d) for d in { os.path.dirname(f)  for u, f in download } if not os.path.exists(d)]
#             print('MADE DIRECTORIES: {}'.format(len(dirs)))
#             prox.downloadThreaded(download)
#             super().__init__([util.Lazy(file, util.imread) for uri, file in resources])
#         else:
#             super().__init__([util.LazyFile(uri, file, prox.download, util.imread) for uri, file in resources])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    # MtgDataset(img_type='small')
    # MtgDataset(img_type='normal')

    import scrython

    for card in scrython.cards:
        print(card)





