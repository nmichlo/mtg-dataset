import os
from logging import getLogger

import requests
from tqdm import tqdm


logger = getLogger(__name__)


# ========================================================================= #
# io                                                                        #
# ========================================================================= #


# TODO: merge with proxy
def direct_download(url, path):
    # paths
    path_temp = path + '.dl'
    # download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path_temp, 'wb') as f:
            pbar = tqdm(unit="B", total=int(r.headers['Content-Length']), unit_scale=True, unit_divisor=1024, desc=f'Downloading: {path}')
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)
        # atomic
        os.rename(path_temp, path)


# TODO: merge with proxy
def smart_download(url, file=None, folder=None, overwrite=False):
    # get names
    if file is None:
        file = os.path.basename(url).split('?')[0]
    if folder is None:
        folder, file = os.path.split(file)
    assert not os.path.dirname(file), 'directory must be specified using folder.'
    # check path
    path = os.path.join(folder, file)
    if os.path.exists(path) and not overwrite:
        logger.debug(f'[SKIPPED] skipped existing: {path}')
        return path
    # mkdirs
    if not os.path.exists(folder):
        logger.debug(f'[MADE] made parent folder: {folder}')
        os.makedirs(folder, exist_ok=True)
    # download
    direct_download(url, path)
    # return path to file
    return path


# TODO: merge with proxy
def get_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
