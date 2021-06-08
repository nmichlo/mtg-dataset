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
