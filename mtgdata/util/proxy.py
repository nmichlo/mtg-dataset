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

import io
import os
import urllib.request
from collections import defaultdict
from datetime import timedelta
from logging import getLogger
from multiprocessing.pool import ThreadPool
from random import Random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import requests
from tqdm import tqdm


_LOGGER = getLogger(__name__)


# ============================================================================ #
# Proxy Scraping                                                               #
# ============================================================================ #


_PROXY_SOURCES = {}
_DEFAULT_SOURCE = None


def set_default_proxy_scraper(name: str):
    if name not in _PROXY_SOURCES:
        raise KeyError(f'Cannot set as default! Scrape function with name: {repr(name)} does not exist.')
    global _DEFAULT_SOURCE
    if _DEFAULT_SOURCE != name:
        _LOGGER.info(f'overridden default proxy scrape_fn: {repr(_DEFAULT_SOURCE)} -> {repr(name)}')
        _DEFAULT_SOURCE = name


def register_proxy_scraper(name: str, scrape_fn: Optional[Callable[[str], List[Dict[str, str]]]] = None, is_default: bool = False):
    if name in _PROXY_SOURCES:
        raise KeyError('scrape function with name: {repr(name)} already exists')
    # decorator
    def wrapper(scrape_fn: Callable[[str], List[Dict[str, str]]]):
        # just in case the decorator call was delayed
        assert name not in _PROXY_SOURCES
        _PROXY_SOURCES.setdefault(name, scrape_fn)
        _LOGGER.debug(f'registered proxy scrape_fn: {repr(name)}')
        # set the default
        if is_default:
            set_default_proxy_scraper(name=name)
    # decorator or function
    if scrape_fn is None:
        return wrapper
    else:
        wrapper(scrape_fn)


def scrape_proxies(source: Optional[str] = None, proxy_type: str = 'all', cache_dir: str = 'data/proxies/cachier', cached: bool = True) -> List[Dict[str, str]]:
    if source is None:
        if _DEFAULT_SOURCE is None:
            raise RuntimeError('no default proxy scrape function has been set.')
        source = _DEFAULT_SOURCE
        _LOGGER.info(f'using default proxy scrape function: {repr(source)}')
    # get source
    try:
        proxy_scrape_fn = _PROXY_SOURCES[source]
    except KeyError:
        raise KeyError(f'proxy scrape function with name: {repr(source)} does not exist. Valid scrape sources are: {sorted(_PROXY_SOURCES.keys())}')
    # wrap the function
    if cached:
        from cachier import cachier
        proxy_scrape_fn = cachier(
            stale_after=timedelta(days=1),
            backend='pickle',
            cache_dir=cache_dir
        )(proxy_scrape_fn)
    # obtain the proxies
    _LOGGER.info(f'scraping proxies from source: {repr(source)}')
    proxy_list = proxy_scrape_fn(proxy_type=proxy_type)
    _LOGGER.info(f'scrapped: {len(proxy_list)} proxies from source: {repr(source)}')
    # done!
    return proxy_list


# ============================================================================ #
# Proxy Scrapers                                                               #
# ============================================================================ #


def _requests_get(url, fake_user_agent=True, params=None):
    # fake a request from a browser
    return requests.get(
        url,
        headers={} if not fake_user_agent else {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'},
        params=params,
    )


@register_proxy_scraper('morph.io')
def _scrape_proxies_morph(proxy_type) -> List[Dict[str, str]]:

    assert 'MORPH_API_KEY' in os.environ, 'MORPH_API_KEY environment variable not set!'
    morph_api_key = os.environ['MORPH_API_KEY']
    morph_api_url = "https://api.morph.io/CookieMichal/us-proxy/data.json"

    query = f"select * from 'data' where (anonymity='elite proxy' or anonymity='anonymous')"

    if 'https' == proxy_type:
        query += " and https='yes'"
    elif 'http' == proxy_type:
        query += " and https='no'"
    elif 'all' == proxy_type:
        pass
    else:
        raise KeyError(f'invalid proxy_type: {proxy_type}')

    r = _requests_get(
        morph_api_url,
        params={
            'key': morph_api_key,
            'query': query
        }
    )

    proxies = []
    for row in r.json():
        proto = 'HTTPS' if row['https'] == 'yes' else 'HTTP'
        url = "{}://{}:{}".format(proto, row['ip'], row['port'])
        proxies.append({proto: url})

    return proxies


@register_proxy_scraper('free-proxy-list.net', is_default=True)
def _scrape_proxies_freeproxieslist(proxy_type) -> List[Dict[str, str]]:
    def can_add(https):
        if proxy_type == 'all':
            return True
        elif proxy_type == 'https':
            return https == 'yes'
        elif proxy_type == 'http':
            return https == 'no'
        else:
            raise KeyError(f'invalid proxy_type: {proxy_type}')

    try:
        from bs4 import BeautifulSoup
    except:
        raise ImportError('BeautifulSoup `bs4` is not installed, cannot scrape proxies!')

    page = _requests_get('https://free-proxy-list.net/', fake_user_agent=True)
    soup = BeautifulSoup(page.content, 'html.parser')
    rows = soup.find_all('tr', recursive=True)

    proxies = []
    for row in rows:
        try:
            ip, port, country, country_long, anonymity, google, https, last_checked = (elem.text for elem in row.find_all('td', recursive=True))
            # check this entry is an ip entry
            if len(ip.split('.')) != 4:
                raise ValueError('not an ip entry')
            # filter entries
            if not can_add(https):
                continue
            # make entry
            proto = 'HTTPS' if (https == 'yes') else 'HTTP'
            url = "{}://{}:{}".format(proto, ip, int(port))
            proxies.append({proto: url})
        except:
            pass

    return proxies


# ============================================================================ #
# Proxy Errors                                                                 #
# ============================================================================ #


class MalformedProxyError(Exception):
    """
    raised if a proxy does not follow the correct format.
    eg. `proxy = {'<protocol>': '<protocol>://<url>'}`
    """

class NoMoreProxiesError(Exception):
    """
    raise if the ProxyDownloader has run out of proxies!
    """

class ProxyDownloadFailedError(Exception):
    """
    raise if the ProxyDownloader download has failed
    """


# ============================================================================ #
# Proxy Download Helper                                                        #
# ============================================================================ #


def make_proxy_opener(proxy: Dict[str, str]):
    if len(proxy) != 1:
        raise MalformedProxyError(f'proxy dictionaries should only have one entry, the key is the protocol, and the value is the url... invalid: {proxy}')
    # build connection
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(proxy),
        urllib.request.ProxyBasicAuthHandler()
    )


def download_with_proxy(url: str, file: str, proxy: Dict[str, str], timeout: Optional[float] = 8):
    data = make_proxy_opener(proxy=proxy).open(url, timeout=timeout).read()
    # download to temp file in case there is an error
    temp_file = file + '.dl'
    with io.FileIO(temp_file, "w") as f:
        f.write(data)
    # make this atomic
    os.rename(temp_file, file)


def _skip_or_prepare_file(file: str, exists_mode: str, make_dirs: bool):
    """
    returns True if the file should be skipped, False otherwise.
    - also prepare the directories or deletion of the file!
    """
    if os.path.exists(file):
        # the file exists
        # make sure it is actually a file, not a directory or link
        if not os.path.isfile(file):
            raise IOError(f'the specified file is not a file: {file}')
        # handle the different modes
        if exists_mode == 'error':
            raise FileExistsError(f'the file already exists: {file}')
        elif exists_mode == 'skip':
            return True
        elif exists_mode == 'overwrite':
            os.unlink(file)
            _LOGGER.warning('overwriting file: {url}')
        else:
            raise KeyError(f'invalid exists_mode={repr(exists_mode)}')
    else:
        # the file does not exist
        # check the parent path
        parent_dir = os.path.dirname(file)
        if not os.path.exists(parent_dir):
            # the parent path does not exist
            if make_dirs:
                os.makedirs(parent_dir, exist_ok=True)
                _LOGGER.debug(f'[MADE] directory: {parent_dir}')
            else:
                raise FileNotFoundError(f'Parent directory does not exist: {parent_dir} Otherwise set make_dirs=True')
        else:
            # the parent path exists
            if not os.path.isdir(parent_dir):
                raise NotADirectoryError(f'Parent directory is not a directory: {parent_dir}')
    return False


# ============================================================================ #
# Proxy Downloader                                                             #
# ============================================================================ #


class ProxyDownloader:

    def __init__(
        self,
        proxies: Optional[Sequence[Dict[str, str]]],
        req_min_remove_count=5,
        req_max_fail_ratio=0.5,
    ):
        # default proxy scraping
        if proxies is None:
            proxies = scrape_proxies()
        # convert
        self._proxies = list(proxies)  # TODO: add support for raw proxy strings?
        # proxy statistics
        self._req_counts = defaultdict(int)
        self._req_fails = defaultdict(int)
        self._req_max_fail_ratio = req_max_fail_ratio
        self._req_min_remove_count = req_min_remove_count
        # random instance
        self._rand = Random()  # TODO: add round robbin mode?

    def random_proxy(self) -> Dict[str, str]:
        if len(self._proxies) <= 0:
            raise NoMoreProxiesError('The proxy downloader has run out of valid proxies.')
        # return a random proxy!
        index = self._rand.randint(0, len(self._proxies) - 1)
        return self._proxies[index]

    def _update_proxy(self, proxy: Dict[str, str], success: bool):
        (purl,) = proxy.values()
        # update uses and failures
        self._req_counts[purl] += 1
        self._req_fails[purl] += int(bool(not success))
        # make remove if there was an error
        counts, fails = self._req_counts[purl], self._req_fails[purl]
        if (counts > self._req_min_remove_count) and (fails / counts > self._req_max_fail_ratio):
            try:
                self._proxies.remove(proxy)
                del self._req_counts[purl]
                del self._req_fails[purl]
            except (ValueError, KeyError):
                pass  # removed in another thread

    def download_threaded(self, url_file_tuples: Sequence[Tuple[str, str]], exists_mode: str = 'error', verbose: bool = False, make_dirs: bool = False, ignore_failures=False, threads=64, attempts: int = 128, timeout: int = 8):
        # check inputs
        if len(url_file_tuples) < 0:
            return []

        def download(url_file):
            url, file = url_file
            try:
                self.download(url=url, file=file, exists_mode=exists_mode, verbose=verbose, make_dirs=make_dirs, attempts=attempts, timeout=timeout)
            except ProxyDownloadFailedError:
                if ignore_failures:
                    return url, file
                else:
                    raise
            return None

        def get_desc():
            if ignore_failures:
                return f'Downloading [p={len(self._proxies)},t={threads},f={len(failed)}]'
            else:
                return f'Downloading [p={len(self._proxies)},t={threads}]'

        # download all files, keeping track of failed items!
        failed = []
        with ThreadPool(processes=threads) as pool:
            with tqdm(desc=get_desc(), total=len(url_file_tuples)) as pbar:
                for pair in pool.imap_unordered(download, url_file_tuples):
                    if pair:
                        failed.append(pair)
                    pbar.desc = get_desc()
                    pbar.update()

        # return all tuples for failed attempts
        return failed

    def download(self, url, file, exists_mode='error', verbose=False, make_dirs=False, attempts: int = 128, timeout: int = 8):
        """
        Download a file using random proxies.
        """
        if _skip_or_prepare_file(file=file, exists_mode=exists_mode, make_dirs=make_dirs):
            if verbose:
                _LOGGER.debug(f"[SKIPPED]: {file} | {url}")
            return
        # attempt download
        for i in range(attempts):
            proxy = self.random_proxy()
            try:
                download_with_proxy(url, file, proxy=proxy, timeout=timeout)
                if verbose:
                    _LOGGER.info(f"[DOWNLOADED]: {file} | {url}")
                self._update_proxy(proxy, success=True)
                return
            except Exception as e:
                if verbose:
                    _LOGGER.debug(f"[FAILED ATTEMPT {i + 1}]: {file} | {url} -- {e}")
                self._update_proxy(proxy, success=False)
        # download failed
        raise ProxyDownloadFailedError(f"[FAILED] tries={attempts}: {file} | {url}")


# ============================================================================ #
# Entrypoint                                                                   #
# ============================================================================ #


if __name__ == '__main__':

    def _command_line_app():
        import argparse
        import logging

        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--cache-dir', type=str, default='data/cache/proxies')
        parser.add_argument('-t', '--proxy-type', type=str, default='all')
        parser.add_argument('-s', '--proxy-source', type=str, default=None)
        parser.add_argument('-f', '--force-download', action='store_true')
        args = parser.parse_args()

        # download the proxies
        logging.basicConfig(level=logging.DEBUG)
        ProxyDownloader(
            proxies=scrape_proxies(
                source=args.proxy_source,
                proxy_type=args.proxy_type,
                cache_dir=args.cache_dir,
                cached=not args.force_download,
            )
        )

    _command_line_app()


# ============================================================================ #
# END                                                                          #
# ============================================================================ #


