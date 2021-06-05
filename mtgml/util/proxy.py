#  MIT License
#
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import functools

from tqdm import tqdm
import json
import time
import random
import requests
import io
import os
import urllib.request
from multiprocessing.pool import ThreadPool


# ============================================================================ #
# Proxy Downloader                                                             #
# ============================================================================ #


class ProxyDownloader:
    # TODO: split proxy scraping and proxy downloading
    # TODO: update to use cachier

    def __init__(
            self,
            cache_folder,
            default_threads=64,
            default_attempts=128,
            default_timeout=8,
            req_min_remove_count=5,
            req_max_fail_ratio=0.5,
            proxy_type='both',
            force_scrape=False,
            check_proxies=False,
            days=10,
            file_name="proxies.json",
            logger=tqdm.write,
            proxy_scrape_fn=None,
    ):
        self.default_threads = default_threads
        self.default_attempts = default_attempts
        self.default_timeout = default_timeout
        self.logger = logger
        self.proxy_type = proxy_type  # TODO: update so this always has an effect
        self.proxies = []

        self._check_proxies = check_proxies

        self.req_counts = {}
        self.req_fails = {}
        self.req_max_fail_ratio = req_max_fail_ratio
        self.req_min_remove_count = req_min_remove_count

        time_ms = int(round(time.time() * 1000))
        file = os.path.join(cache_folder, file_name)

        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        # proxy scrape fn
        self._proxy_scrape_fn = _scrape_proxies_freeproxieslists if (proxy_scrape_fn is None) else proxy_scrape_fn

        if not os.path.isfile(file) or force_scrape:
            self.logger(f"Updating proxies file: {file}")
            self.proxies = self._scrape_and_dump(file, time_ms)
        else:
            try:
                with open(file, "r") as file_stream:
                    read_obj = json.load(file_stream)
                proxies = read_obj["proxies"]
                ms_time_created = read_obj["created"]
                if time_ms - ms_time_created > 1000 * 60 * 60 * 24 * days:
                    self.logger("Proxy file is old... Scraping proxies!")
                    self.proxies = self._scrape_and_dump(file, time_ms)
                else:
                    self.proxies = proxies
            except:
                self.logger("Invalid proxy file... Scraping proxies!")
                self.proxies = self._scrape_and_dump(file, time_ms)

    def random_proxy(self):
        index = random.randint(0, len(self.proxies) - 1)
        return self.proxies[index]

    def _scrape_and_dump(self, file, time_ms):
        proxies = self._proxy_scrape_fn(self.proxy_type)
        if self._check_proxies:
            proxies = self._get_good_proxies(proxies)
        with open(file, "w") as file_stream:
            json.dump({"created": time_ms, "proxies": proxies}, file_stream)
        self.logger(f"Saved: {len(proxies)} proxies to: {file}")
        return proxies

    def _get_good_proxies(self, proxies, check_url='https://api.scryfall.com'):
        def is_good_proxy(proxy):
            try:
                proxy_handler = urllib.request.ProxyHandler(proxy)
                proxy_auth_handler = urllib.request.ProxyBasicAuthHandler()
                opener = urllib.request.build_opener(proxy_handler, proxy_auth_handler)
                read = opener.open(check_url, timeout=self.default_timeout) #.read
            except:
                return None
            return proxy

        with ThreadPool(processes=self.default_threads) as pool:
            good_proxies = []
            with tqdm(pool.imap_unordered(is_good_proxy, proxies), total=len(proxies), desc="Good Proxies [0]") as t:
                for proxy in t:
                    if proxy:
                        good_proxies.append(proxy)
                        t.desc = f"Good Proxies [{len(good_proxies)}]"

        return good_proxies

    def download_threaded(self, url_file_tuples, skip_existing=True, verbose=False, make_dirs=False):
        if len(url_file_tuples) < 0:
            return []

        # make download function with arguments
        partial = functools.partial(self.download, skip_existing=skip_existing, verbose=verbose, make_dirs=make_dirs)
        func = lambda x: partial(*x)
        get_desc = lambda: f'Downloading [p={len(self.proxies)},t={self.default_threads}]'

        # download using threadpool
        with ThreadPool(processes=self.default_threads) as pool:
            failed = []
            with tqdm(desc=get_desc(), total=len(url_file_tuples)) as pbar:
                for pair in pool.imap_unordered(func, url_file_tuples):
                    if pair:
                        failed.append(pair)
                    pbar.desc = get_desc()
                    pbar.update()

        # return all tuples for failed attempts
        return failed

    def _update_proxy(self, proxy, success):
        assert len(proxy) == 1, 'This should never happen'
        purl = list(proxy.values())[0]

        c = self.req_counts.get(purl, 0) + 1
        f = self.req_fails.get(purl, 0) + int(bool(not success))
        self.req_counts[purl] = c
        self.req_fails[purl] = f
        if (c >= self.req_min_remove_count) and (f / c > self.req_max_fail_ratio):
            try:
                self.proxies.remove(proxy)
                del self.req_counts[purl]
                del self.req_fails[purl]
            except:
                pass # removed in another thread

    def download(self, url, file, skip_existing=True, verbose=False, make_dirs=False):
        # skip if file already exists
        if skip_existing and os.path.exists(file):
            if verbose:
                self.logger(f"[SKIPPED]: {file} | {url}")
            return None

        # make parent directories if they dont exist
        dirname = os.path.dirname(file)
        if not os.path.exists(dirname):
            if make_dirs:
                os.makedirs(dirname, exist_ok=True)
                if verbose:
                    self.logger(f'[MADE] directory: {dirname}')
            else:
                raise FileNotFoundError(f'Directory does not exist: {dirname} Otherwise set make_dirs=True')

        for i in range(self.default_attempts):
            proxy = self.random_proxy()
            try:
                # build connection
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler(proxy),
                    urllib.request.ProxyBasicAuthHandler()
                )
                read = opener.open(url, timeout=self.default_timeout).read()
                # download to temp file in case there is an error
                temp_file = file + '.dl'
                with io.FileIO(temp_file, "w") as f:
                    f.write(read)
                # make this atomic
                os.rename(temp_file, file)
                # log success
                if verbose:
                    self.logger(f"[DOWNLOADED] try={i}: {file} | {url}")
                # adjust stats
                self._update_proxy(proxy, success=True)
                return None
            except Exception as e:
                if verbose:
                    self.logger(str(e))
                self._update_proxy(proxy, success=False)

        self.logger(f"[FAILED] tries={self.default_attempts}: {file} | {url}")
        return url, file


# ============================================================================ #
# Proxy Scrapers                                                               #
# ============================================================================ #


def _scrape_proxies_morph(proxy_type):

    assert 'MORPH_API_KEY' in os.environ, 'MORPH_API_KEY environment variable not set!'
    morph_api_key = os.environ['MORPH_API_KEY']
    morph_api_url = "https://api.morph.io/CookieMichal/us-proxy/data.json"

    query = f"select * from 'data' where (anonymity='elite proxy' or anonymity='anonymous')"

    if 'https' == proxy_type:
        query += " and https='yes'"
    elif 'http' == proxy_type:
        query += " and https='no'"
    elif 'both' == proxy_type:
        pass
    else:
        raise KeyError(f'invalid proxy_type: {proxy_type}')

    r = requests.get(
        morph_api_url, params={
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


def _scrape_proxies_freeproxieslists(proxy_type):
    def can_add(https):
        if proxy_type == 'both':
            return True
        elif proxy_type == 'https':
            return https == 'yes'
        elif proxy_type == 'http':
            return https == 'no'
        else:
            raise KeyError(f'invalid proxy_type: {proxy_type}')

    from bs4 import BeautifulSoup
    page = _fetch_page_content('https://free-proxy-list.net/')
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


def _fetch_page_content(url):
    print(f'fetching: {url}')
    # fake a request from a browser
    return requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
    })


if __name__ == '__main__':
    ProxyDownloader('data/proxy')


# ============================================================================ #
# Helper                                                                       #
# ============================================================================ #


