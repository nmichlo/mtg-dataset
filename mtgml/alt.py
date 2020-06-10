import asyncio
import functools
from io import FileIO
from urllib.parse import urlparse
import aiohttp
import os
from proxybroker import Broker, ProxyPool
from tqdm import tqdm


# ========================================================================= #
# alt                                                                   #
# ========================================================================= #


async def _download(url, file, proxy_pool, timeout, loop, attempts=10, verbose=False):
    # proxy_handler = urllib.request.ProxyHandler(self.random_proxy())
    # proxy_auth_handler = urllib.request.ProxyBasicAuthHandler()
    # opener = urllib.request.build_opener(proxy_handler, proxy_auth_handler)
    # read = opener.open(url, timeout=3).read()  # add timeout
    # f = io.FileIO(file, "w")
    # f.write(read)

    for i in range(attempts):
        try:
            # get proxy
            proxy = await proxy_pool.get(scheme=urlparse(url).scheme)
            proxy_url = f'http://%s:%d' % (proxy.host, proxy.port)

            try:
                # try download
                _timeout = aiohttp.ClientTimeout(total=timeout)
                async with aiohttp.ClientSession(timeout=_timeout, loop=loop) as session, session.get(url, proxy=proxy_url) as response:
                    data = await response.read()
                    # download to temp file in case there is an error
                    temp_file = file + '.dl'
                    with FileIO(temp_file, "w") as wfile:
                        wfile.write(data)
                    # make this atomic
                    os.rename(temp_file, file)
                # success
                if verbose:
                    tqdm.write(f"[Downloaded] try={i}: {url} : {file}")
                return None
            except Exception as e:
                # fail
                if verbose:
                    tqdm.write(f'[Error] try={i}: {url} : {file} | {e}')

            # replace used proxy
            proxy_pool.put(proxy)
        except Exception as e:
            tqdm.write(f'[Unexpected Error] Something went wrong with getting proxies: {e}')

    tqdm.write(f"[Failed] tries={attempts}: {url} : {file}")
    return url, file



async def worker(semaphore, coro, args):
    async with semaphore:
        return await coro(*args)

async def _download_all(url_file_tuples, proxy_pool, workers=64, timeout=10, loop=None, attempts=10, verbose=False):
    partial = functools.partial(_download, proxy_pool=proxy_pool, timeout=timeout, loop=loop, attempts=attempts, verbose=verbose)

    # pool
    semaphore = asyncio.BoundedSemaphore(workers)
    tasks = [worker(semaphore, partial, (url, file)) for url, file in url_file_tuples]

    failed = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(url_file_tuples), desc='Downloading'):
        fail = await task
        if fail:
            failed.append(fail)

    return failed


def proxy_download_all(url_file_tuples, attempts=256, workers=256, verbose=False):
    loop = asyncio.get_event_loop()

    proxies = asyncio.Queue(loop=loop)
    proxy_pool = ProxyPool(proxies)

    broker = Broker(
        proxies,
        timeout=8,
        max_conn=200,
        max_tries=3,
        verify_ssl=False,
        loop=loop,
    )

    tasks = asyncio.gather(
        broker.find(types=['HTTPS', ('HTTP', ('Anonymous', 'High'))], strict=False, limit=0),
        _download_all(url_file_tuples, proxy_pool, workers=workers, loop=loop, attempts=attempts, verbose=verbose),
    )

    return loop.run_until_complete(tasks)



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
