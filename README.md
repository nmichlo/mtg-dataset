# Magic The Gathering - Dataset

## Features

**MTG Card Face Dataset**
- Automatically scrape and download card images from [Scryfall](https://scryfall.com)
- Multithreaded download through a randomized proxy list
- Only return valid images that are not placeholders
- Return all the faces of a card
- Normalise data, some images are incorrectly sized
- **Cached**

**Convert to HDF5**
- Convert the data to an hdf5 dataset
- Much faster than raw `jpg` or `png` image accesses
- Metadata json file allows linking back to original scryfall information.

**Pickle HD5F Dataset Class**
- Load the converted HDF5 dataset from disk from multiple threads / processes

## Download Images

**Command Line**

You can download all the `normal` quality [images](https://scryfall.com/docs/api/images)
from the `default` Scryfall [bulk](https://scryfall.com/docs/api/bulk-data) data
by running the file `mtgdata.scryfall`. Various arguments can be specified, please
see the argparse arguments at the bottom of the file for more information.

```bash
python3 mtgdata/scryfall
```

**Programmatically**

Alternatively you can download the images from within python by simply instantiating
the `mtgdata.ScryfallDataset` object. Similar arguments can be specified as that of the
command line approach.

```
from mtgdata import ScryfallDataset 
data = ScryfallDataset()
```

**Proxy Issues?**

The scrape logic used to obtain the proxy list for `mtgdata.utils.proxy.ProxyDownloader` will
probably go out of date. You can override the *default* scrape logic used by the Dataset download
logic  by registering a new scrape function.

```python3
from mtgdata.utils.proxy import register_proxy_scraper
from typing import List, Dict

@register_proxy_scraper(name='my_proxy_source', is_default=True)
def custom_proxy_scraper(proxy_type: str) -> List[Dict[str, str]]:
    # you should respect this setting, but we will just ignore it
    assert proxy_type in ('all', 'http', 'https')
    # proxies is a list of dictionaries, where each dictionary only has one entry:
    # - the key is the protocol
    # - the value is the matching full url
    return [
        {'HTTP': 'http://<my-http-proxy>.com'},
        {'HTTPS': 'http://<my-https-proxy>.com'},
    ]
```

## Convert Images to HDF5

The images can be convert to hdf5 format by running the file `mtgdata.scryfall_convert`.
Various arguments can be specified, please see the argparse arguments at the bottom of
the file for more information.

```bash
python3 mtgdata/scryfall_convert
```

The resulting data file will have the `data` key corresponding to the images data.

We provide a helper dataset class for loading this generated file.

```python3
from torch.utils.data import DataLoader
from mtgdata.util import H5pyDataset

# this h5py dataset supports pickling, and can be wrapped with a pytorch dataset.
data = H5pyDataset(
    h5_path='data/mtg-default_cards-normal-60459x224x160x3.h5',
    h5_dataset_name='data',
)

# you can wrap the dataset with a pytorch dataloader like usual, and specify more than one worker
dataloader = DataLoader(data, shuffle=True, num_workers=2, batch_size=64)

# to load the data into memory as a numpy array, you can call `data = data.numpy()`
# this will take a long time depending on your disk speed and use a lot of memory.
```
