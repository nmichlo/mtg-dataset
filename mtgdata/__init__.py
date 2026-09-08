#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
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

__all__ = (
    "ScryfallDataset",
    "ScryfallCardFaceDatasetManager",
    "ScryfallImageType",
    "ScryfallBulkType",
    "Hdf5Dataset",
    "dataset_save_as_hdf5",
    "generate_converted_dataset",
)

import importlib
from typing import TYPE_CHECKING

# base dependencies only -- safe to import eagerly
from mtgdata.scryfall import ScryfallBulkType
from mtgdata.scryfall import ScryfallCardFaceDatasetManager
from mtgdata.scryfall import ScryfallDataset
from mtgdata.scryfall import ScryfallImageType

if TYPE_CHECKING:
    from mtgdata.scryfall_convert import dataset_save_as_hdf5
    from mtgdata.scryfall_convert import generate_converted_dataset
    from mtgdata.util.hdf5 import Hdf5Dataset

# these names pull in h5py, numpy, tqdm and torch, which live behind the
# `[convert]` extra. importing them here would make a plain
# `pip install mtgdata` unusable -- even `import mtgdata` would raise.
_LAZY_CONVERT = {
    "Hdf5Dataset": "mtgdata.util.hdf5",
    "NumpyDataset": "mtgdata.util.hdf5",
    "dataset_save_as_hdf5": "mtgdata.scryfall_convert",
    "generate_converted_dataset": "mtgdata.scryfall_convert",
}


def __getattr__(name: str) -> object:
    module = _LAZY_CONVERT.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list:
    return sorted({*__all__, *_LAZY_CONVERT})
