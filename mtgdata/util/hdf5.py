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
import warnings

import h5py
from torch.utils.data import Dataset as Dataset


# ========================================================================= #
# hdf5 utils                                                                #
# NOTE: this class is taken from disent -- github.com/nmichlo/disent        #
# ========================================================================= #


class NumpyDataset(Dataset):

    def __init__(self, data, transform=None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        yield from (self[i] for i in range(len(self._data)))

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        elem = self._data[item]
        if self._transform is not None:
            elem = self._transform(elem)
        return elem

    @property
    def shape(self):
        return self._data.shape


class Hdf5Dataset(Dataset):
    """
    This class supports pickling and unpickling of a read-only
    SWMR h5py file and corresponding dataset.

    WARNING: this should probably not be used across multiple hosts?
    """

    def __init__(self, h5_path: str, h5_dataset_name: str = 'data', transform=None):
        self._h5_path = h5_path
        self._h5_dataset_name = h5_dataset_name
        self._hdf5_file, self._hdf5_data = self._make_hdf5()
        self._transform = transform

    def _make_hdf5(self):
        # TODO: can this cause a memory leak if it is never closed?
        hdf5_file = h5py.File(self._h5_path, 'r', swmr=True)
        hdf5_data = hdf5_file[self._h5_dataset_name]
        return hdf5_file, hdf5_data

    def __iter__(self):
        yield from (self[i] for i in range(len(self._hdf5_data)))

    def __len__(self):
        return self._hdf5_data.shape[0]

    def __getitem__(self, item):
        elem = self._hdf5_data[item]
        if self._transform is not None:
            elem = self._transform(elem)
        return elem

    @property
    def shape(self):
        return self._hdf5_data.shape

    def numpy(self, warn=True):
        if warn:
            if self._transform is not None:
                warnings.warn('Transform is not applied to the data when numpy() is called.')
        return self._hdf5_data[:]

    def numpy_dataset(self) -> NumpyDataset:
        return NumpyDataset(data=self.numpy(warn=False), transform=self._transform)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.close()

    # CUSTOM PICKLE HANDLING -- h5py files are not supported!
    # https://docs.python.org/3/library/pickle.html#pickle-state
    # https://docs.python.org/3/library/pickle.html#object.__getstate__
    # https://docs.python.org/3/library/pickle.html#object.__setstate__
    # TODO: this might duplicate in-memory stuffs.

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_hdf5_file', None)
        state.pop('_hdf5_data', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hdf5_file, self._hdf5_data = self._make_hdf5()

    def close(self):
        self._hdf5_file.close()
        del self._hdf5_file
        del self._hdf5_data


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
