import h5py
from torch.utils.data import Dataset


# ========================================================================= #
# hdf5 utils                                                                #
# NOTE: this class is taken from disent -- github.com/nmichlo/disent        #
# ========================================================================= #


class H5pyDataset(Dataset):
    """
    This class supports pickling and unpickling of a read-only
    SWMR h5py file and corresponding dataset.

    WARNING: this should probably not be used across multiple hosts?
    """

    def __init__(self, h5_path: str, h5_dataset_name: str, transform=None):
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
