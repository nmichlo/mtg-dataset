import json
import os


def init_dir(*paths, is_file=False) -> str:
    path = os.path.join(*paths)
    dirs = path if not is_file else os.path.dirname(path)
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    return path


# ============================================================================ #
# Json File Cache                                                              #
# ============================================================================ #

CACHE = True

class JsonCache(object):
    def __init__(self, cache_file, refresh=not CACHE):
        self.path = cache_file
        self.data = None
        self.refresh = refresh
        self.save = True

    def __enter__(self) -> dict:
        init_dir(self.path, is_file=True)
        if self.refresh or not os.path.isfile(self.path):
            self.data = {}
            self.refresh = False
        else:
            try:
                with open(self.path, 'r') as file_stream:
                    self.data = json.load(file_stream)
            except Exception as e:
                self.data = {}
                print("WARNING: Error loading cache: {} ({})".format(self.path, e))
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        init_dir(self.path, is_file=True)
        if self.save:
            with open(self.path, 'w') as file_stream:
                json.dump(self.data, file_stream)
        else:
            print('Skipping Save Cache: {}'.format(self.path))
        self.data = None


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
