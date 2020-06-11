
# ========================================================================= #
# collections                                                                   #
# ========================================================================= #


class CircularQueue(object):
    def __init__(self, max_size):
        self.list = [None] * max_size
        self.max_size = max_size
        self.size = 0
        self.end = 0
        self.on_remove = None

    def __iter__(self):
        # not sure why len + getitem doesnt work properly
        yield from self.list[:self.size]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i >= self.size:
            raise KeyError('Index out of bounds')
        return self.list[i]

    def append(self, item):
        if self.size < self.max_size:
            self.size += 1
            replaced = None
        else:
            replaced = self.first()
        self.list[self.end] = item
        self.end = (self.end + 1) % self.max_size
        return replaced

    def first(self):
        return self[(self.end - self.size) % self.max_size]

    def last(self):
        return self[(self.end - 1) % self.max_size]

    def __repr__(self):
        return self.list[:self.size].__repr__()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
