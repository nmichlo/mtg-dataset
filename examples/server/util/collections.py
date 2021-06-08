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
