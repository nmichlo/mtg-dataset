import time
from collections import deque
from typing import Sequence


def is_first_iter(items, n=1):
    # normalise the length
    assert n >= 0
    # yield the values
    yield from ((i < n, item) for i, item in enumerate(items))


def is_last_iter(items, n=1):
    iterator = iter(items)
    # lookahead stack
    prev = deque()
    # add items to prev
    assert n >= 0
    for i in range(n):
        try:
            prev.append(next(iterator))
        except StopIteration:
            break
    # check if there are next items and replace the current item
    while len(prev) >= n:
        # get the next item
        try:
            prev.append(next(iterator))
        except StopIteration:
            break
        # done
        yield False, prev.popleft()
    # return the remaining items
    yield from ((True, p) for p in prev)


def iter_pairs(items, empty_error=False):
    iterator = iter(items)
    # get the first item
    try:
        prev = next(iterator)
    except StopIteration:
        if empty_error:
            raise IndexError('there are no items to generate pairs')
        return
    # get the second item
    try:
        curr = next(iterator)
    except StopIteration:
        if empty_error:
            raise IndexError('there are not enough items to generate pairs')
        return
    # return the pairs
    while True:
        yield prev, curr
        # get the next pair
        try:
            prev, curr = curr, next(iterator)
        except StopIteration:
            break


def is_last_list(items: Sequence, n=1):
    items = list(items)
    # normalise the length
    assert n >= 0
    n = len(items) - n
    # yield the values
    return list((i >= n, item) for i, item in enumerate(items))


def is_first_list(items: Sequence, n=1):
    return list(is_first_iter(items, n=n))
