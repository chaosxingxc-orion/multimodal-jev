"""Spread each class across an epoch without replacement or weight changes.

A candidate permutation of existing real slots, not automatic admission of a
new training schedule. Each class receives seeded normalized-progress positions;
small classes no longer disappear after the first few round-robin rounds.
"""
from collections import defaultdict
from fractions import Fraction
import hashlib


def class_spread_indices(labels, *, seed, epoch):
    if (not labels or type(seed) is not int or type(epoch) is not int or epoch < 0
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError('nonempty class labels, integer seed and nonnegative epoch required')
    classes = defaultdict(list)
    for index, label in enumerate(labels):
        classes[label].append(index)
    def key(*parts):
        return hashlib.sha256(repr((seed, epoch, *parts)).encode()).digest()
    denominator = 2 ** 64
    ranked = []
    for label, indices in classes.items():
        # Preserve the existing seed-shuffled order within each class.
        phase = int.from_bytes(key('phase', label)[:8], 'big')
        for ordinal, index in enumerate(indices):
            position = Fraction(ordinal * denominator + phase, len(indices) * denominator)
            ranked.append((position, key('tie', label), index))
    result = [index for _, _, index in sorted(ranked)]
    if sorted(result) != list(range(len(labels))):
        raise ValueError('epoch permutation lost or repeated a real slot')
    return result
