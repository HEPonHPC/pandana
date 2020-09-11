"""This module defines the class Cut.
"""
import numpy as np
import pandas as pd


class Cut:
    # TODO: Cut makes use of protected data from Loader; this should be removed.

    """Represents a selection criterion to be applied to a dataframe."""

    def __init__(self, cut, invert=False):
        if not isinstance(cut, list):
            cut = [cut]
        if not isinstance(invert, list):
            invert = [invert]
        # TODO: rather than maintaining separate lists, consider
        # a data structure that unites the bool and the Cut, so
        # that they can't get out-of-sync, and avoiding the need for
        # a zip() to combine them in __call__.
        assert len(cut) == len(invert), "invalid cut definition!"

        self._cut = list(cut)
        self._invert = list(invert)

        # index that runs over the cutlist
        self.filteridx = 0

        # use these to keep track of cuts already computed
        self._filter = [0] * len(self._cut)
        self._cutid = [0] * len(self._cut)

    def reset_cutindices(self):
        """Set _filter and _cutid to original state."""
        # need to reset after use by Loader
        self._filter = [0] * len(self._cut)
        self._cutid = [0] * len(self._cut)

    def __call__(self, tables):
        # tables is empty anyway. takes negligible time
        if not tables.gone:
            cutlist = [
                (~c(tables) if b else c(tables))
                for c, b in zip(self._cut, self._invert)
            ]
            if not tables.interactive:
                # tables is empty anyway. takes negligible time
                # return dummy cut series
                return cutlist[0]
            else:
                cut_df = pd.concat(cutlist, axis=1).all(axis=1)
                cutidx = cut_df.index[np.where(cut_df)]
                tables._indices = cutidx
                return cut_df

        # cutid holds the filtered index list after applying the cut on the entire dataset
        cutidx = self._cutid[self.filteridx]
        # actual cut that was already computed
        applycut = self._filter[self.filteridx]

        # cut is being computed for the first time
        if cutidx == 0:
            cut0 = self._cut[self.filteridx](tables)
            if self._invert[self.filteridx]:
                cut0 = ~cut0

            # find filtered index list
            cutidx = cut0.index[np.where(cut0)]

            applycut = cut0
            self._cutid[self.filteridx] = cutidx
            self._filter[self.filteridx] = applycut

        self.filteridx += 1

        # check if filtered index list is empty and if so, stop computing other cuts
        canfiltermore = all([len(cutidx.codes[k]) for k in range(len(cutidx.codes))])

        # if its not empty, run next cut on the filtered list rather than the entire dataset
        if len(self._cut) > self.filteridx and canfiltermore:
            return self(tables[cutidx])
        # use filtered index list for evaluation of the var that comes later
        tables._indices = cutidx
        self.filteridx = 0
        self.reset_cutindices()
        return applycut

    def __and__(self, other):
        return Cut(self._cut + other._cut, self._invert + other._invert)

    def __invert__(self):
        return Cut(self._cut, [not b for b in self._invert])

    def __or__(self, other):
        def orcut(tables):
            idx = tables._indices
            df1 = self(tables)
            tables._indices = idx
            df2 = other(tables)
            # or operators are not commutative???
            compare = pd.concat([df1, df2], axis=1, join="outer").fillna(False)
            return compare.any(axis=1)

        return Cut(orcut)
