import pandas as pd
import numpy as np
import boost_histogram as bh

class Spectrum:
    """Represents a histogram of some quantity."""

    def __init__(self, loader, cut, var, weight = None):
        # Associate this spectrum with the loader
        loader.add_spectrum(self)

        self._cut = cut
        self._var = var
        self._wgt = weight

        self._dfvars = []
        self._dfwgts = []

    def fill(self, tables):
        # Compute the var and complete cut
        dfvar = self._var(tables)
        dfcut = self._cut(tables)

        # We allow the cut to have any subset of the indices used in the var
        # The two dataframes need to be aligned in this case
        if not dfvar.index.equals(dfcut.index):
            dfvar, dfcut = dfvar.align(dfcut, axis=0, join='inner')
        dfvar = dfvar.loc[dfcut.to_numpy()]

        # Compute weights
        if self._wgt is not None:
            dfwgt = self._wgt(tables)
            # align the weights to the var
            # TODO: Is 0 the right fill?
            dfwgt, _ = dfwgt.align(dfvar, axis=0, join='right', fill_value=0)
        else:
            dfwgt = pd.Series(1, dfvar.index, name='weight')

        self._dfvars.append(dfvar)
        self._dfwgts.append(dfwgt)

    def finish(self):
        assert(len(self._dfvars) == len(self._dfwgts))
        if len(self._dfvars) > 1:
            self._df = pd.concat(self._dfvars, axis=0)
            self._weight = pd.concat(self._dfwgts, axis=0)
        else:
            self._df = self._dfvars[0]
            self._weight = self._dfwgts[0]

    def df(self):
        return self._df

    def weight(self):
        return self._weight

    def entries(self):
        return self._df.shape[0]

    def histogram(self, bins, range=None):
        n, bins = bh.numpy.histogram(self._df, bins, range, weights=self._weight,
                                     storage = bh.storage.Double())
        return n, bins

    def integral(self):
        return self._weight.sum()

    def to_text(self, file_name, sep=" ", header=False):
        self._df.to_csv(file_name, sep=sep, index=True, header=header)

    def __add__(self, other):
        df = pd.concat([self._df, other._df])
        wgt = pd.concat([self._weight, other._weight])
        return FilledSpectrum(df, wgt)

class FilledSpectrum(Spectrum):
    """Construct a spectrum directly from a Series or DataFrame"""
    def __init__(self, df, weight):
        self._df = df
        self._weight = weight

    def fill(self):
        print("This spectrum was constructed already filled.")


# Save spectra to an hdf5 file. Takes a single or a list of spectra
def save_spectra(filename, spectra, groups):
    if not isinstance(spectra, list):
        spectra = [spectra]
    if not isinstance(groups, list):
        groups = [groups]
    assert len(spectra) == len(groups), "Each spectrum must have a group name."

    # idk why we are giving things to the store
    store = pd.HDFStore(filename, "w")

    for spectrum, group in zip(spectra, groups):
        store[group + "/dataframe"] = spectrum.df()
        store[group + "/weights"] = spectrum.weight()

    store.close()

def load_spectra(filename, groups):
    """Load a spectrum from a file."""
    if not isinstance(groups, list):
        groups = [groups]

    # ah that's more like it
    store = pd.HDFStore(filename, "r")

    ret = []
    for group in groups:
        df = store[group + "/dataframe"]
        weight = store[group + "/weights"]

        ret.append(FilledSpectrum(df, weight))

    store.close()

    if len(groups) == 1:
        return ret[0]
    return ret
