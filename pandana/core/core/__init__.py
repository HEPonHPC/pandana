import numpy as np
import pandas as pd
import h5py

from pandana.core.core.cut import Cut

from pandana.core.core.dfproxy import dfproxy

# How to index the data
KL = ['run', 'subrun', 'cycle', 'evt', 'subevt']
KLN = ['run', 'subrun', 'cycle', 'evt']
KLS = ['run', 'subrun', 'evt']

class Spectrum():
    def __init__(self, tables, cut, var, weight=None, name=None):
        self._name = name

        # associate this Spectrum, cut with Loader for filling
        tables.add_spectrum(self)
        tables.add_cut(cut)

        # save the var and cut functions so we can call __init__ during fill
        self._varfcn = var
        self._cutfcn = cut
        self._weightfcn = weight

        # keep a reference to Loader for fill
        self._tables = tables

        # compute cut, var, and weights
        # tables is an empty cache of all the necessary branches and leaves initially
        # after tables.Go() the branches and leaves are filled with data from the files given
        self._cut = cut(self._tables)
        self._df = var(self._tables)
        self._df = self._df.dropna()

        # initial weights are all 1
        self._weight = pd.Series(1, self._df.index, name='weight')
        if weight:
            # apply all the weights
            if type(weight) is list:
                for w in weight:
                    self._weight = w(tables, self._weight)
            else: self._weight = weight(tables, self._weight)

    def fill(self):
        # Loader.Go() has been called
        self.__init__(self._tables, self._cutfcn, self._varfcn, weight=self._weightfcn, name=self._name)

        # Just to be sure...
        assert np.array_equal(self._df.index, self._weight.index), 'var and weights have different rows'

        # reset tables global index
        self._tables.reset_index()

        # Set dataframe name if desired
        if self._name: self._df = self._df.rename(self._name)

        # Grab Spectrum POT from tables
        self._POT = self._tables._POT

    def POT(self):
        return self._POT

    def df(self):
        return self._df

    def weight(self):
        return self._weight

    def histogram(self, bins=None, range=None, POT=None):
        if not POT: POT = self._POT
        n, bins = np.histogram(self._df, bins, range, weights = self._weight)
        return n*POT/self._POT, bins

    def entries(self):
        return self._df.shape[0]

    def integral(self,POT=None):
        if not POT: POT = self._POT
        return self._weight.sum()*POT/self._POT

    def to_text(self, fileName, sep=' ', header=False):
        self._df.to_csv(fileName, sep=sep, index=True, header=header)

    def __add__(self, b):
        df = pd.concat([self._df, b._df])
        pot = self._POT + b._POT
        return FilledSpectrum(df, pot)

# For constructing spectra without having to fill
class FilledSpectrum(Spectrum):
    def __init__(self, df, pot, weight=None):
        self._df = df
        self._POT = pot

        if weight is not None: 
            self._weight = weight
        else:
            self._weight = pd.Series(1, self._df.index, name='weight')
        
    def fill(self):
        print('This spectrum was constructed already filled.')

# Save spectra to an hdf5 file. Takes a single or a list of spectra
def save_spectra(fname, spectra, groups):
    if not type(spectra) is list: spectra = [spectra]
    if not type(groups) is list : groups = [groups]
    assert len(spectra)==len(groups), 'Each spectrum must have a group name.'

    # idk why we are giving things to the store
    store = pd.HDFStore(fname, 'w')
    
    for spectrum, group in zip(spectra, groups):
        store[group+'/dataframe'] = spectrum.df()
        store.get_storer(group+'/dataframe').attrs.pot = spectrum.POT()
        store[group+'/weights']   = spectrum.weight()

    store.close()

# alternate save data function that doesn't utilise pytables
def save_tree(fname, spectra, groups, attrs=True):
    if not type(spectra) is list: spectra = [spectra]
    if not type(groups) is list : groups = [groups]
    assert len(spectra)==len(groups), 'Each spectrum must have a group name.'

    f = h5py.File(fname, 'w')
    for spectrum, group in zip(spectra, groups):
        g = f.create_group(group)
        df = spectrum.df()
        vals = df.values
        ismap = 'map' in group
        if ismap:
            for i in range(len(vals)):
                vals[i] = vals[i].reshape(1, vals[i].shape[0])
            vals = np.stack(np.concatenate(vals), axis = 0)

        g.create_dataset('df', data=vals)
        if attrs:
            g.create_dataset('pot', data=spectrum.POT())
            g.create_dataset('weights', data=spectrum.weight())
        index = df.index.names
        indexdf = df.reset_index()
        for name in index:
            g.create_dataset(name, data=indexdf[name].values)

    f.close()


# Load spectra from a file. Takes one or a list of group names to read
def load_spectra(fname, groups):
    if not type(groups) is list: groups = [groups]
    
    # ah that's more like it
    store = pd.HDFStore(fname, 'r')

    ret = []
    for group in groups:
        df = store[group+'/dataframe']
        pot = store.get_storer(group+'/dataframe').attrs.pot
        weight = store[group+'/weights']

        ret.append(FilledSpectrum(df, pot, weight=weight))

    store.close()

    if len(groups) == 1: return ret[0]
    return ret

