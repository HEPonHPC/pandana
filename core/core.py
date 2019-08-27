import numpy as np
import pandas as pd
import h5py
from progbar import ProgressBar
import time
from PandAna.core.filesource import *
import os

# How to index the data
KL = ['run', 'subrun', 'cycle', 'evt', 'subevt']
KLN = ['run', 'subrun', 'cycle', 'evt']
KLS = ['run', 'subrun', 'evt']

class spectrum():
    def __init__(self, tables, cut, var, weight=None, name=None):
        self._name = name

        # associate this spectrum, cut with loader for filling
        tables.add_spectrum(self)
        tables.add_cut(cut)

        # save the var and cut functions so we can call __init__ during fill
        self._varfcn = var
        self._cutfcn = cut
        self._weightfcn = weight

        # keep a reference to loader for fill
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
        # loader.Go() has been called
        self.__init__(self._tables, self._cutfcn, self._varfcn, weight=self._weightfcn, name=self._name)

        # Just to be sure...
        assert np.array_equal(self._df.index, self._weight.index), 'var and weights have different rows'

        # reset tables global index
        self._tables.reset_index()

        # Set dataframe name if desired
        if self._name: self._df = self._df.rename(self._name)

        # Grab spectrum POT from tables
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
        return filledSpectrum(df, pot)

# For constructing spectra without having to fill
class filledSpectrum(spectrum):
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

        ret.append(filledSpectrum(df, pot, weight=weight))

    store.close()

    if len(groups) == 1: return ret[0]
    return ret

class Var():
    def __init__(self, var):
        self._var = var

    def __call__(self, tables):
        return self._var(tables)

    def __eq__(self, val):
        return Cut(lambda tables: self(tables) == val)

    def __ne__(self, val):
        return Cut(lambda tables: self(tables) != val)

    def __lt__(self, val):
        return Cut(lambda tables: self(tables) < val)

    def __le__(self, val):
        return Cut(lambda tables: self(tables) <= val)

    def __gt__(self, val):
        return Cut(lambda tables: self(tables) > val)

    def __ge__(self, val):
        return Cut(lambda tables: self(tables) >= val)
    
    def __add__(self, other):
        return Var(lambda tables: self(tables) + other(tables))
    
    def __sub__(self, other):
        return Var(lambda tables: self(tables) - other(tables))

    def __mult__(self, other):
        return Var(lambda tables: self(tables)*other(tables))
    
    def __truediv__(self, other):
        return Var(lambda tables: self(tables)/other(tables))

class Cut():
    def __init__(self, cut, invert=False):
        if type(cut) is not list: cut = [cut]
        if type(invert) is not list: invert = [invert]
        assert len(cut) == len(invert), "invalid cut definition!" 

        self._cut = list(cut)
        self._invert = list(invert)

        # index that runs over the cutlist 
        self.filteridx = 0

        # use these to keep track of cuts already computed
        self._filter = [0]*len(self._cut)
        self._cutid = [0]*len(self._cut)

    def reset_cutindices(self):
        # need to reset after use by loader
        self._filter = [0]*len(self._cut)
        self._cutid = [0]*len(self._cut)

    def __call__(self, tables):
        # tables is empty anyway. takes negligible time 
        if not tables.gone:
            cutlist = [(~c(tables) if b else c(tables)) for c, b in zip(self._cut, self._invert)]
            #return dummy cut series
            return cutlist[0]

        # cutid holds the filtered index list after applying the cut on the entire dataset
        cutidx = self._cutid[self.filteridx]
        # actual cut that was already computed
        applycut = self._filter[self.filteridx]

        # cut is being computed for the first time
        if cutidx is 0:
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
        else:
            # use filtered index list for evaluation of the var that comes later 
            tables._tables['indices'] = cutidx
            self.filteridx = 0
            self.reset_cutindices()
            return applycut

    def __and__(self, other):
        return Cut(self._cut + other._cut, self._invert + other._invert)

    def __invert__(self):
        return Cut(self._cut, [not b for b in self._invert])

    def __or__(self, other):
        def orcut(tables):
            idx = tables._tables['indices']
            df1 = self(tables)
            tables._tables['indices'] = idx
            df2 = other(tables)
            # or operators are not commutative???
            compare = pd.concat([df1,df2], axis=1, join='outer').fillna(False)
            return compare.any(axis=1)
        return Cut(orcut)


class dfproxy(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ['_proxycols']
    _internal_names_set = set(_internal_names)

    # proxy for a dataframe that builds a cache of columns needed to be read from the files
    # needed before Go() so loader knows what to load
    @property
    def _constructor(self):
        return dfproxy

    def __init__(self, data=[], **kwargs):
        pd.DataFrame.__init__(self, data, **kwargs)
        self._proxycols = list(self.columns.values)

    def __getitem__(self, key):
        # add the column
        if type(key) is str and not key in self._proxycols:
            self._proxycols.append(key)
            self.__setitem__(key, np.nan)
            return self.__getitem__(key)
        # or all the columns
        if type(key) is list and not set(key)<=set(self._proxycols):
            for k in key:
                self._proxycols.append(k)
                self.__setitem__(k, np.nan)
            return self.__getitem__(key)
        # assume dataframe is being sliced inside cut/var, don't do anything
        if type(key) is not str and type(key) is not list:
            return self
        return pd.DataFrame.__getitem__(self, key)

    def __setitem__(self, key, val):
        pd.DataFrame.__setitem__(self, key, val)

class loader():
    def __init__(self, filesource, stride = 1, offset = 0, limit = None, index=None):

        self._files = sourcewrapper(filesource, stride, offset, limit)
        
        # _tables stores the entire dataset read from file
        # index key holds the global index range to be accessed from the dataset by a cut/var
        self._tables = {'indices':0}
        self.gone = False
        self.histdefs = []
        self.cutdefs = []
        self.index=index
        self.dflist = {}

        # add an extra var from spilltree to keep track of exposure
        self.sum_POT()

    def getSource(self):
        return self._files

    def sum_POT(self):
        self._POT = (self['spill']['spillpot']).sum()

    def add_spectrum(self, spec):
        if not spec in self.histdefs:
            self.histdefs.append(spec)

    def add_cut(self, cut):
        if not cut in self.cutdefs:
            self.cutdefs.append(cut)

    def reset_index(self):
        # reset after each spectrum fill
        self._tables['indices'] = 0

    def __setitem__(self, key, df):
        # set multiindex for recTree data
        index = KL if key.startswith('rec') else KLN if key.startswith('neutrino') else KLS
        if self.index and key.startswith('rec'):
          index = self.index
        df.set_index(index, inplace=True)
        self._tables[key] = df

    def __getitem__(self, key):
        # actually build the cache before Go()
        if type(key) == str and not key in self._tables: 
            # Pick up the right index
            index = KL if key.startswith('rec') else KLN if key.startswith('neutrino') else KLS
            if self.index and key.startswith('rec'):
              index = self.index
            self[key] = dfproxy(columns=index)
        # assume key is a filtered index range after a cut 
        if type(key) is not str:
            self._tables['indices'] = key
            return self
        # no filtering 
        if self._tables['indices'] is 0:
            return self._tables[key]
        # use global index to slice dataframe requested
        elif self._tables[key].dropna().empty:
            # sometimes there's no data available in the file, allow it but warn
            print "Warning! No data read for %s" % key
            return self._tables[key]
        else:
            dfslice = self._tables[key].loc[self._tables['indices']]
            return dfslice

    def setupGo(self):
        if self.gone: 
            return
        self.gone = True
        self._filegen = self._files()

        print("Reading data from %s files : \n" % self._filegen.nFiles())

    def getFile(self):
        return self._filegen()

    def setFile(self, f):
        self.openfile = f
    
    def closeFile(self):
        self.openfile.close()

    def readData(self):
        for key in self._tables:
            if key is 'indices':
                continue
            if not key in self.dflist:
                self.dflist[key] = []
            # branches from cache
            group = self.openfile.get(key)
            values = {}
            # leaves from cache
            keycache = self._tables[key]._proxycols
            for k in keycache:
                dataset = group.get(k)[()]
                if dataset.shape[1] == 1:
                    dataset = dataset.flatten()
                else:
                    dataset = list(dataset)
                values[k] = dataset
            self.dflist[key].append(pd.DataFrame(values))

    def fillSpectra(self):
        for key in self.dflist:
            # set index for all dataframes
            self[key] = pd.concat(self.dflist[key])
        self.dflist = {}
        # Compute POT and then fill spectra
        self.sum_POT()
        
        spec_idx = 0
        spec_progbar = ProgressBar(len(self.histdefs))
        print("Filling %s spectra\n" % len(self.histdefs))
        for spec in self.histdefs:
            spec_idx += 1
            spec_progbar.Update(spec_idx)
            spec.fill()
        
    def Go(self):
        t0 = time.time()
        self.setupGo()
        file_idx = 0
        file_progbar = ProgressBar(self._filegen.nFiles())
        while True:
          try:
            fname = self.getFile()
            self.setFile(h5py.File(fname, 'r'))
            self.readData()
            self.closeFile()
       
            file_idx += 1
            file_progbar.Update(file_idx)
          except StopIteration:
            break
        
        self.fillSpectra()
        # cleanup
        self.cleanup()
        print("\nTotal time : %s sec\n" % (time.time() - t0))

    def cleanup(self):
        # free up some memory
        self._tables = {'indices':0}
        # remove associations with spectra
        self.histdefs = []

# Different loaders end up starting their own SAM projects, even for the exact same queries. 
# This doesn't guarantee that they'll run over the files in the same order. 
# Coupled with the fact that the projects can be shared over different grid jobs, 
# this can result in unexpected behaviour if the macro expects them to share the same data downstream. 
# This class allows the user to use a single project over multiple loaders 
class associate(loader):
    def __init__(self, loaders):
        self.loaders = loaders
        assert len(self.loaders) > 0, "Can't associate empty list of loaders!"
        # ensure all loaders have the same file source ID 
        assert all([self.loaders[0].getSource() == l.getSource() for l in self.loaders]), \
            "Can only associate loaders with unique queries"
        
        self._files = self.loaders[0].getSource()
        self.gone = False

    def Go(self):
        t0 = time.time()
        self.setupGo()
        print("Associating %d loaders with the same dataset : \n" % len(self.loaders))
        file_idx = 0
        file_progbar = ProgressBar(self._filegen.nFiles())
        while True:
          try:
            fname = self.getFile()
            self.setFile(h5py.File(fname, 'r'))
            for ldr in self.loaders:
              ldr.gone = True
              ldr.setFile(self.openfile)
              ldr.readData()
            self.closeFile()
            file_idx += 1
            file_progbar.Update(file_idx)
          except StopIteration:
            break
        ldr_idx = 1
        for ldr in self.loaders:
          print ("\n------------------------------")
          print ("Filling spectra for loader %d" % ldr_idx)
          ldr.fillSpectra()
          ldr.cleanup()
          ldr_idx += 1
        print("\nTotal time : %s sec\n" % (time.time() - t0))

class interactive_loader():
    def __init__(self, files):
        self._files = files
        self._tables = {}

    def keys(self, contain=None):
        f = self._files[0]
        h5 = h5py.File(f, 'r')
        for k in h5.keys():
            if contain:
                if contain in k:
                    print(k)
            else:
                print(k)
        h5.close()

    def __getitem__(self, key):
        if not key in self._tables:
            dflist=[]
            for fname in self._files:
                f = h5py.File(fname,'r')
                group = f.get(key)
                values = {}
                for k in group.keys():
                    dataset = group.get(k).value
                    if dataset.shape[1] == 1:
                        dataset = dataset.flatten()
                    else:
                        dataset = list(dataset)
                    values[k] = dataset
                dflist.append(pd.DataFrame(values))
                f.close()
            df = pd.concat(dflist)
            if not (key.startswith('spill') or key.startswith('neutrino')):
                df.set_index(KL, inplace=True)
            self._tables[key] = df
        return self._tables[key]
