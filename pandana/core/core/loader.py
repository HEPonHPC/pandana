import collections
import time
import h5py
import pandas as pd
from mpi4py import MPI

from pandana import SourceWrapper, KL, KLN, KLS, DFProxy

class Loader():

    def __init__(self, filesource, stride = 1, offset = 0, limit = None, index=None):
        self._files = SourceWrapper(filesource, stride, offset, limit)
        # _tables stores the entire dataset read from file
        # index key holds the global index range to be accessed from the dataset by a cut/var
        self._tables = {'indices': 0}
        self.gone = False
        self.histdefs = []
        self.cutdefs = []
        self.index=index
        self.dflist = collections.defaultdict(list)
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
        # reset after each Spectrum fill
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
            self[key] = DFProxy(columns=index)
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
            print("Warning! No data read for %s" % key)
            return self._tables[key]
        else:
            dfslice = self._tables[key].loc[self._tables['indices']]
            return dfslice

    def setupGo(self):
        """
        Call the SourceWrapper for the file to be processed.
        :return: None
        """
        if self.gone:
            return
        self.gone = True
        self._filegen = self._files()

        print(("Reading data from %s files : \n" % self._filegen.nFiles()))

    def getFile(self):
        return self._filegen()

    def setFile(self, f):
        self.openfile = f

    def closeFile(self):
        self.openfile.close()

    def createDataFrames(self):
        '''
        Create the DataFrames corresponding to the current open file.
        This populates self.dflist.
        :return: None
        '''
        for tablename in self._tables:
            if tablename is 'indices':
                continue
            self.createDataFrame(tablename)

    def createDataFrame(self, tablename):
        # branches from cache
        group = self.openfile.get(tablename)
        # leaves from cache
        values = {k: self._readDataset(group, k) for k in self._tables[tablename]._proxycols}
        self.dflist[tablename].append(pd.DataFrame(values))

    def _readDataset(self, group, datasetname):
        # Determine range to be read here.
        # Regardless of the dataset, we want to read all the entries corresponding to the range of events
        # (not runs, subruns, or subevents, but events) we are to process.
        # dataset is a numpy.array, not a h5py.Dataset.
        ds = group.get(datasetname)  # ds is a h5py.Dataset
        dataset = ds[()]  # read the whole dataset. Fails if it is non-scalar.
        if dataset.shape[1] == 1:
            dataset = dataset.flatten()
        else:
            dataset = list(dataset)  # Can this ever be called? What would the global result be?
        return dataset

    def fillSpectra(self):
        for key in self.dflist:
            # set index for all dataframes
            self[key] = pd.concat(self.dflist[key])
        self.dflist = {}
        # Compute POT and then fill spectra
        self.sum_POT()

        print(("Filling %s spectra\n" % len(self.histdefs)))
        for spec in self.histdefs:
            spec.fill()

    def Go(self):
        '''
        Iterate through all associated files, reading all required data. When done, fill all specified spectra.
        :return: None
        '''
        # TODO: Consider accumulating the results into spectra file-by-file, rather than reading all files before
        # filling any spectra.
        t0 = time.time()

        self.setupGo()
        file_idx = 0
        while True:
          try:
            fname = self.getFile()
            self.setFile(h5py.File(fname, 'r'))
            self.createDataFrames()
            self.closeFile()

            file_idx += 1
          except StopIteration:
            break

        self.fillSpectra()
        # cleanup
        self.cleanup()
        print(("\nTotal time : %s sec\n" % (time.time() - t0)))

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


class AssociateLoader(Loader):
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
        print(("Associating %d loaders with the same dataset : \n" % len(self.loaders)))
        file_idx = 0
        while True:
          try:
            fname = self.getFile()
            self.setFile(h5py.File(fname, 'r'))
            for ldr in self.loaders:
              ldr.gone = True
              ldr.setFile(self.openfile)
              ldr.createDataFrames()
            self.closeFile()
            file_idx += 1
          except StopIteration:
            break
        ldr_idx = 1
        for ldr in self.loaders:
          print ("\n------------------------------")
          print(("Filling spectra for Loader %d" % ldr_idx))
          ldr.fillSpectra()
          ldr.cleanup()
          ldr_idx += 1
        print(("\nTotal time : %s sec\n" % (time.time() - t0)))


class InteractiveLoader():
    def __init__(self, files):
        self._files = files
        self._tables = {}

    def keys(self, contain=None):
        f = self._files[0]
        h5 = h5py.File(f, 'r')
        for k in list(h5.keys()):
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
                for k in list(group.keys()):
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