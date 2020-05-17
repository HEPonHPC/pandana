import collections
import time
import h5py
import pandas as pd
from mpi4py import MPI

from pandana import SourceWrapper, DFProxy
from pandana.core.indices import KL, KLN, KLS
from pandana import utils


def readDatasetFromGroup(group, datasetname, begin, end):
    # Determine range to be read here.
    # Regardless of the dataset, we want to read all the entries corresponding to the range of events
    # (not runs, subruns, or subevents, but events) we are to process.
    # dataset is a numpy.array, not a h5py.Dataset.
    ds = group.get(datasetname)  # ds is a h5py.Dataset
    # dataset = ds[()]  # read the whole dataset. Fails if it is non-scalar.
    dataset = ds[begin:end]
    if dataset.shape[1] == 1:
        dataset = dataset.flatten()
    else:
        dataset = list(dataset)  # Can this ever be called? What would the global result be?
    return dataset

def createDataFrameFromFile(current_file, tablename, proxycols, begin_evt, end_evt, idcol):
    # branches from cache
    group = current_file.get(tablename)
    event_seq_numbers = group[idcol][()].flatten()
    begin_row, end_row = event_seq_numbers.searchsorted([begin_evt, end_evt+1])
    # leaves from cache
    values = {k: readDatasetFromGroup(group, k, begin_row, end_row) for k in proxycols}
    return pd.DataFrame(values)

class Loader():

    def __init__(self, filesource, idcol, stride = 1, offset = 0, limit = None, index=None):
        self.idcol = idcol
        self._files = SourceWrapper(filesource, stride, offset, limit)
        # _tables stores the entire dataset read from file
        # index key holds the global index range to be accessed from the dataset by a cut/var
        self._tables = {'indices': None}
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
        self._tables['indices'] = None

    def __setitem__(self, key, df):
        # set multiindex for recTree data
        index = KL if key.startswith('rec') else KLN if key.startswith('neutrino') else KLS
        if self.index and key.startswith('rec'):
          index = self.index
        df.set_index(index, inplace=True)
        self._tables[key] = df

    def __getitem__(self, key):
        # actually build the cache before Go()
        # TODO: Clarify this: is the behavior of __getitem__ different before Go() is called and after it is called?
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
        if self._tables['indices'] is None:
            return self._tables[key]
        # use global index to slice dataframe requested
        elif self._tables[key].dropna().empty:
            # sometimes there's no data available in the file, allow it but warn
            print("Warning! No data read for %s" % key)
            return self._tables[key]
        else:
            dfslice = self._tables[key].loc[self._tables[key].index.intersection(self._tables['indices'])]
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

    def getFile(self):
        return self._filegen()

    def calculateEventRange(self, group, rank, nranks):
        begin, end = utils.mpiutils.calculate_slice_for_rank(rank, nranks, group[self.idcol].size)
        span = group[self.idcol][begin : end].flatten()
        b, e = span[0], span[-1]
        return b, e

    def createDataFrames(self, aFile):
        '''
        Create the DataFrames corresponding to aFile, which must be open.
        This populates self.dflist.
        :return: None
        '''
        comm = MPI.COMM_WORLD
        begin_evt, end_evt = self.calculateEventRange(aFile.get('spill'), comm.rank, comm.size)
        for tablename in self._tables:
            # TODO: It seems like self._tables['indices'] is sufficiently different from all other
            # values stored that it should be a member of Loader directly.
            if tablename == 'indices':
                continue
            # TODO: Loader should not need to access a protected member of DFProxy.
            new_df = createDataFrameFromFile(aFile, tablename, self._tables[tablename]._proxycols, begin_evt, end_evt, self.idcol)
            self.dflist[tablename].append(new_df)

    def fillSpectra(self):
        for key in self.dflist:
            # set index for all dataframes
            assert(isinstance(self[key], DFProxy))
            self[key] = pd.concat(self.dflist[key])
            assert(isinstance(self[key], pd.DataFrame))
        self.dflist = {}
        # Compute POT and then fill spectra
        self.sum_POT()

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
            with h5py.File(fname, 'r') as current_file:
                self.createDataFrames(current_file)
            file_idx += 1
          except StopIteration:
            break

        self.fillSpectra()
        # cleanup
        self.cleanup()

    def cleanup(self):
        # free up some memory
        self._tables = {'indices':0}
        # remove associations with spectra
        self.histdefs = []


