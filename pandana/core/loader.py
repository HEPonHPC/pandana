import collections
import time
import h5py
import pandas as pd
from mpi4py import MPI

from pandana.core import SourceWrapper, DFProxy
from pandana.core.indices import KL, KLN, KLS
from pandana import utils
from time import time as now


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
        dataset = list(
            dataset
        )  # Can this ever be called? What would the global result be?
    return dataset


def createDataFrameFromFile(
    current_file, tablename, proxycols, begin_evt, end_evt, idcol
):
    # branches from cache
    group = current_file.get(tablename)
    event_seq_numbers = group[idcol][()].flatten()
    begin_row, end_row = event_seq_numbers.searchsorted([begin_evt, end_evt + 1])
    # leaves from cache
    values = {k: readDatasetFromGroup(group, k, begin_row, end_row) for k in proxycols}
    return pd.DataFrame(values)


class Loader:
    def __init__(
        self, filesource, idcol, main_table_name, stride=1, offset=0, limit=None, index=None, logger=None
    ):
        self.idcol = idcol
        self.main_table_name = main_table_name
        self._files = SourceWrapper(filesource, stride, offset, limit)
        self.interactive = False
        # _tables stores the entire dataset read from file
        # index key holds the global index range to be accessed from the dataset by a cut/var
        self._tables = {}
        self._indices = None
        self.gone = False
        self.histdefs = []
        self.cutdefs = []
        self.index = index
        self.logger = logger
        self.dflist = collections.defaultdict(list)
        # add an extra var from spilltree to keep track of exposure
        self._POT = None
        # TODO: This call appears redundant, but it is not. Why not?
        # self.sum_POT() will be called again later, but if this call is removed
        # we fail to collect any events passing cuts.
        self.sum_POT()

    def getSource(self):
        return self._files

    def sum_POT(self):
        if self.main_table_name == "spill":
            self._POT = (self[self.main_table_name]["spillpot"]).sum()

    def add_spectrum(self, spec):
        if not spec in self.histdefs:
            self.histdefs.append(spec)

    def add_cut(self, cut):
        if not cut in self.cutdefs:
            self.cutdefs.append(cut)

    def reset_index(self):
        # TODO: This does not reset self.index; it resets self._indices.
        # Why is this?
        #
        # reset after each Spectrum fill
        self._indices = None

    def __setitem__(self, key, df):
        # TODO: This is very NOvA-specific. What is the generalization of this?
        # set multiindex for recTree data
        index = (
            KL if key.startswith("rec") else KLN if key.startswith("neutrino") else KLS
        )
        # TODO: What does the Loader having an index imply? Give a test with a meaningful name.
        if self.index and key.startswith("rec"):
            index = self.index
        df.set_index(index, inplace=True)
        self._tables[key] = df

    def __getitem__(self, key):
        # TODO: What can __getitem__ return?
        #    if key is not a string, it *sets* a value and returns self.
        #    it can return an entry from self._tables
        #    it can return a slice of an etry from self._tables
        # actually build the cache before Go()
        # TODO: Clarify this: is the behavior of __getitem__ different before Go() is called and after it is called?
        if type(key) == str and not key in self._tables:
            self.set_proxy_for_key(key)
        # assume key is a filtered index range after a cut
        if type(key) is not str:
            self._indices = key
            return self
        # no filtering
        if self._indices is None:
            return self._tables[key]
        # use global index to slice dataframe requested
        elif self._tables[key].dropna().empty:
            # sometimes there's no data available in the file, allow it but warn
            # Avoid printing messages in MPI programs.
            if MPI.COMM_WORLD.size == 1:
                print("Warning! No data read for %s" % key)
            return self._tables[key].dropna()
        else:
            dfslice = self._tables[key].loc[
                self._tables[key].index.intersection(self._indices)
            ]
            return dfslice

    def set_proxy_for_key(self, key):
        # Pick up the right index
        # TODO: This is experiment-specific. Remove this to some facility that can be provided to the Loader. How should
        # this functionality be provided by different experiments? Is the Loader the right thing to be managing such an
        # index?
        index = (
            KL if key.startswith("rec") else KLN if key.startswith("neutrino") else KLS
        )

        # TODO: What does is mean for self.index to be falsy here? A well-named predicate would be better. Does it mean
        # that some other function has not yet been called? Which function? The key (table name) starting with 'rec'
        # is experiment-specific. Is the relevant feature that key is the name of a table that has one entry per
        # primary atomic unit of processing (for the StandardRecord format, the slice/subevent)? Note the related test
        # above, which also sets the value of index to a special value under the same condition.
        if self.index and key.startswith("rec"):
            # If we already have an index, this will just reset the value determined above.
            index = self.index

        # Note we are already in a call to Loader.__getitem__; we are going to now call Loader.__setitem__.
        self[key] = DFProxy(columns=index)

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
        assert(group is not None)
        begin, end = utils.mpiutils.calculate_slice_for_rank(
            rank, nranks, group[self.idcol].size
        )
        span = group[self.idcol][begin:end].flatten()
        b, e = span[0], span[-1]
        return b, e

    def createDataFrames(self, a_file):
        """
        Create the DataFrames corresponding to aFile, which must be open.
        This populates self.dflist.
        :return: None
        """
        comm = MPI.COMM_WORLD
        begin_evt, end_evt = self.calculateEventRange(
            a_file.get(self.main_table_name), comm.rank, comm.size
        )
        for tablename in self._tables:
            # TODO: Loader should not need to access a protected member of DFProxy.
            new_df = createDataFrameFromFile(
                a_file,
                tablename,
                self._tables[tablename]._proxycols,
                begin_evt,
                end_evt,
                self.idcol,
            )
            self.dflist[tablename].append(new_df)

    def fillSpectra(self):
        for key in self.dflist:
            # set index for all dataframes
            assert isinstance(self[key], DFProxy)
            self[key] = pd.concat(self.dflist[key])
            assert isinstance(self[key], pd.DataFrame)
        self.dflist = {}
        # Compute POT and then fill spectra
        self.sum_POT()

        for spec in self.histdefs:
            spec.fill()

    def Go(self):
        """
        Iterate through all associated files, reading all required data. When done, fill all specified spectra.
        :return: None
        """
        # TODO: Consider accumulating the results into spectra file-by-file, rather than reading all files before
        # filling any spectra.
        t0 = time.time()
        self.setupGo()
        if self.logger is not None:
            self.logger.info(f"main 0 NA aftersetupGo {now()}")
        file_idx = 0
        while True:
            try:
                fname = self.getFile()
                with h5py.File(fname, "r") as current_file:
                    self.createDataFrames(current_file)
                file_idx += 1
            except StopIteration:
                break

        if self.logger is not None:
            self.logger.info(f"main 0 NA beforefillSpectra {now()}")
        self.fillSpectra()
        if self.logger is not None:
            self.logger.info(f"main 0 NA afterfillSpectra {now()}")
        # cleanup
        self.cleanup()
        if self.logger is not None:
            self.logger.info(f"main 0 NA aftercleanup {now()}")

    def cleanup(self):
        # free up some memory
        self._indices = None
        # remove associations with spectra
        self.histdefs = []
