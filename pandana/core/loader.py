import h5py
import numpy as np
from mpi4py import MPI

from time import time as now
import logging
logger = logging.getLogger("timing")

from pandana import utils
from pandana.core.datagroup import DataGroup

class Loader:
    """A class for accessing data in h5py files."""

    def __init__(self, file, idcol, main_table_name, indices):
        self._file = file
        self._idcol = idcol
        self._main_table_name = main_table_name
        self._indices = indices

        self._specdefs = []

        self._keys = {}

        self.ComputedVars = {}
        self.ComputedCuts = {}

        self._gone = False

    def add_spectrum(self, spec):
        if not spec in self._specdefs:
            self._specdefs.append(spec)

    def __getitem__(self, key):
        # An h5 file is assumed to be opened and
        # the event ranges already computed
        if not self._gone:
            return
        # If this is the first time this group is being accessed,
        # read in the group and store it with the table
        if not key in self._keys:
            self._keys[key] = DataGroup(self._openfile, key,
                                        self._begin_evt, self._end_evt,
                                        self._idcol, self._indices)
        return self._keys[key]

    def calculateEventRange(self, group, rank, nranks):
        assert group is not None
        begin, end = utils.mpiutils.calculate_slice_for_rank(
            rank, nranks, group[self._idcol].size
        )
        span = group[self._idcol][begin:end].flatten()
        b, e = span[0], span[-1]

        return b, e

    def Go(self):
        """
        Iterate through the associated spectra and compute the cuts and vars for each
        :return: None
        """
        if self._gone:
            return
        self._gone = True

        logger.info(f"main 0 NA startGo {now()}")

        # Open the input file
        self._openfile = h5py.File(self._file, 'r')

        # Compute the event range for this MPI rank
        comm = MPI.COMM_WORLD
        self._begin_evt, self._end_evt = self.calculateEventRange(
            self._openfile.get(self._main_table_name), comm.rank, comm.size
        )

        logger.info(f"main 0 NA beforefillSpectra {now()}")

        # FILL ALL SPECTRA
        for spec in self._specdefs:
            spec.fill()

        logger.info(f"main 0 NA afterfillSpectra {now()}")

        # EXTERMINATE
        self._openfile.close()
        self.cleanup()

        logger.info(f"main 0 NA aftercleanup {now()}")

    def cleanup(self):
        self._specdefs = None
        self.ComputedVars = None
        self.ComputedCuts = None
