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

    def __init__(self, filename, idcol, main_table_name, indices):
        if type(filename) is not list: filename = [filename]
        self._file = filename
        self._idcol = idcol
        self._main_table_name = main_table_name
        self._indices = indices

        self._specdefs = []

        self._keys = {}

        self.ComputedVars = {}
        self.ComputedCuts = {}

    def add_spectrum(self, spec):
        if not spec in self._specdefs:
            self._specdefs.append(spec)

    def __getitem__(self, key):
        # An h5 file is assumed to be opened and
        # the event ranges already computed
        # from calling Go()

        # If this is the first time this group is being accessed,
        # initialize the group and store it with the table
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
        logger.info(f"main 0 NA startGo {now()}")

        for f in self._file:
            # Open the input file
            self._openfile = h5py.File(f, 'r')

            # Compute the event range for this MPI rank
            comm = MPI.COMM_WORLD
            self._begin_evt, self._end_evt = self.calculateEventRange(
                self._openfile.get(self._main_table_name), comm.rank, comm.size
            )

            logger.info(f"main 0 NA beforefillSpectra {now()}")

            # FILL ALL SPECTRA for this file
            for spec in self._specdefs:
                spec.fill()

            logger.info(f"main 0 NA afterfillSpectra {now()}")

            # EXTERMINATE
            self._openfile.close()
            self.reset()

        # Combine together result for each file
        for spec in self._specdefs:
            spec.finalize()

        logger.info(f"main 0 NA aftercleanup {now()}")

    def reset(self):
        self._keys = {}
        self.ComputedVars = {}
        self.ComputedCuts = {}
