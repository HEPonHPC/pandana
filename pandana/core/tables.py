import h5py
from mpi4py import MPI

from pandana import utils
from pandana.core.datagroup import DataGroup


class Tables:
    def __init__(self, f, idcol, main_table_name, indices):
        self._file = h5py.File(f, "r")
        self._idcol = idcol
        self._main_table_name = main_table_name
        self._indices = indices

        # Compute the event range for this MPI rank
        # Default to all the data
        self._begin_evt, self._end_evt = None, None
        comm = MPI.COMM_WORLD
        if comm.size > 1:
            self._begin_evt, self._end_evt = self.calculateEventRange(
                self._file.get(self._main_table_name), comm.rank, comm.size
            )

        self._keys = {}

    def __getitem__(self, key):
        # An h5 file is assumed to be opened and
        # the event ranges already computed
        # from calling Go()

        # If this is the first time this group is being accessed,
        # initialize the group and store it with the table
        if not key in self._keys:
            self._keys[key] = DataGroup(
                self._file,
                key,
                self._begin_evt,
                self._end_evt,
                self._idcol,
                self._indices,
            )
        return self._keys[key]

    def calculateEventRange(self, group, rank, nranks):
        assert group is not None
        begin, end = utils.mpiutils.calculate_slice_for_rank(
            rank, nranks, group[self._idcol].size
        )
        span = group[self._idcol][begin:end].flatten()
        b, e = span[0], span[-1]

        return b, e

    def closeFile(self):
        self._file.close()
