from pandana.core.tables import Tables
from pandana.utils.mpiutils import round_robin
import time
from mpi4py import MPI


class Loader:
    """A class for accessing data in h5py files."""

    def __init__(self, files, idcol, main_table_name, indices):
        self._idcol = idcol
        self._main_table_name = main_table_name
        self._indices = indices

        self._specdefs = []

        # If we have more ranks than files,
        # distribute files among ranks to reduce the number of
        # system calls
        # 64 ranks can read from 4 files by having 16 ranks read each file
        # for a total of 64 system calls.
        # compare to if each rank read a slice of each file ( 64 * 4 = 256 system calls)

        comm = MPI.COMM_WORLD
        if comm.size > len(files):
            self._local_rank_id = comm.rank // len(files)
            self._nranks_per_file, leftover = divmod(comm.size, len(files))
            offset = comm.rank % len(files)
            if offset < self._leftover:
                self._nranks_per_file += 1
            self._files = [files[offset]]

        else:
            self._nranks_per_file = comm.size
            self._local_rank_id = comm.rank
            self._files = files[comm.rank :: comm.size]

        self.rank = comm.rank

    def add_spectrum(self, spec):
        if not spec in self._specdefs:
            self._specdefs.append(spec)

    def Go(self):
        print(f"[{self.rank}] Loader.Go: {time.perf_counter()}")
        """
        Iterate through the associated spectra and compute the cuts and vars for each
        :return: None
        """
        for f in self._files:
            # Construct the tables for this file
            tables = Tables(
                f,
                self._idcol,
                self._main_table_name,
                indices=self._indices,
                nranks_per_file=self._nranks_per_file,
            )

            # FILL ALL SPECTRA for this file
            for spec in self._specdefs:
                spec.fill(tables)

            tables.closeFile()

        self.Finish()

    def Finish(self):
        print(f"[{self.rank}] Loader.Finish: {time.perf_counter()}")
        # Combine together result for each file
        for spec in self._specdefs:
            spec.finish()
