import collections
import time
import h5py
import pandas as pd
from mpi4py import MPI

from pandana.core.loader import Loader
from pandana import utils


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
          ldr.fillSpectra()
          ldr.cleanup()
          ldr_idx += 1
