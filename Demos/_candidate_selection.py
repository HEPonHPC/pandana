import logging

from mpi4py import MPI

logger = logging.getLogger(__name__)
handler = logging.FileHandler(f"cand_sel_{MPI.COMM_WORLD.rank}.log")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
#logger.info('kind id name event timestamp')
logger.info('dset\tfirst\tlast')

from time import time as now
import h5py
import numpy as np

class MonkeyPatchedFile(h5py._hl.files.File):
    orig_init = h5py._hl.files.File.__init__
    orig_close = h5py._hl.files.File.close
    def __init__(self, *args, **kwargs):
        #logger.info('file {0} {1} open {2}'.format(id(self), args[0], now()))
        MonkeyPatchedFile.orig_init(self, *args, **kwargs)
        # We keep our own copy of the filename, because when __del__ is called, we can no longer
        # call get the filename from the base class -- because the file has already been closed,
        # an no longer has a filename.
        self.monkey_patched_filename = args[0]

    def __del__(self):
        #logger.info('file {0} {1} finalize {2}'.format(id(self), self.monkey_patched_filename, now()))
        pass

    def close(self):
        #logger.info('file {0} {1} close {2}'.format(id(self), self.monkey_patched_filename, now()))
        MonkeyPatchedFile.orig_close(self)
h5py._hl.files.File = MonkeyPatchedFile
h5py.File = MonkeyPatchedFile

class MonkeyPatchedDataset(h5py.Dataset):
    orig_getitem = h5py._hl.dataset.Dataset.__getitem__
    def __getitem__(self, *args):
        logger.info(f"{self.name}\t{args}")
        #logger.info('dset {0} {1} startread {2}'.format(id(self), self.name, now()))
        res = MonkeyPatchedDataset.orig_getitem(self, *args)
        #logger.info('dset {0} {1} endread {2}'.format(id(self), self.name, now()))
        return res

h5py._hl.dataset.Dataset = MonkeyPatchedDataset
h5py.Dataset = MonkeyPatchedDataset

from context import pandana
from pandana.core import *
from pandana.core.loader import Loader
from pandana.core.var import Var
from nova.cut.analysis_cuts import kNumuCutND
from mpi4py import MPI

def main(input_files, idcol, max_files):
    tables = Loader(input_files, idcol, limit=max_files)
    energy = Var(lambda tables: tables['rec.slc']['calE'])
    my_spectrum = Spectrum(tables, kNumuCutND, energy)
    tables.Go()

    #print('my_spectrum internal dataframe: ')
    #print((my_spectrum.df().head()))

    nbins = 50
    n, _ = my_spectrum.histogram(bins=nbins, range=(1, 4))
    total_n = MPI.COMM_WORLD.reduce(n, op=MPI.SUM, root = 0)
    total_pot = MPI.COMM_WORLD.reduce(my_spectrum.POT(), op=MPI.SUM, root = 0)
    if MPI.COMM_WORLD.rank == 0:
        print('Selected ', total_n.sum(), ' events from ', total_pot, '  POT.')
