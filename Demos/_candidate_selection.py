import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("h5file.log")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('id name event timestamp')
from time import time as now
import h5py

class MonkeyPatchedFile(h5py.File):
    orig_init = h5py.File.__init__
    orig_close = h5py.File.close
    def __init__(self, *args, **kwargs):
        logger.info('{0} {1} open {2}'.format(id(self), args[0], now()))
        MonkeyPatchedFile.orig_init(self, *args, **kwargs)
        # We keep our own copy of the filename, because when __del__ is called, we can no longer
        # call get the filename from the base class -- because the file has already been closed,
        # an no longer has a filename.
        self.monkey_patched_filename = args[0]

    def __del__(self):
        logger.info('{0} {1} finalize {2}'.format(id(self), self.monkey_patched_filename, now()))

    def close(self):
        logger.info('{0} {1} close {2}'.format(id(self), self.monkey_patched_filename, now()))
        MonkeyPatchedFile.orig_close(self)


h5py.File = MonkeyPatchedFile
from context import pandana
from pandana.core import *
from pandana.core.core.loader import Loader
from pandana.core.core.var import Var
from pandana.cut.analysis_cuts import kNumuCutND


def main(input_files, max_files):
    tables = Loader(input_files, limit=max_files)
    energy = Var(lambda tables: tables['rec.slc']['calE'])
    my_spectrum = Spectrum(tables, kNumuCutND, energy)
    logger.info("Before calling Go")
    tables.Go()
    logger.info("After calling Go")

    print('my_spectrum internal dataframe: ')
    print((my_spectrum.df().head()))

    n, _ = my_spectrum.histogram(bins=50, range=(1, 4))
    print('Selected ', n.sum(), ' events from ', my_spectrum.POT(), '  POT.')
