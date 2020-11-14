import logging

from mpi4py import MPI

logger = logging.getLogger("timing")
handler = logging.FileHandler(f"cand_sel_{MPI.COMM_WORLD.rank}.log", mode="w")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('kind id name event timestamp')

import numpy as np
from time import time as now

from context import pandana
from pandana.core import *
from pandana.core.loader import Loader
from pandana.core.var import Var
from nova.cut.analysis_cuts import kNumuCutND
from mpi4py import MPI


def main(input_files, idcol):
    logger.info(f'main 0 NA start {now()}')
    tables = Loader(input_files, idcol, "spill", logger=logger)
    logger.info(f'main 0 NA afterLoader {now()}')
    energy = Var(lambda tables: tables["rec.slc"]["calE"])
    logger.info(f'main 0 NA afterVar {now()}')
    my_spectrum = Spectrum(tables, kNumuCutND, energy) # all the Loader.__getitem__ calls happen here
    logger.info(f'main 0 NA afterSpectrum {now()}')
    tables.Go()
    logger.info(f'main 0 NA afterGo {now()}')

    # print('my_spectrum internal dataframe: ')
    # print((my_spectrum.df().head()))

    nbins = 50
    logger.info(f'main 0 NA beforehist {now()}')
    n, _ = my_spectrum.histogram(bins=nbins, range=(1, 4))
    logger.info(f'main 0 NA afterhist {now()}')
    total_n = MPI.COMM_WORLD.reduce(n, op=MPI.SUM, root=0)
    total_pot = MPI.COMM_WORLD.reduce(my_spectrum.POT(), op=MPI.SUM, root=0)
    logger.info(f'main 0 NA afterreduce {now()}')
    if MPI.COMM_WORLD.rank == 0:
        print("Selected ", total_n.sum(), " events from ", total_pot, "  POT.")
