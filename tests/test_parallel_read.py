import argparse
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('file_list')
args = parser.parse_args()

from pandana.core.loader import Loader
from pandana.core.tables import Tables
from pandana.core.datagroup import DataGroup

with open(args.file_list, 'r') as f:
    files = [l.strip() for l in f.readlines()]

loader = Loader(files, 'evt.seq', 'spill', [])

comm = MPI.COMM_WORLD


print(comm.rank, loader._local_rank_id, loader._nranks_per_file, loader._files)



