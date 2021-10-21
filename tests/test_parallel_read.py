import argparse
from mpi4py import MPI
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("file_list")
args = parser.parse_args()

from pandana.core.loader import Loader
from pandana.core.spectrum import Spectrum
from pandana.core.var import Var

with open(args.file_list, "r") as f:
    files = [l.strip() for l in f.readlines()]


KL = ["run", "subrun", "cycle", "batch", "evt", "subevt"]
loader = Loader(files, "evt.seq", "spill", indices=KL,)


evt_seq_spec = Spectrum(loader, None, Var(lambda tables: tables["rec.slc"]["evt.seq"]))

loader.Go()

evt_seq = evt_seq_spec.df().reset_index()[KL + ["evt.seq"]]

comm = MPI.COMM_WORLD
evt_seq = comm.gather(evt_seq, root=0)


if comm.rank == 0:
    evt_seq = pd.concat(evt_seq)
    assert evt_seq.shape == evt_seq.drop_duplicates().shape

    # if assertion is passed and shape is the same
    # as when read with one rank, then consider this working
    print(evt_seq.shape)
