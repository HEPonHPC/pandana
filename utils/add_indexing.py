""""This program adds in eid column to each group in a file.

The eid is calculated from the run/subrun/evt numbers in the group.
"""
import h5py as h5
import numpy as np
from math import ceil
from mpi4py import MPI
from vfuncs import (
    eid as make_eid,
    make_evtseq_map,
    apply_evtseq_map,
    make_vector_index,
    ffill_vector_index,
)
import pandas as pd

def fprint(*args, **kwargs):
    myrank = MPI.COMM_WORLD.Get_rank()
    print(
        f'[{myrank}]',
        *args,
        **kwargs,
        flush=True
    )

def calculate_my_slice(dset):
    myrank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    rows_per_rank = ceil(dset.size / size)
    start = rows_per_rank * myrank
    end = min(rows_per_rank * (myrank + 1), dset.size)
    return start, end

def add_neutrino_idx(file):
    event_index_levels = ["run", "subrun", "cycle", "batch", "evt"]
    neutrino_groups = [name for name in file if name.startswith("neutrino")]
    myrank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_rank()

    neutrino_group = file["neutrino"]
    start, stop = calculate_my_slice(neutrino_group["run"])
    length = stop - start
    if myrank == 0:
        start_pad = 0
    else:
        start_pad = 1
    if myrank == world_size - 1:
        stop_pad = 0
    else:
        stop_pad = 1

    padded_start = start - start_pad
    padded_stop = stop + stop_pad
    padded_length = padded_stop - padded_start

    if "neutrino_idx" not in neutrino_group:
        index_data = np.array(
            [
                neutrino_group[idx_name][padded_start:padded_stop].flatten()
                for idx_name in event_index_levels
            ]
        )
        index = pd.MultiIndex.from_arrays(index_data)

        duplicated = index.duplicated().astype(np.short)
        del index
        del index_data

        neutrino_vector_index = make_vector_index(duplicated)

        shape = (neutrino_group["run"].size, 1)
        dset = neutrino_group.create_dataset(
            "neutrino_idx",
            shape=shape,
            shuffle=True,
            compression="gzip",
            compression_opts=6,
            dtype=neutrino_vector_index.dtype,
        )

        with dset.collective:
            dset[start:stop, 0] = neutrino_vector_index[start_pad : start_pad + length]


def add_eid(file):
    """
    Add an 'eid' column to each group in the file.
    :param file: an h5.File object, already open, in update mode
    :return None
    """
    for group_name in file:
        if "geant" in group_name:
            continue
        if "daughter" in group_name:
            continue
        a_group = file[group_name]
        add_eid_to_group(a_group)
        file.flush()


def add_eid_to_group(group):
    """
    Add and 'eid' column to the given group.
    :param group: an h5.Group object
    :return: None
    """

    if "eid" not in group.keys():
        start, stop = calculate_my_slice(group["run"])

        run = group["run"][start:stop].flatten()
        subrun = group["subrun"][start:stop].flatten()
        cycle = group["cycle"][start:stop].flatten()
        batch = group["batch"][start:stop].flatten()
        evt = group["evt"][start:stop].flatten()
        assert run.size == subrun.size
        assert run.size == cycle.size
        assert run.size == batch.size
        assert run.size == evt.size

        gb = run.size * run.dtype.itemsize / 1e9
        fprint(
            f"Processing group {group.name} with column length {run.size} ({gb} GB)",
        )
        myrank = MPI.COMM_WORLD.Get_rank()
        eid_col = make_eid(run, subrun, cycle, batch, evt)
        fprint(f"Made eid_col data ({eid_col.dtype})")
        shape = (group["run"].size, 1)

        # Seems to be a bug in h5py.
        # Attempting to do a collective write with compression
        # results in Segfault with no other error message
        dset = group.create_dataset(
            "eid",
            shape=shape,
            shuffle=True,
            compression="gzip",
            compression_opts=6,
            dtype=eid_col.dtype,
        )
        fprint(f"Created {group.name}/eid")
        with dset.collective:
            dset[start:stop, 0] = eid_col
        fprint(
            f"Wrote {eid_col.size} elements from index {start} to {stop} ({stop - start})"
        )


def add_evtseq(file):
    """Add an 'evtseq' column to each group in the file.

    This column will be in strictly increasing order,
    and will contain a unique value for each distinct run/subrun/evt.
    :param file: an h5.File object, already open, in update mode

    :return: None
    """
    evtseq_map = make_evtseq_map(file["/spill/eid"][:].flatten())
    for group_name in file:
        group = file[group_name]
        if "evt.seq" not in group.keys():
            print("Processing group {0}".format(group_name), flush=True)
            try:
                start, stop = calculate_my_slice(group["eid"])
                eid = group["eid"][start:stop].flatten().astype(np.uint64)
                evtseq_col = apply_evtseq_map(evtseq_map, eid)
                shape = (group["eid"].size, 1)

                # same bug here
                dset = group.create_dataset(
                    "evt.seq",
                    shape=shape,
                    shuffle=True,
                    compression="gzip",
                    compression_opts=6,
                    dtype=evtseq_col.dtype,
                )
                with dset.collective:
                    dset[start:stop, 0] = evtseq_col

                file.flush()
            except KeyError:
                print(f"Skipping {group_name}. Does not contain eid column", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("{0}: you must specify a file to update".format(sys.argv[0]))
        sys.exit(1)
    with h5.File(
        sys.argv[1], "r+", driver="mpio", comm=MPI.COMM_WORLD
    ) as file_to_mutate:
        fprint("Starting to add eid columns")
        add_eid(file_to_mutate)
        
        fprint("Starting to add evtseq columns")
         add_evtseq(file_to_mutate)

        fprint("Starting to add neutrino_idx column")
        add_neutrino_idx(file_to_mutate)
