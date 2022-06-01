""""This program adds in eid column to each group in a file.

The eid is calculated from the run/subrun/evt numbers in the group.
"""
import h5py as h5
from vfuncs import (
    eid as make_eid,
    make_evtseq_map,
    apply_evtseq_map,
    make_vector_index,
    ffill_vector_index,
)
import numpy as np
import pandas as pd

def find_level(dset_name):
    level_has_idx = np.array([level.endswith('_idx') for level in str(dset_name).split('.')], dtype=int)
    if level_has_idx.any():
        return np.nonzero(level_has_idx)[0][0]
    else:
        return np.inf
def find_lowest_level_with_idx(group):
    dset_names = list(group.keys())
    idx_level = [find_level(dset_name) for dset_name in dset_names]
    return dset_names[np.argmin(idx_level)]

def add_neutrino_vector_index(file):
    """Add neutrino_idx Dataset to {group_name} if 
    group_name starts with neutrino, ie a nuTree branch
    """
    event_index_levels = ['run', 'subrun', 'cycle', 'batch', 'evt']
    neutrino_groups = [name for name in file if name.startswith("neutrino")]
    # groups representing vectors or branching off from vectors
    # have an additional _idx dataset 
    # and potentially many rows for each neutrino_idx
    # so we have to calculate the new dataset differently
    vector_groups = [
        np.array([
            dset.endswith('_idx')
            for dset in list(file[g].keys())
            if dset != 'neutrino_idx'
        ]).any() 
        for g in neutrino_groups
    ]
    for neutrino_group, isvector in zip(neutrino_groups, vector_groups):
        if not isvector:
            if 'neutrino_idx' not in file[neutrino_group].keys():
                index_data = np.array([file[neutrino_group][idx_name][:].flatten() for idx_name in event_index_levels])
                index = pd.MultiIndex.from_arrays(index_data)
                neutrino_vector_index = make_vector_index(index.duplicated().astype(int))
                shape = (index.shape[0],1)
                file[neutrino_group].create_dataset(
                    "neutrino_idx",
                    data=neutrino_vector_index,
                    shape=shape,
                    shuffle=True,
                    compression="gzip",
                    compression_opts=6,
                )
            else:
                print(f'Skipping {neutrino_group}')
        else:
            if 'neutrino_idx' not in file[neutrino_group].keys():
                index_data = np.array([file[neutrino_group][idx_name][:].flatten() for idx_name in event_index_levels])
                index = pd.MultiIndex.from_arrays(index_data)
                vector_index = file[neutrino_group][find_lowest_level_with_idx(file[neutrino_group])][:].flatten()
                reference = file['neutrino']['neutrino_idx'][:].flatten()
                neutrino_vector_index = ffill_vector_index(reference.astype(int),
                                                           vector_index.astype(int))
                shape = (index.shape[0],1)
                file[neutrino_group].create_dataset(
                    "neutrino_idx",
                    data=neutrino_vector_index,
                    shape=shape,
                    shuffle=True,
                    compression="gzip",
                    compression_opts=6,
                )
            else:
                print(f'Skipping {neutrino_group}')
            

def add_eid(file):
    """
    Add an 'eid' column to each group in the file.
    :param file: an h5.File object, already open, in update mode
    :return None
    """
    for group_name in file:
        a_group = file[group_name]
        add_eid_to_group(a_group)


def add_eid_to_group(group):
    """
    Add and 'eid' column to the given group.
    :param group: an h5.Group object
    :return: None
    """
    if "eid" not in group.keys():
        run = group["run"][:].flatten()
        subrun = group["subrun"][:].flatten()
        cycle = group["cycle"][:].flatten()
        batch = group["batch"][:].flatten()
        evt = group["evt"][:].flatten()
        print(f"Processing group {group.name} with column length {run.size}")
        eid_col = make_eid(run, subrun, cycle, batch, evt)
        shape = (run.size, 1)
        group.create_dataset(
            "eid",
            data=eid_col,
            shape=shape,
            shuffle=True,
            compression="gzip",
            compression_opts=6,
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
            print("Processing group {0}".format(group_name))
            eid = group["eid"][:].flatten()
            evtseq_col = apply_evtseq_map(evtseq_map, eid)
            shape = (eid.size, 1)
            group.create_dataset(
                "evt.seq",
                data=evtseq_col,
                shape=shape,
                shuffle=True,
                compression="gzip",
                compression_opts=6,
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("{0}: you must specify a file to update".format(sys.argv[0]))
        sys.exit(1)
    with h5.File(sys.argv[1], "r+") as file_to_mutate:
        print("Starting to add eid columns")
        add_eid(file_to_mutate)
        print("Starting to add evtseq columns")
        add_evtseq(file_to_mutate)
        print("Starting to add neutrino vector index")
        add_neutrino_vector_index(file_to_mutate)
        
