""""This program adds in eid column to each group in a file.

The eid is calculated from the run/subrun/evt numbers in the group.
"""
import h5py as h5
from .vfuncs import eid as make_eid, make_evtseq_map, apply_evtseq_map

def add_eid(file):
    """
    Add an 'eid' column to each group in the file.
    :param file: an h5.File object, already open, in update mode
    :return None
    """
    for group_name in file:
        group = file[group_name]
        if "eid" not in group.keys():
            run = group["run"][:].flatten()
            print(
                "Processing group {0} with column length {1}".format(
                    group_name, run.size
                )
            )
            subrun = group["subrun"][:].flatten()
            evt = group["evt"][:].flatten()
            eid_col = make_eid(run, subrun, evt)
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
        if "evtseq" not in group.keys():
            print("Processing group {0}".format(group_name))
            eid = group["eid"][:].flatten()
            evtseq_col = apply_evtseq_map(evtseq_map, eid)
            shape = (eid.size, 1)
            group.create_dataset(
                "evtseq",
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
