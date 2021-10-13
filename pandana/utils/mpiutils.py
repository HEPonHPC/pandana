"""This module provides MPI utility functions.
"""


def calculate_slice_for_rank(myrank, nranks_per_file, arraysz):
    """Calculate the slice indices for array processing in MPI programs.

    Return (low, high), a tuple containing the range of indices
    in an array of size arraysz, to be processed by MPI rank myrank of
    a total nranks. We assure as equitable a distribution of ranks as
    possible.
    """

    local_rank_id = myrank % nranks_per_file
    if nranks > arraysz:
        raise ValueError("nranks must not be larger than array size")

    # Each rank will get either minsize or minsize+1 elements to work on.
    minsize, leftovers = divmod(arraysz, nranks_per_file)

    # Ranks [0, leftovers) get minsize+1 elements
    # Ranks [leftovers, nranks) get minsize elements
    slice_size = minsize + 1 if local_rank_id < leftovers else minsize

    if myrank < leftovers:
        low = local_rank_id * slice_size
        high = low + slice_size
    else:
        # The calculation of 'low' is the algebraically simplified version of
        # the more obvious:
        #  low = leftovers*(my_size_size_bytes + 1) + (myrank - leftovers)*my_size_size_bytes
        low = leftovers + local_rank_id * slice_size
        high = low + slice_size
    return low, high
