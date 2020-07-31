"""This module provides some numba-accelerated utility functions.

"""
import numba
import numpy as np


@numba.vectorize(["uint64(uint32,uint32,uint32)"])
def eid(run, subrun, evt):
    """Calculate a single-number event id from run/subrun/event.

    :param run: numpy array of uint32 run numbers
    :param subrun: numpy array of uint32 subrun numbers
    :param evt: numpy array of uint32 event numbers
    :return: numpy array of uint64 event ids
    """
    return (run << 30) + (subrun << 20) + evt


@numba.jit(nopython=True)
def make_evtseq_map(eids):
    """Return a numba.typed.Dict that maps the eid values to evtseq values.

    The evtseq values will go from 0 (inclusive) to eid.size (exclusive).

    :param eid: a numpy.array of uint64 event ids
    :return: a numba.type.Dict, which maps an eid to a uint64 evtseq.
    """
    result = numba.typed.Dict.empty(
        key_type=numba.types.uint64, value_type=numba.types.uint64
    )
    for i in range(np.uint64(eids.size)):
        result[eids[i]] = i
    return result


@numba.jit(nopython=True)
def apply_evtseq_map(mapping, eids):
    """Return a numpy.array of evtseq values

    The evtseq values are the result of mapping each eid to its corresponding evtseq,
    using the given mapping.

    :param mapping: a numba.typed.Dict carrying the evtseq value for each eid
    :param eid: a numpy.array of uint64 values, the eids to be mapped
    :return: a numpy.array of uint64 values
    """
    return np.array([mapping[x] for x in eids])
