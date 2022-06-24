"""This module provides some numba-accelerated utility functions.

"""
import numba
import numpy as np


@numba.jit(nopython=True)
def ffill_vector_index(reference, vector_index):
    """Forward-fill according to a reference array of values
    traversed at a pace set by a vector-like index
    array where the reference and vector index arrays are different shapes

    Eg. Calculates the neutrino index vector for the neutrino primaries
    group given the neutrino vector index from the neutrinos group (left)
    and neutrino primaries (right)

    :param left_index: numpy array of the reference vector
    :param right_index: numpy array of the vector index
    :return: numpy array of the ffilled vector
    """
    trigger = np.ones_like(vector_index)
    trigger[1:] = ~(np.diff(vector_index) > 0)
    ffill = np.zeros_like(trigger)

    iref = -1
    for i in range(len(ffill)):
        iref = iref + trigger[i]
        ffill[i] = reference[iref]
    return ffill


@numba.jit(nopython=True)
def make_vector_index(indices_duplicated):
    """Return an array of values meant to index
    a vector nested within a non-unique set of event indices

    :param indices_duplicated: numpy array of boolean values
    indicating whether an index has been repeated.
    Eg, result of df.index.duplicated()
    :return: numpy array of uint64 nested vector indices
    """
    vidx = np.zeros_like(indices_duplicated)
    for i in range(1, len(indices_duplicated)):
        vidx[i] = (vidx[i - 1] + 1) * indices_duplicated[i]
    return vidx


@numba.vectorize(
    ["uint64(uint32,uint32,int64,int64,uint32)"],
)
def eid(run, subrun, cycle, batch, evt):
    """Calculate a single-number event id from run/subrun/event.

    :param run: numpy array of uint32 run numbers
    :param subrun: numpy array of uint32 subrun numbers
    :param cycle: numpy array of int32 subrun numbers
    :param batch: numpy array of int32 subrun numbers
    :param evt: numpy array of uint32 event numbers
    :return: numpy array of uint64 event ids
    """
    return (run << 45) + (subrun << 35) + (cycle << 25) + (batch << 20) + (evt)


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
