import numba
import numpy as np


@numba.jit(nopython=True)
def make_evtseq_map(eid):
    """
    Create a numba.typed.Dict that maps the given order of eid values to evtseq values. The evtseq values will go
    from 0 (inclusive) to eid.size (exclusive).
    :param eid: a numpy.array of uint64 event ids
    :return: a numba.type.Dict, which maps an eid to a uint64 evtseq.
    """
    result = numba.typed.Dict.empty(key_type=numba.types.uint64, value_type=numba.types.uint64)
    for i in range(np.uint64(eid.size)):
        result[eid[i]] = i
    return result


@numba.jit(nopython=True)
def apply_evtseq_map(mapping, eid):
    """
    Return a numpy.array of uint64, the result of mapping each eid to its corresponding evtseq, using the given mapping.
    :param mapping: a numba.typed.Dict carrying the evtseq value for each eid
    :param eid: a numpy.array of uint64 values, the eids to be mapped
    :return: a numpy.array of uint64 values, the resulting evtseq
    """
    return np.array([mapping[x] for x in eid])


@numba.jit(nopython=True)
def apply_evtseq_map(mapping, eid):
    """
    Return a numpy.array of uint64, the result of mapping each eid to its corresponding evtseq, using the given mapping.
    :param mapping: a numba.typed.Dict carrying the evtseq value for each eid
    :param eid: a numpy.array of uint64 values, the eids to be mapped
    :return: a numpy.array of uint64 values, the resulting evtseq
    """
    return np.array([mapping[x] for x in eid])
