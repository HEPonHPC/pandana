import unittest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import nothing
from utils.vfuncs import eid
import numpy as np


st = arrays(dtype=np.uint32, shape=10, fill=nothing())


class MyTestCase(unittest.TestCase):
    @given(st, st, st, st, st, st)
    def test_eids_collide_only_when_inputs_are_equal(self, r1, s1, e1, r2, s2, e2):
        eids_match = eid(r1, s1, e1) == eid(r2, s2, e2)
        inputs_match = np.logical_and(r1 == r2, np.logical_and(s1 == s2, e1 == e2))
        assert np.all(eids_match == inputs_match)


if __name__ == "__main__":
    unittest.main()
