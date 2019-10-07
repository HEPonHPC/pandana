import unittest
from pandana.utils.mpiutils import calculate_slice_for_rank


class TestCalculateSliceForRank(unittest.TestCase):
    def test_perfect_division(self):
        self.assertEqual(calculate_slice_for_rank(0, 3, 6), (0, 2))
        self.assertEqual(calculate_slice_for_rank(1, 3, 6), (2, 4))
        self.assertEqual(calculate_slice_for_rank(2, 3, 6), (4, 6))

    def test_one_too_many_slots(self):
        self.assertEqual(calculate_slice_for_rank(0, 3, 13), (0, 5))
        self.assertEqual(calculate_slice_for_rank(1, 3, 13), (5, 9))
        self.assertEqual(calculate_slice_for_rank(2, 3, 13), (9, 13))

    def test_one_rank(self):
        self.assertEqual(calculate_slice_for_rank(0, 1, 13), (0, 13))

    def test_one_too_few_slots(self):
        self.assertEqual(calculate_slice_for_rank(0, 4, 11), (0, 3))
        self.assertEqual(calculate_slice_for_rank(1, 4, 11), (3, 6))
        self.assertEqual(calculate_slice_for_rank(2, 4, 11), (6, 9))
        self.assertEqual(calculate_slice_for_rank(3, 4, 11), (9, 11))

    def test_too_many_ranks(self):
        self.assertEqual(calculate_slice_for_rank(0, 3, 1), (0, 1))
        self.assertEqual(calculate_slice_for_rank(1, 3, 1), (1, 1))
        self.assertEqual(calculate_slice_for_rank(2, 3, 1), (1, 1))

    def test_one_slot_per_rank(self):
        for myrank in range(0, 100):
            start, stop = calculate_slice_for_rank(myrank, 100, 100)
            self.assertEqual(stop - start, 1)

