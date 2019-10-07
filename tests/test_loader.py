from .context import pandana
from unittest import TestCase
import unittest
from pandana.core.core.loader import Loader
import h5py as h5
import numpy as np


class TestLoader(TestCase):
    def setUp(self):
        self.loader = Loader(None)
        self.file = h5.File('fake', 'w', driver='core', backing_store = False)
        self.grp = self.file.create_group('spill')
        self.grp.create_dataset('evt', data = np.array([1, 2, 4, 5, 6, 11, 15]), shape = (7, 1))

    def tearDown(self):
        self.file.close()

    def test_calculateEventRange_simple(self):
        # Answers should be:
        #   [1, 4]
        #   [5, 6]
        #   [11, 15]
        nranks = 3
        b, e = self.loader.calculateEventRange(self.grp, 0, nranks)
        self.assertEqual(b, 1)
        self.assertEqual(e, 4)
        self.assertEqual(b, 1)
        self.assertEqual(e, 4)
        b, e = self.loader.calculateEventRange(self.grp, 1, nranks)
        self.assertEqual(b, 5)
        self.assertEqual(e, 6)
        b, e = self.loader.calculateEventRange(self.grp, 2, nranks)
        self.assertEqual(b, 11)
        self.assertEqual(e, 15)

    def test_calculateEventRange_too_many_ranks(self):
        with self.assertRaises(IndexError):
            b, e = self.loader.calculateEventRange(self.grp, 8, 9)
