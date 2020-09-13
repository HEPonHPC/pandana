from unittest import TestCase
import h5py as h5
import numpy as np
import pandas as pd

# from utils.make_df import make_df


class TestMakeDF(TestCase):
    def setUp(self):
        # We need a small file to create and put into self.f
        pass

    def tearDown(self):
        pass

    def test_read_list(self):
        # spill = make_df(self.f, "spill", ["run", "subrun", "evt"])
        # self.assertIsInstance(spill, pd.DataFrame)
        # self.assertEqual(len(spill.index), 2453)
        # self.assertEqual(spill.columns, ["run", "subrun", "evt"])
        pass

    def test_read_all(self):
        # spill = make_df(self.f, "spill")
        # self.assertIsInstance(spill, pd.DataFrame)
        # self.assertEqual(len(spill.index), 2453)
        # self.assertEqual(len(spill.columns), 54)
        pass
