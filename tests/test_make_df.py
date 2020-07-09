from unittest import TestCase
import h5py as h5
import numpy as np
import pandas as pd
from pandana.utils.make_df import make_df

class TestMakeDF(TestCase):
    def setUp(self):
        f = h5.File("/scratch-shared/paterno/small/small.h5caf.h5", "r")

    def tearDown(self):
        f.close()

    def test_read_list(self):
        spill = pandana.utils.make_df(self.f, "spill", ["run", "subrun", "evt"])
        self.assertIsInstance(spill, pd.DataFrame)
        self.assertEqual(len(spill.index), 2453)
        self.assertEqual(spill.columns, ["run", "subrun", "evt"])

    def test_read_all(self):
        spill = pandana.utils.make_df(self.f, "spill")
        self.assertIsInstance(spill, pd.DataFrame)
        self.assertEqual(len(spill.index), 2453)
        self.assertEqual(len(spill.columns), 54)
