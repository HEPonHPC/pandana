from .context import pandana
from unittest import TestCase
import pandana.core.loader
from pandana.core.loader import Loader
from pandana.core.var import Var
from pandana.core.cut import Cut
import h5py as h5
import numpy as np
import pandas as pd


class TestLoader(TestCase):
    def setUp(self):
        self.indices = ['run', 'subrun', 'evt', 'subevt']
        self.file = h5.File("fake", "w", driver="core", backing_store=False)
        self.nranks = 3
        self.run_id = 12
        self.subrun_id = 2
        self.event_ids = [1, 2, 4, 5, 6, 11, 15]
        self.event_id_to_evtseq = {eid: n for n, eid in enumerate(self.event_ids)}
        np.random.seed(123)
        self.create_spill_group()
        self.create_rec_energy_numu_group()
        self.loader = Loader(None, idcol="evt.seq", main_table_name="spill", indices=self.indices)
        self.loader._openfile = self.file
        self.loader._begin_evt = 0
        self.loader._end_evt = 100

    def create_spill_group(self):
        """
        Create a group that has one energy per event, but *not* dense event numbers.
        sets self.spill_group
        :return: None
        """
        self.spill_group = self.file.create_group("spill")
        n_events = len(self.event_ids)
        shape = (n_events, 1)
        u4 = np.dtype("u4")
        u8 = np.dtype("u8")
        self.spill_group.create_dataset(
            "run", data=np.repeat(self.run_id, n_events), dtype=u4, shape=shape
        )
        self.spill_group.create_dataset(
            "subrun", data=np.repeat(self.subrun_id, n_events), dtype=u4, shape=shape
        )
        self.spill_group.create_dataset(
            "evt", data=np.array(self.event_ids), dtype=u4, shape=shape
        )
        self.spill_group.create_dataset(
            "evt.seq", data=np.arange(n_events), dtype=u8, shape=shape
        )
        self.spill_group.create_dataset(
            "spillpot", data=np.random.random(n_events), shape=shape
        )

    def create_rec_energy_numu_group(self):
        """
        Create a group that has one entry per slice, but where some events have no slices.
        sets self.rec_energy_numu_group
        :return: None
        """
        self.rec_energy_numu_group = self.file.create_group("rec.energy.numu")
        # Slices are organized so that we have events with zero slices in
        # middle, at start, and at end of spans processed by each rank.
        slices_per_event = [5, 0, 3, 0, 3, 1, 0]

        n_slices = sum(slices_per_event)
        shape = (n_slices, 1)
        u4 = np.dtype("u4")
        u8 = np.dtype("u8")
        self.rec_energy_numu_group.create_dataset(
            "run", data=np.repeat(self.run_id, n_slices), dtype=u4, shape=shape
        )
        self.rec_energy_numu_group.create_dataset(
            "subrun", data=np.repeat(self.subrun_id, n_slices), dtype=u4, shape=shape
        )

        # Create indexing columns to keep data "lined up" across tables.
        evt_column = []
        subevt_column = []
        for event_id, slices_this_event in zip(self.event_ids, slices_per_event):
            evt_column.append([event_id for _ in range(slices_this_event)])
            subevt_column.append([i for i in range(slices_this_event)])

        self.rec_energy_numu_group.create_dataset(
            "evt",
            data=np.array([item for sublist in evt_column for item in sublist]),
            dtype=u4,
            shape=shape,
        )
        self.rec_energy_numu_group.create_dataset(
            "subevt",
            data=np.array([item for sublist in subevt_column for item in sublist]),
            dtype=u4,
            shape=shape,
        )
        self.rec_energy_numu_group.create_dataset(
            "evt.seq",
            data=np.array(
                [
                    self.event_id_to_evtseq[event_id]
                    for event_id in self.rec_energy_numu_group["evt"][:].flatten()
                ]
            ),
            dtype=u8,
            shape=shape,
        )
        self.assertEqual(
            self.rec_energy_numu_group["evt"].size,
            self.rec_energy_numu_group["subevt"].size,
        )
        self.assertTrue(
            np.all(
                self.rec_energy_numu_group["evt"][()].flatten()
                == np.array([1, 1, 1, 1, 1, 4, 4, 4, 6, 6, 6, 11])
            )
        )
        self.assertTrue(
            np.all(
                self.rec_energy_numu_group["subevt"][()].flatten()
                == np.array([0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 0])
            )
        )

        # Other columns are just random numbers...
        f4 = np.dtype("f4")
        for name in [
            "E",
            "angleE",
            "calccE",
            "hadcalE",
            "hadtrkE",
            "ndhadcalactE",
            "ndhadcalcatE",
            "ndhadcaltranE",
            "ndhadtrkactE",
            "ndhadtrkcatE",
            "ndhadtrktranE",
            "ndtrkcalactE",
            "ndtrkcalcatE",
            "ndtrkcaltranE",
            "recomuonE",
            "recotrkcchadE",
            "shiftedtrkccE",
            "trkccE",
            "trknonqeE",
            "trkqeE",
            "ucrecomuonE",
        ]:
            self.rec_energy_numu_group.create_dataset(
                name, data=np.random.random(n_slices).astype(f4), shape=shape
            )

    def tearDown(self):
        self.file.close()

    def test_calculateEventRange_simple(self):
        # Answers should be:
        #   [0, 2]
        #   [3, 4]
        #   [5, 6]
        b, e = self.loader.calculateEventRange(self.spill_group, 0, self.nranks)
        self.assertEqual(b, 0)
        self.assertEqual(e, 2)
        b, e = self.loader.calculateEventRange(self.spill_group, 1, self.nranks)
        self.assertEqual(b, 3)
        self.assertEqual(e, 4)
        b, e = self.loader.calculateEventRange(self.spill_group, 2, self.nranks)
        self.assertEqual(b, 5)
        self.assertEqual(e, 6)

    def test_calculateEventRange_too_many_ranks(self):
        too_many_ranks = len(self.event_ids) + 1
        with self.assertRaises(ValueError):
            b, e = self.loader.calculateEventRange(self.spill_group, 0, too_many_ranks)

    def test_createSpillDataFrame(self):
        dflist = []
        for rank in range(self.nranks):
            b, e = self.loader.calculateEventRange(self.spill_group, rank, self.nranks)
            dflist.append(
                pandana.core.DataGroup(
                    self.file,
                    "spill",
                    b, e,
                    "evt.seq",
                    self.indices)[['spillpot']]
            )
        all(self.assertIsInstance(df, pd.DataFrame) for df in dflist)
        num_rows_in_dataframes = [len(df.index) for df in dflist]
        print("createSpillDataFrame")
        for df in dflist:
            print(df)
        self.assertEqual(num_rows_in_dataframes, [3, 2, 2])
        print("end createSpillDataFrame")

    def test_createRecEnergyNumuDataFrame(self):
        print("Full dataframe")
        print(
            pandana.core.DataGroup(
                self.file,
                "rec.energy.numu",
                0, 100,
                "evt.seq",
                self.indices)[['trkccE']]
        )

        dflist = []
        for rank in range(self.nranks):
            b, e = self.loader.calculateEventRange(self.spill_group, rank, self.nranks)
            dflist.append(
                pandana.core.DataGroup(
                    self.file,
                    "rec.energy.numu",
                    b, e,
                    "evt.seq",
                    self.indices)[['trkccE']]
            )
        all(self.assertIsInstance(df, pd.DataFrame) for df in dflist)
        num_rows_in_dataframes = [len(df.index) for df in dflist]
        for df in dflist:
            print(df)
        self.assertEqual(num_rows_in_dataframes, [8, 3, 1])

    def test_cutloader(self):
        def kSimpleVar(tables):
            df = tables['rec.energy.numu']['trkccE']
            return df
        kSimpleVar = Var(kSimpleVar)
        
        kCut1 = kSimpleVar > 0.5
        self.assertEqual(kCut1(self.loader).sum(), 4)

        kCut2 = kSimpleVar <= 0.5
        
        kAndCut = kCut1 & kCut2
        self.assertEqual(kAndCut(self.loader).sum(), 0)

        kOrCut = kCut1 | kCut2
        self.assertEqual(kOrCut(self.loader).sum(), 12)

        kSpillCut = Cut(lambda tables: tables['spill']['spillpot'] > 0.5)

        kUnalignAndCut = kCut1 & kSpillCut
        self.assertEqual(kUnalignAndCut(self.loader).sum(), 3)

        kUnalignOrCut = kCut1 | kSpillCut
        self.assertEqual(kUnalignOrCut(self.loader).sum(), 9)
