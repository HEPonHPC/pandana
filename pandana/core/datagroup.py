import pandas as pd

class DataGroup:
    """Represents a group in an hdf5 file"""

    def __init__(self, h5pyfile, group, begin_evt, end_evt, idcol, indices):
        self._group = h5pyfile.get(group)

        # The dataframe indices are a subet of all available indices
        # Primary loop is over the available indices to keep a consistent order
        self._index = [k for k in indices if k in self._group.keys()]

        # Compute the row range for this group
        event_seq_numbers = self._group[idcol][()].flatten()
        self._begin_row, self._end_row = event_seq_numbers.searchsorted([begin_evt, end_evt + 1])

        self._df = None

    def readDatasetFromGroup(self, datasetname):
        # Determine range to be read here.
        # Regardless of the dataset, we want to read all the entries corresponding to the range of events
        # (not runs, subruns, or subevents, but events) we are to process.
        # dataset is a numpy.array, not a h5py.Dataset.
        ds = self._group.get(datasetname)  # ds is a h5py.Dataset
        dataset = ds[self._begin_row:self._end_row]

        if dataset.shape[1] == 1:
            dataset = dataset.flatten()
        # If more than one element is in the dataset per row,
        # it is assumed we want a single column with the data as a list
        else:
            dataset = list(dataset)

        return dataset

    def __getitem__(self, key):
        """
        Access a data member of this h5 group as a dataframe
        key can be a single key or a list of keys to access multiple columns at once

        :return: A pd.Series or pd.DataFrame
        """

        # If this is the first time this group is being accessed,
        # gather the key values together with the index values
        # and create the DataFrame
        if self._df is None:
            if isinstance(key, list):
                values = {k: self.readDatasetFromGroup(k) for k in self._index+key}
            else:
                values = {k: self.readDatasetFromGroup(k) for k in self._index+[key]}
            self._df = pd.DataFrame(values)
            self._df.set_index(self._index, inplace=True)
        # Otherwise populate each new key
        else:
            if isinstance(key, list):
                for k in key:
                    if not k in self._df:
                        values = self.readDatasetFromGroup(k)
                        self._df[k] = values
            else:
                if not key in self._df:
                    values = self.readDatasetFromGroup(key)
                    self._df[key] = values

        return self._df[key]

    def __str__(self):
        return self._df.__str__()
