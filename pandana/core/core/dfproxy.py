import numpy as np
import pandas as pd


class dfproxy(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ['_proxycols']
    _internal_names_set = set(_internal_names)

    # proxy for a dataframe that builds a cache of columns needed to be read from the files
    # needed before Go() so Loader knows what to load
    @property
    def _constructor(self):
        return dfproxy

    def __init__(self, data=None, **kwargs):
        if data is None:
            data = []
        pd.DataFrame.__init__(self, data, **kwargs)
        self._proxycols = list(self.columns.values)

    def __getitem__(self, key):
        # add the column
        if type(key) is str and not key in self._proxycols:
            self._proxycols.append(key)
            self.__setitem__(key, np.nan)
            return self.__getitem__(key)
        # or all the columns
        if type(key) is list and not set(key)<=set(self._proxycols):
            for k in key:
                self._proxycols.append(k)
                self.__setitem__(k, np.nan)
            return self.__getitem__(key)
        # assume dataframe is being sliced inside cut/var, don't do anything
        if type(key) is not str and type(key) is not list:
            return self
        return pd.DataFrame.__getitem__(self, key)

    def __setitem__(self, key, val):
        pd.DataFrame.__setitem__(self, key, val)