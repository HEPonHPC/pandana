import numpy as np
import pandas as pd


class DFProxy(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ['_proxycols']
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        """_constructor property is required by Pandas for subclasss of pd.DataFrame.
        """
        return DFProxy

    def __init__(self, data=None, **kwargs):
        if data is None:
            data = []
        pd.DataFrame.__init__(self, data, **kwargs)
        self._proxycols = list(self.columns.values)

    def __getitem__(self, key):
        # TODO: What are all the legal types that can be used as 'key'?
        # add the column
        if type(key) is str and not key in self._proxycols:
            self._proxycols.append(key)
            self.__setitem__(key, np.nan)
            # TODO: Evaluate replacing recursive all with direct call to pd.DataFrame.__getitem__(self, key)
            return self.__getitem__(key)
        # or all the columns
        if type(key) is list and not set(key) <= set(self._proxycols):
            for k in key:
                self._proxycols.append(k)
                self.__setitem__(k, np.nan)  # Default values are floating point
            return self.__getitem__(key)
        # assume dataframe is being sliced inside cut/var, don't do anything
        if type(key) is not str and type(key) is not list:
            return self
        return pd.DataFrame.__getitem__(self, key)

    def __setitem__(self, key, val):
        # TODO: Remove this function; it is a needless override of the base class.
        pd.DataFrame.__setitem__(self, key, val)
