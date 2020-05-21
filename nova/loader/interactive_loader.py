import collections
import time
import h5py
import pandas as pd
from mpi4py import MPI

from pandana import SourceWrapper, DFProxy
from pandana.core.loader import Loader
from pandana.core.indices import KL, KLN, KLS
from pandana import utils

class InteractiveLoader():
    def __init__(self, files):
        if type(files) is not list:
            files = [files]
        self._files = files
        self._tables = {}
        self.gone = False
        self.interactive = True
        self._indices = None

    def keys(self, contain=None):
        f = self._files[0]
        h5 = h5py.File(f, 'r')
        keys = list(h5.keys())
        for k in list(h5.keys()):
            if contain:
                if contain in k:
                    print(k)
            else:
                print(k)
        h5.close()
        
        return keys

    def __getitem__(self, key):
        if not key in self._tables:
            dflist=[]
            for fname in self._files:
                f = h5py.File(fname,'r')
                group = f.get(key)
                values = {}
                for k in list(group.keys()):
                    dataset = group.get(k)[()]
                    if dataset.shape[1] == 1:
                        dataset = dataset.flatten()
                    else:
                        dataset = list(dataset)
                    values[k] = dataset
                dflist.append(pd.DataFrame(values))
                f.close()
            df = pd.concat(dflist)
            if not (key.startswith('spill') or key.startswith('neutrino')):
                df.set_index(KL, inplace=True)
            self._tables[key] = df
        return self._tables[key]
