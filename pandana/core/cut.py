from functools import lru_cache

import pandas as pd

class Cut:
    """Represents a selection criterion to be applied to a dataframe."""

    def __init__(self, cut):
        self._cut = cut

    # Remember result for each instance of tables
    @lru_cache(maxsize=1)
    def __call__(self, tables):
        return self._cut(tables)

    def __invert__(self):
        return Cut(lambda tables: ~self(tables))

    def __and__(self, other):
        def AndCut(tables):
            df1 = self(tables)
            df2 = other(tables)

            if not df1.index.equals(df2.index):
                df2, df1 = df2.align(df1, axis=0, join='inner')
            ret = df1.to_numpy() & df2.to_numpy()
            ret = pd.Series(ret, index=df1.index)

            return ret
        return Cut(AndCut)

    def __or__(self, other):
        def OrCut(tables):
            df1 = self(tables)
            df2 = other(tables)

            if not df1.index.equals(df2.index):
                df2, df1 = df2.align(df1, axis=0, join='inner')
            ret = df1.to_numpy() | df2.to_numpy()
            ret = pd.Series(ret, index=df1.index)

            return ret
        return Cut(OrCut)
