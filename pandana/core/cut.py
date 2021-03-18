import pandas as pd

class Cut:
    """Represents a selection criterion to be applied to a dataframe."""

    def __init__(self, cut):
        self._cut = cut

    def __call__(self, tables):
        # If this is the first call construct the dataframe and store in the tables
        if not self in tables.ComputedCuts:
            df = self._cut(tables)
            tables.ComputedCuts[self] = df
        # Otherwise access directly from the tables
        return tables.ComputedCuts[self]

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
