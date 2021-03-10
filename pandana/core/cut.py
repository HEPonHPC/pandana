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
        return Cut(lambda tables: self(tables) & other(tables))

    def __or__(self, other):
        # It seems | is not commutative for dataframes? Do this for now
        def orcut(tables):
            df1 = self(tables)
            df2 = other(tables)
            df = pd.concat([df1,df2], axis=1, join='outer').fillna(False)
            return df.any(axis=1)
        return Cut(orcut)
