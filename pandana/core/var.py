from pandana.core.cut import Cut

class Var:
    """Represents a variable.

    A variable may be directly read from a dataframe,
    calulcated from one or more things that were read,
    or calculated from other Vars.
    """

    def __init__(self, var):
        self._var = var
        
        self._CurrDF = None
        self._CurrTab = None

    # Remember result for each instance of tables
    def __call__(self, tables):
        if tables is not self._CurrTab:
            self._CurrDF = self._var(tables)
            self._CurrTab = tables
        return self._CurrDF

    def __eq__(self, val):
        return Cut(lambda tables: self(tables) == val)

    def __ne__(self, val):
        return Cut(lambda tables: self(tables) != val)

    def __lt__(self, val):
        return Cut(lambda tables: self(tables) < val)

    def __le__(self, val):
        return Cut(lambda tables: self(tables) <= val)

    def __gt__(self, val):
        return Cut(lambda tables: self(tables) > val)

    def __ge__(self, val):
        return Cut(lambda tables: self(tables) >= val)

    def __add__(self, other):
        return Var(lambda tables: self(tables) + other(tables))

    def __sub__(self, other):
        return Var(lambda tables: self(tables) - other(tables))

    def __mul__(self, other):
        return Var(lambda tables: self(tables) * other(tables))

    def __truediv__(self, other):
        return Var(lambda tables: self(tables) / other(tables))
