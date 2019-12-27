from pandana import Cut


class Var():
    def __init__(self, var):
        self._var = var

    def __call__(self, tables):
        return self._var(tables)

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

    def __mult__(self, other):
        return Var(lambda tables: self(tables)*other(tables))

    def __truediv__(self, other):
        return Var(lambda tables: self(tables)/other(tables))