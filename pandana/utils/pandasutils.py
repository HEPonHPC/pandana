import pandas as pd
from pandana.core.var import Var
from pandana.core.cut import Cut

def CutToVar(kCut):
    return Var(lambda tables: kCut(tables))

def VarsToVarND(kVars, join='inner'):
    def kVarND(tables):
        dflist = [var(tables) for var in kVars]
        df = pd.concat(dflist, axis=1, join=join)
        return df
    return Var(kVarND)
