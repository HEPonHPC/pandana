from context import pandana
from pandana.core import *
from pandana.core.core.var import Var
from pandana.cut.analysis_cuts import kNumuCutND


def main(input_files, max_files):
    tables = loader(input_files, limit=max_files)
    energy = Var(lambda tables: tables['rec.slc']['calE'])
    my_spectrum = Spectrum(tables, kNumuCutND, energy)
    tables.Go()

    print('my_spectrum internal dataframe: ')
    print((my_spectrum.df().head()))

    n, _ = my_spectrum.histogram(bins=50, range=(1, 4))
    print('Selected ', n.sum(), ' events from ', my_spectrum.POT(), '  POT.')
