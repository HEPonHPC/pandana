import sys

from pandana.core import *
from nova.utils.index import index

# Simple var
kSlcE = Var(lambda tables: tables["rec.slc"]["calE"])

# Simple cut
kEnergyCut = (kSlcE > 1) & (kSlcE < 4)

# Construct loader from a file
fname = sys.argv[1]
tables = Loader(fname, idcol='evt.seq', main_table_name='spill', indices=index)

# Create a Spectrum
myspectrum = Spectrum(tables, kEnergyCut, kSlcE)

# Let's do it!
tables.Go()

print("myspectrum internal dataframe: ")
print(myspectrum.df())

n, bins = myspectrum.histogram(bins=10, range=(1,4))
print('Selected',n.sum(),'events.')
print('Bin Contents: ')
print(n)
