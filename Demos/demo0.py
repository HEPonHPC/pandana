import os
import sys

from pandana.core.loader import Loader
from pandana.core.var import Var

newdir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, newdir)

from pandana.core import *
#  import matplotlib.pyplot as plt
#  import xkcd

# Simple var
kSlcE = Var(lambda tables: tables['rec.slc']['calE'])

# Simple cut
kEnergyCut = (kSlcE > 1) & (kSlcE < 4)

# Latest h5s from Karl
loc = sys.argv[1]
files = [os.path.join(loc, f) for f in os.listdir(loc) if 'h5caf.h5' in f]
tables = Loader(files, limit=50)

# Create a Spectrum
myspectrum = Spectrum(tables, kEnergyCut, kSlcE)

# Let's do it!
tables.Go()

print('myspectrum internal dataframe: ')
print(myspectrum.df().head())

#  n, bins = myspectrum.histogram(bins=50, range=(1,4))
#
#  print('Selected ' + str(n.sum()) + ' events from ' + str(myspectrum.POT()) + ' POT.')
#
#  plt.hist(bins[:-1], bins=bins, weights=n, histtype='step', color='xkcd:dark blue', label='FluxSwap')
#  plt.xlabel('Slice calE')
#  plt.ylabel('Events')
#
#  plt.legend(loc='upper right')
#
#  plt.show()
