import os
import sys
sys.path.append('../..')

from PandAna.core import *
#  import matplotlib.pyplot as plt
#  import xkcd

# Simple var
kSlcE = Var(lambda tables: tables['rec.slc']['calE'])

# Simple cut
kEnergyCut = (kSlcE > 1) & (kSlcE < 4)

# Latest h5s from Karl
loc = '/pnfs/nova/persistent/users/karlwarb/HDF5-Training-19-02-26/FD-FluxSwap-FHC'
files = [os.path.join(loc, f) for f in os.listdir(loc) if 'h5caf.h5' in f]
tables = loader(files, limit=50)

# Create a spectrum
myspectrum = spectrum(tables, kEnergyCut, kSlcE)

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
