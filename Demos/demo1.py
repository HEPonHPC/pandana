import os
import sys
sys.path.append('../..')

from pandana.core import *
import matplotlib.pyplot as plt

# Better var
def kPngE(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png']['calE']
    return df.groupby(level = KL).sum()
kPngE = Var(kPngE)
# NOTE: No check that there are actually prongs in the event
# The rec.vtx.elastic.fuzzyk.png dataset already only has events with prongs

# Vertex fiducial cut
def kFiducial(tables):
    df = tables['rec.vtx.elastic']
    return (df['vtx.x'] < 180) & \
        (df['vtx.x'] > -180) & \
        (df['vtx.y'] < 180) & \
        (df['vtx.y'] > -180) & \
        (df['vtx.z'] < 1000) & \
        (df['vtx.z'] > 50)
kFiducial = Cut(kFiducial)

# Latest h5s from Karl
loc = '/pnfs/nova/persistent/users/karlwarb/HDF5-Training-19-02-26/FD-FluxSwap-FHC'
files = [os.path.join(loc, f) for f in os.listdir(loc) if 'h5caf.h5' in f]
tables = loader(files, limit=100)

# Create a spectrum
myspectrum = spectrum(tables, kFiducial, kPngE)

# Let's do it!
tables.Go()

print('myspectrum internal dataframe: ')
print((myspectrum.df().head()))

n, bins = myspectrum.histogram(bins=20, range=(1,4))

print(('Selected '+ str(n.sum()) + ' events'))

plt.hist(bins[:-1], bins=bins, weights=n)

plt.xlabel('Prong Energy')
plt.ylabel('Events')

plt.legend(loc='upper right')

plt.show()
