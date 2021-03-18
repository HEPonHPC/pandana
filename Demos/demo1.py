import sys

from pandana.core import *
from nova.utils.index import index, KL

# Better var
def kPngE(tables):
    df = tables["rec.vtx.elastic.fuzzyk.png"]["calE"]
    return df.groupby(level=KL).sum()
kPngE = Var(kPngE)
# NOTE: No check that there are actually prongs in the event
# The rec.vtx.elastic.fuzzyk.png dataset already only has events with prongs

# Vertex fiducial cut
def kFiducial(tables):
    df = tables["rec.vtx.elastic"]
    return (
        (df["vtx.x"] < 180)
        & (df["vtx.x"] > -180)
        & (df["vtx.y"] < 180)
        & (df["vtx.y"] > -180)
        & (df["vtx.z"] < 1000)
        & (df["vtx.z"] > 50)
    ).groupby(level=KL).first()
kFiducial = Cut(kFiducial)

# Dumb oscillation weights
kDumbOsc = Var(lambda tables: tables['rec.mc.nu']['woscdumb'].groupby(level=KL).first())

# Initialize the loader
fname = sys.argv[1]
tables = Loader(fname, idcol='evt.seq', main_table_name='spill', indices=index)

# Create a Spectrum
myspectrum = Spectrum(tables, kFiducial, kPngE, kDumbOsc)

# Let's do it!
tables.Go()

print("Weighted number of selected events")
print((myspectrum.integral()))

