import os
import sys
from glob import glob
from context import pandana
from pandana.core import *
from pandana.cut.analysis_cuts import kNumuCutND

def main():
  # import matplotlib.pyplot as plt
  
  kSlcE = Var(lambda tables: tables['rec.slc']['calE'])
  
  # Read data files.
  #
  # This needs to be improved, and will probably vastly change when we have one
  # or few large (concatenated) files.
  
  input_directory = sys.argv[1]
  maxfiles = 50
  if (len(sys.argv) > 2):
      maxfiles = int(sys.argv[2])
  input_files = glob(os.path.join(input_directory, '*.h5caf.h5'))
  tables = loader(input_files, limit=maxfiles)
  
  # Create a spectrum for the events passing our candidate cut.
  myspectrum = spectrum(tables, kNumuCutND, kSlcE)
  
  # Let's do it!
  tables.Go()
  
  print('myspectrum internal dataframe: ')
  print((myspectrum.df().head()))
  
  n, bins = myspectrum.histogram(bins=50, range=(1, 4))
  print('Selected ', n.sum(), ' events from ', myspectrum.POT(), '  POT.')
  
  # plt.hist(bins[:-1], bins=bins, weights=n, histtype='step', label='Candidates!')
  # plt.xlabel('Slice calE')
  # plt.ylabel('Events')
  # plt.legend(loc='upper right')
  # plt.show()
