import _candidate_selection
from glob import glob
import os
import sys

input_directory = sys.argv[1]
idcol = sys.argv[2]
max_files = 50
if len(sys.argv) > 3:
    max_files = int(sys.argv[3])
input_files = glob(os.path.join(input_directory, '*.h5caf.h5'))
_candidate_selection.main(input_files, idcol, max_files)

