import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_directory", help = "directory from which to read input files")
parser.add_argument("id_col", help = "name of the globally unique event id column")
parser.add_argument("max_files", help = "maximum number of files to read", type = int)
args = parser.parse_args()

import _candidate_selection
from glob import glob
import os
import sys

input_files = glob(os.path.join(args.input_directory, '*.h5caf.h5'))
_candidate_selection.main(input_files, args.id_col, args.max_files)

