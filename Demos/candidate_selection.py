import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs="+", help="one or more input files to read")
parser.add_argument("id_col", help="name of the globally unique event id column")
args = parser.parse_args()

import _candidate_selection

# Note: 'filename' is actually an array of filenames
_candidate_selection.main(args.filename, args.id_col)
