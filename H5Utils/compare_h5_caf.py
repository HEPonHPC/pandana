#!/usr/bin/env python

import argparse
import time
import re
import os
import sys
import h5py
import numpy as np
import warnings

warnings.filterwarnings("ignore",category = RuntimeWarning)

KL = ['run', 'subrun', 'cycle', 'evt', 'subevt']
skip = ['genVersion', 'daughterlist', 'motherlist', 'fuzzyk.png.bpf','hough','vdt','spill.bpos', 'spill.int', 'training', 'cvnmap']

parser = argparse.ArgumentParser(description = 'Compare contents of a HDF5 file with its CAF equivalent.\rUses TTree Draw to check leaf content from the CAF files and prints a message if its different from the HDF5 file.\rSome branches are just too expensive to evaluate or the leaves are too deep for TTree::Draw to work. Others have no data filled or are defunct.\rBranches that contain these are skipped : '+str(skip))
parser.add_argument('-h5','--h5file', type=str, required=True,
                    help='Path to hdf5 file')
parser.add_argument('-r','--rootfile', type=str, required=True,
                    help='Path to CAF file')
parser.add_argument('-br', '--branches', type=str, 
                    help='Check only branches matching regex condition')
parser.add_argument('-a', '--all', action='store_true',
                    help='Check all branches')
opts = parser.parse_args()

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import *
gROOT.SetBatch(True)
h5file = os.path.abspath(opts.h5file)
rootfile = os.path.abspath(opts.rootfile)

fh5 = h5py.File(h5file, 'r')
fcaf = TFile(rootfile, 'read')

keystocheck = []
if opts.all:
    keystocheck = fh5.keys()
if opts.branches:
    keystocheck = [key for key in fh5.keys() if re.match(opts.branches, key)]
if not len(keystocheck):
    sys.exit('No valid keys found. Please provide either valid option -br or option -a')

start = time.time()
for branchkey in keystocheck:
    if any([check in branchkey for check in skip]):
        continue
    
    print "Testing... ", branchkey
    print "============================"
    roottree = recTree
    rootbranchkey = branchkey

    branch = fh5.get(branchkey)
    branchkeys = []
    
    if branchkey.startswith('neutrino'):
        roottree = nuTree
        rootbranchkey = re.sub(r'neutrino','nu',branchkey)
        branchkeys = [key for key in branch.keys() if (key not in KL) and ('idx' not in key)]
    if branchkey.startswith('spill'):
        roottree = spillTree
        branchkeys = [key for key in branch.keys() if 'idx' not in key]
    if branchkey.startswith('rec'):
        branchkeys = [key for key in branch.keys() if (key not in KL) and ('idx' not in key)]

    idx = 0
    total = len(branchkeys)
    for key in branchkeys:
        h5data = branch.get(key)[()].flatten()

        rootleafkey = rootbranchkey+'.'+key
        if key == 'value':
            rootleafkey = re.sub(r'.value','',rootbranchkey+'.'+key)
        
        roottree.SetEstimate(100000000)
        roottree.Draw(rootleafkey)
        nentries = roottree.GetSelectedRows()
        rootdata = np.zeros(nentries)
        roottemp = roottree.GetV1()
        for i in range(nentries):
            rootdata[i] = roottemp[i]
        
        if len(h5data) != len(rootdata):
            print "************"
            print "NEntries mismatch!"
            continue
        if not all(h5data == h5data):
            if not (len(h5data) == len(rootdata)):
                print "************"
                print branchkey+'.'+key
                print "Entries w/ NaNs. Size mismatch!"
                continue
            h5data = h5data[h5data==h5data]
            rootdata = rootdata[rootdata == rootdata]
            if not all(h5data == rootdata):
                print "************"
                print branchkey+'.'+key
                print "Non NaN data doesn't match!"
                continue
        else:
            if not all(h5data == rootdata):
                print "************"
                print branchkey+'.'+key
                print "Doesn't match"
                continue

print "============================"
print "Total time taken : ", time.time() - start
