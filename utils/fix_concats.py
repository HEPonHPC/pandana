import sys

import h5py
import numpy as np
import pandas as pd
import numba as nb

def readDataset(group, datasetname):
    ds = group[datasetname][:]

    if ds.shape[1] == 1:
        ds = ds.flatten()
    return ds

def create_reference(f):
    # Gather all index information
    cols = ['run','subrun','cycle','batch']
    dic = {col:readDataset(f['rec.hdr'],col) for col in cols}

    assert dic['run'].shape[0]

    # Group together to just the index information
    df = pd.DataFrame(dic)
    df = df.groupby(['run','subrun','cycle','batch'], sort=False, as_index=False).first()

    # Groupby changes the datatype for some reason
    # so manually change it back to the original
    return [df[c].to_numpy().astype(dic[c].dtype) for c in cols]

@nb.njit
def get_batch(gruns, gsubs, gcycs, gevts, refruns, refsubs, refcycs, refbats):
    retbats = []

    curridx = 0
    lastevt = -1

    # Loop through group index information and pull the corresponding
    # reference value. Importantly, the data in both groups are sorted
    # identically so we can use the evt number to differentiate
    # subsequent indices with the same run/subrun/cycle but different
    # batch numbers.
    for r,s,c,e in zip(gruns, gsubs, gcycs, gevts):
        # If we come to a new event,
        # we are guaranteed to be at a new index
        if e < lastevt:
            curridx += 1
        lastevt = e
        # We'll do a loop here, but the loop is only needed if data is missing
        # This does happen with the decaf cuts
        for i in range(curridx, refruns.shape[0]):
            if r == refruns[i] and s == refsubs[i] and c == refcycs[i]:
                retbats.append(refbats[i])
                curridx = i
                break

    return np.array(retbats)

def fix_spill(spill, RefRun, RefSubrun, RefCycle, RefBatch):
    if 'batch' not in spill.keys():
        cols = ['run','subrun','evt']
        spillrun, spillsubrun, spillevt = [readDataset(spill, col) for col in cols]

        # Use a placeholder cycle for the function
        spillcycle = np.zeros_like(spillrun)

        # Get the cycle and batch from the reference
        # For the cycle use placeholder in the reference to pull
        # the reference cycle instead of batch
        spillcycle = get_batch(spillrun, spillsubrun, spillcycle, spillevt, \
                               RefRun, RefSubrun, spillcycle, RefCycle)
        spillbatch = get_batch(spillrun, spillsubrun, spillcycle, spillevt, \
                               RefRun, RefSubrun, RefCycle, RefBatch)

        assert spillcycle.shape == spillrun.shape
        assert spillbatch.shape == spillrun.shape

        # Let's check the resulting index is unique
        df = pd.DataFrame({'run':spillrun, \
                           'subrun':spillsubrun, \
                           'cycle':spillcycle, \
                           'batch':spillbatch, \
                           'evt':spillevt})
        df = df.set_index(['run','subrun','cycle','batch','evt'])
        assert df.index.is_unique, 'spill nonunique index!'

        shape = spillrun.shape + (1,)

        spill.create_dataset(
            'cycle',
            data=spillcycle,
            shape=shape,
            shuffle=True,
            compression="gzip",
            compression_opts=6,
        )

        spill.create_dataset(
            'batch',
            data=spillbatch,
            shape=shape,
            shuffle=True,
            compression="gzip",
            compression_opts=6,
        )
    else:
        print('batch already exists - skipping')

def fix_group(group, RefRun, RefSubrun, RefCycle, RefBatch):
    if 'batch' not in group.keys():
        cols = ['run','subrun','cycle','evt']
        grun, gsub, gcyc, gevt = [readDataset(group, col) for col in cols]

        gbat = get_batch(grun, gsub, gcyc, gevt, \
                         RefRun, RefSubrun, RefCycle, RefBatch)

        assert gbat.shape == grun.shape, 'failed shape'

        shape = grun.shape + (1,)

        group.create_dataset(
            'batch',
            data=gbat,
            shape=shape,
            shuffle=True,
            compression="gzip",
            compression_opts=6,
        )
    else:
        print('batch already exists - skipping')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("{0}: you must specify a file to update".format(sys.argv[0]))
        sys.exit(1)

    with h5py.File(sys.argv[1], 'r+') as f:
        # The rec.hdr group has all the information we need
        # Use it to create a reference list
        print('Creating reference values')
        try:
            RefRun, RefSubrun, RefCycle, RefBatch = create_reference(f)
        except AssertionError:
            print(f'File {sys.argv[1]} has no data in it! Exiting...')
            sys.exit(1)

        # Use the reference to complete the other groups
        print('Fixing groups')
        for group in f:
            # This is some metadata or similar
            if not group+'/run' in f:
                continue

            print(f'Processing group {group}')
            # Everything in the spill tree is missing both batch and cycle
            if group.startswith('spill'):
                fix_spill(f[group], RefRun, RefSubrun, RefCycle, RefBatch)
            # Everything else is missing batch
            else:
                fix_group(f[group], RefRun, RefSubrun, RefCycle, RefBatch)
