from pandana.core.core.indices import KL
from pandana.core.core.var import Var

KLP = KL + ['rec.vtx.elastic.fuzzyk.png_idx']

#Prong Vars

def kPngMap(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.cvnmaps']
    return df['cvnmap']
kPngMap = Var(kPngMap)

def kPngPur(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.prongtrainingdata']
    return df['purity3d']
kPngPur = Var(kPngPur)

def kPngNCellX(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.prongtrainingdata']
    return df['ncellx']
kPngNCellX = Var(kPngNCellX)

def kPngNCellY(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.prongtrainingdata']
    return df['ncelly']
kPngNCellY = Var(kPngNCellY)

def kPngLabel(tables):
    return tables['rec.vtx.elastic.fuzzyk.png.prongtrainingdata']['label3d']
kPngLabel = Var(kPngLabel)

def kPngPDG(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.truth']
    return df['pdg']
kPngPDG = Var(kPngPDG)

#Slice Vars

def kSliceInter(tables):
    return tables['rec.training.trainingdata']['interaction']
kSliceInter = Var(kSliceInter)

def kSliceMap(tables):
    return tables['rec.training.cvnmaps']['cvnmap']
kSliceMap = Var(kSliceMap)
