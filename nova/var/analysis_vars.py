import numpy as np
import pandas as pd
from pandana.core.var import Var

from nova.utils.misc import *
from nova.var.numuE_utils import *
from nova.utils.index import KL

import numba as nb

# cvnProd3Train is messed up in the current iteration of files.
kCVNe = Var(lambda tables: tables['rec.sel.cvn2017']['nueid'])
kCVNm = Var(lambda tables: tables['rec.sel.cvn2017']['numuid'])
kCVNnc = Var(lambda tables: tables['rec.sel.cvn2017']['ncid'])

kNHit = Var(lambda tables: tables['rec.slc']['nhit'])

kRHC   = Var(lambda tables: tables['rec.spill']['isRHC'])
kDetID = Var(lambda tables: tables['rec.hdr']['det'])

# Containment vars
@nb.vectorize([nb.int32(nb.int32,nb.int32)], nopython=True)
def calcFirstLivePlane(mask, fp):
    fd = fp//64
    dmin = fd

    for i in range(fd, -1, -1):
        temp = mask >> i
        if temp & 1 == 0:
            break
        else:
            dmin = i
    return 64*dmin

def planestofront(tables):
    mask = tables['rec.hdr']['dibmask']
    fp = tables['rec.slc']['firstplane']
    return fp - pd.Series(calcFirstLivePlane(mask.to_numpy(dtype=np.int32), fp.to_numpy(dtype=np.int32)), index=mask.index)
planestofront = Var(planestofront)

@nb.vectorize([nb.int32(nb.int32,nb.int32)], nopython=True)
def calcLastLivePlane(mask, lp):
    ld = lp//64
    dmax = ld

    for i in range(ld, 14, 1):
        temp = mask >> i
        if temp & 1 == 0:
            break
        else:
            dmax = i
    return 64*(dmax+1)-1

def planestoback(tables):
    mask = tables['rec.hdr']['dibmask']
    lp = tables['rec.slc']['lastplane']
    return pd.Series(calcLastLivePlane(mask.to_numpy(dtype=np.int32), lp.to_numpy(dtype=np.int32)), index=mask.index) - lp
planestoback = Var(planestoback)

###################################################################################
#
# Nue Vars
#
###################################################################################

def kLongestProng(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png']['len']
    return df.groupby(level=KL).agg(np.max)    
kLongestProng = Var(kLongestProng)

kDistAllTop    = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngtop'])
kDistAllBottom = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngbottom'])
kDistAllWest   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngwest'])
kDistAllEast   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngeast'])
kDistAllBack   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngback'])
kDistAllFront  = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngfront'])

kHitsPerPlane = Var(lambda tables: tables['rec.sel.nuecosrej']['hitsperplane'])

kPtP = Var(lambda tables: tables['rec.sel.nuecosrej']['partptp'])

def kMaxY(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    df = df[['start.y','stop.y']].max(axis=1)
    return df.groupby(level=KL).agg(np.max)
kMaxY = Var(kMaxY)

kSparsenessAsymm = Var(lambda tables: tables['rec.sel.nuecosrej']['sparsenessasymm'])

kCaloE = Var(lambda tables: tables['rec.slc']['calE'])

def kNueCalibrationCorrFunc(det, ismc, run):
  if (det != detector.kFD): return 1.
  if not ismc: return 0.9949
  if run < 20753: return 0.9949/0.9844
  return 1.
kNueCalibrationCorrFunc = np.vectorize(kNueCalibrationCorrFunc, otypes=[np.float32])

def kNueCalibrationCorr(tables):
    hdr_df = tables['rec.hdr'][['det','ismc']]
    hdr_df['run'] = hdr_df.index.get_level_values('run')

    scale = pd.Series(kNueCalibrationCorrFunc(hdr_df['det'], hdr_df['ismc'], hdr_df['run']),
                      index=hdr_df.index)
    return scale
kNueCalibrationCorr = Var(kNueCalibrationCorr)

def kEMEnergy(tables):
    lng_png = kLongestProng(tables)

    shwlid_df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    prim_png_calE = shwlid_df['calE'].groupby(level=KL).first()
      
    cvn_png_df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart']
    cvn_em_pid_df = cvn_png_df[['photonid',  \
                                'pizeroid',  \
                                'electronid']].sum(axis=1)

    cvn_em_calE = shwlid_df['calE'].where((cvn_em_pid_df >= 0.5), 0).groupby(level=KL).agg(np.sum)
        
    cvn_em_calE[cvn_em_calE == 0] = prim_png_calE
    cvn_em_calE[lng_png >= 500] = prim_png_calE

    cvn_em_calE *= kNueCalibrationCorr(tables)

    return cvn_em_calE
kEMEnergy = Var(kEMEnergy)

def kHadEnergy(tables):
    EMEnergy = kEMEnergy(tables)

    calE = tables['rec.slc']['calE']*kNueCalibrationCorr(tables)
   
    HadEnergy = calE - EMEnergy
    return HadEnergy.where(HadEnergy > 0, 0)
kHadEnergy = Var(kHadEnergy)

def kNueEnergy(tables):
    EMEnergy = kEMEnergy(tables)
    HadEnergy = kHadEnergy(tables)
    isRHC = kRHC(tables)

    p0 =  0.0
    p1 =  1.00756
    p2 =  1.07093
    p3 =  0.0
    p4 =  1.28608e-02
    p5 =  2.27129e-01
    norm = 0.0501206
    if isRHC.agg(np.all):
      p0 = 0.0
      p1 = 0.980479
      p2 = 1.45170
      p3 = 0.0
      p4 = -5.82609e-03
      p5 = -2.27599e-01
      norm = 0.001766
  
    NueEnergy = 1./(1+norm)*(HadEnergy*HadEnergy*p5 + \
                                     EMEnergy*EMEnergy*p4 +   \
                                     EMEnergy*HadEnergy*p3 +  \
                                     HadEnergy*p2 +              \
                                     EMEnergy*p1 + p0)
    return  NueEnergy.where((HadEnergy >= 0) & (EMEnergy >= 0), -5.)
kNueEnergy = Var(kNueEnergy)

###################################################################################
#
# Numu Vars
#
###################################################################################


def kCosNumi(tables):
    df = tables['rec.trk.kalman.tracks'][['dir.x','dir.y', 'dir.z']]
    # Primary kalman track only
    df = df.groupby(level=KL).first()
    CosNumi = pd.Series(np.zeros_like(df.shape[0]), index=df.index)

    # Use separate beam dir for each detector
    det = kDetID(tables)
    CosNumi[det == detector.kND] = df.mul(BeamDirND, axis=1).sum(axis=1)
    CosNumi[det == detector.kFD] = df.mul(BeamDirFD, axis=1).sum(axis=1)
    return CosNumi
kCosNumi = Var(kCosNumi)

def kNumuMuEND(tables):
  det = kDetID(tables)
  hdr_df = tables['rec.hdr'][['ismc']]
  hdr_df['run'] = hdr_df.index.get_level_values('run')
  runs = hdr_df['run']
  isRHC = kRHC(tables)

  ntracks = (tables['rec.trk.kalman']['ntracks'] != 0)
  
  trklenact = tables['rec.energy.numu']['ndtrklenact']/100.
  trklencat = tables['rec.energy.numu']['ndtrklencat']/100.
  trkcalactE = tables['rec.energy.numu']['ndtrkcalactE']
  trkcaltranE = tables['rec.energy.numu']['ndtrkcaltranE']

  df = pd.concat([runs, isRHC, trklenact, trklencat, det[det == detector.kND]], axis=1, join='inner')

  muE = pd.Series(kApplySpline(df['run'], detector.kND, df['isRHC'], 'act', df['ndtrklenact']) + \
                  kApplySpline(df['run'], detector.kND, df['isRHC'], 'cat', df['ndtrklencat']),
                  index=df.index)
  
  muE[(trkcalactE == 0.) & (trkcaltranE == 0.)] = -5.
  return muE.where(ntracks, -5.)
kNumuMuEND = Var(kNumuMuEND)

def kNumuMuEFD(tables):
  det = kDetID(tables)
  hdr_df = tables['rec.hdr']
  ismc = hdr_df['ismc']
  runs = hdr_df['run']
  isRHC = kRHC(tables)

  ntracks = (tables['rec.trk.kalman']['ntracks'] != 0)

  trklen = tables['rec.trk.kalman.tracks']['len']/100
  trklen[ismc == 1] *= 0.9957
  trklen = trklen.groupby(level=KL).first()

  df = pd.concat([runs, isRHC, trklen, det[det == detector.kFD]], axis=1, join='inner')

  muE = pd.Series(kApplySpline(df['run'], detector.kFD, df['isRHC'], 'muon', df['len']),
                  index=df.index)

  return muE.where(ntracks, -5.)
kNumuMuEFD = Var(kNumuMuEFD)

def kNumuMuE(tables):
  dfND = kNumuMuEND(tables)
  dfFD = kNumuMuEFD(tables)

  return pd.concat([dfND, dfFD])
kNumuMuE = Var(kNumuMuE)

def kNumuHadE(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)

  hdr_df = tables['rec.hdr']
  ismc = hdr_df['ismc']
  runs = hdr_df['run']
  
  hadvisE = tables['rec.energy.numu']['hadtrkE'] + tables['rec.energy.numu']['hadcalE']
  hadvisE.name = 'hadvisE'
  ntracks = (tables['rec.trk.kalman']['ntracks'] > 0)

  periods = pd.Series(GetPeriod(runs, det), index = det.index)
  hadvisE[(det == detector.kFD) & (ismc == 1) & (periods <= 2)] *= 0.9844
  hadvisE[(det == detector.kFD) & (ismc == 1) & (periods >  2)] *= 0.9949
  
  hadE = pd.Series(kApplySpline(runs, det, isRHC, 'had', hadvisE), 
                   index=det.index)
  return hadE.where(ntracks, -5.)
kNumuHadE = Var(kNumuHadE)

kNumuE = kNumuMuE + kNumuHadE
kCCE = kNumuE

###################################################################################
#
# Nus Vars
#
###################################################################################

kNusScaleFDFHC = 1.2
kNusScaleFDRHC = 1.18
kNusScaleNDFHC = 1.11
kNusScaleNDRHC = 1.15

def kNusEnergy(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)
  cale = kCaloE(tables).copy()
  cale[(isRHC == 1) & (det == detector.kFD)] *= kNusScaleFDRHC
  cale[(isRHC == 1) & (det == detector.kND)] *= kNusScaleNDRHC
  cale[(isRHC == 0) & (det == detector.kFD)] *= kNusScaleFDFHC
  cale[(isRHC == 0) & (det == detector.kND)] *= kNusScaleNDFHC
  return cale
kNusEnergy = Var(kNusEnergy)

kClosestSlcTime = Var(lambda tables: tables['rec.slc']['closestslicetime'])
kClosestSlcMinDist = Var(lambda tables: tables['rec.slc']['closestslicemindist'])
kClosestSlcMinTop = Var(lambda tables: tables['rec.slc']['closestsliceminfromtop'])
