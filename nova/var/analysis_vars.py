import numpy as np
import pandas as pd
from pandana.core.indices import KL
from pandana.core.var import Var
from pandana.utils import *

from nova.utils.misc import *
from nova.var.numuE_utils import *

# cvnProd3Train is messed up in the current iteration of files.
kCVNe = Var(lambda tables: tables['rec.sel.cvn2017']['nueid'])
kCVNm = Var(lambda tables: tables['rec.sel.cvn2017']['numuid'])
kCVNnc = Var(lambda tables: tables['rec.sel.cvn2017']['ncid'])

kNHit = Var(lambda tables: tables['rec.slc']['nhit'])

kRHC   = Var(lambda tables: tables['rec.spill']['isRHC'])
kDetID = Var(lambda tables: tables['rec.hdr']['det'])

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

def kNueCalibrationCorr(det, ismc, run):
  if (det != detector.kFD): return 1.
  if not ismc: return 0.9949
  if run < 20753: return 0.9949/0.9844
  return 1.

def kEMEnergy(tables):
    lng_png = kLongestProng(tables)
    isRHC = kRHC(tables)

    shwlid_df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    prim_png = shwlid_df['calE'][(shwlid_df['rec.vtx.elastic.fuzzyk.png_idx']==0)]

  
    png_df = tables['rec.vtx.elastic.fuzzyk.png']
    cvn_png_df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart']
    
    cvn_em_pid_df = cvn_png_df[['photonid',  \
                                'pizeroid',  \
                                'electronid']].sum(axis=1)
    cvn_had_pid_df = cvn_png_df[['pionid',   \
                                 'protonid', \
                                 'neutronid',\
                                 'otherid',  \
                                 'muonid']].sum(axis=1)

    cvn_em_calE = shwlid_df['calE'].where(                     \
                            (cvn_em_pid_df > 0) &              \
                            (cvn_em_pid_df >= cvn_had_pid_df), \
                            0).groupby(level=KL).agg(np.sum)

    if isRHC.agg(np.all):
      cvn_em_calE[cvn_em_calE == 0] = prim_png[cvn_em_calE == 0]
    else:
      cvn_em_calE[lng_png >= 500] = prim_png[lng_png >= 500]

    hdr_df = tables['rec.hdr']
    det = hdr_df['det']
    ismc = tables['rec.hdr']['ismc']
    runs = hdr_df.assign(run=hdr_df.index.get_level_values('run'))['run']
    df = pd.concat([runs, det, ismc], axis=1).dropna()
    scale = df.apply(lambda x: kNueCalibrationCorr(x['det'], x['ismc'], x['run']), axis=1, result_type='reduce')

    cvn_em_calE *= scale
    cvn_em_calE.name = 'calE'

    return cvn_em_calE
kEMEnergy = Var(kEMEnergy)

def kHadEnergy(tables):
    EMEnergy = kEMEnergy(tables)
    
    hdr_df = tables['rec.hdr']
    det = hdr_df['det']
    ismc = tables['rec.hdr']['ismc']
    runs = hdr_df.assign(run=hdr_df.index.get_level_values('run'))['run']
    df = pd.concat([runs, det, ismc], axis=1).dropna()
    scale = df.apply(lambda x: kNueCalibrationCorr(x['det'], x['ismc'], x['run']), axis=1, result_type='reduce')
   
    calE = tables['rec.slc']['calE']*scale
    calE.name = 'calE'
   
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
    df = tables['rec.trk.kalman.tracks']
    # Primary kalman track only
    df = df[df['rec.trk.kalman.tracks_idx']==0]
    KalDir = df[['dir.x','dir.y', 'dir.z']]

    # Use separate beam dir for each detector
    det = kDetID(tables)
    if (det == detector.kFD).agg(np.all) and not det.empty:
        CosFD = KalDir.mul(BeamDirFD, axis=1).sum(axis=1)
        return CosFD
    if (det == detector.kND).agg(np.all) and not det.empty:
        CosND = KalDir.mul(BeamDirND, axis=1).sum(axis=1)
        return CosND
kCosNumi = Var(kCosNumi)

def kNumuMuE(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)

  hdr_df = tables['rec.hdr']
  runs = hdr_df.assign(run=hdr_df.index.get_level_values('run'))['run']
  ntracks = (tables['rec.trk.kalman']['ntracks'] != 0)
  
  if (det == detector.kFD).agg(np.all) and not det.empty:
    ismc = tables['rec.hdr']['ismc']
    
    trks = tables['rec.trk.kalman.tracks']
    trklen = trks['len'][(trks['rec.trk.kalman.tracks_idx'] == 0)]/100
    if not ismc.agg(np.all) and not ismc.empty:
      trklen = trklen*0.9957
    
    df = pd.concat([runs, det, isRHC, trklen], axis=1).dropna()
    muE = df.apply(lambda x: \
                   GetSpline(x[0], x[1], x[2], "muon")(x[3]), axis = 1)
    return muE.where(ntracks, -5.)
  
  else:
    trklenact = tables['rec.energy.numu']['ndtrklenact']/100.
    trklencat = tables['rec.energy.numu']['ndtrklencat']/100.
    trkcalactE = tables['rec.energy.numu']['ndtrkcalactE']
    trkcaltranE = tables['rec.energy.numu']['ndtrkcaltranE']

    muE = pd.Series(kApplySpline(runs, det, isRHC, 'act', trklenact) + kApplySpline(runs, det, isRHC, 'cat', trklencat),
                    index=det.index)
    muE[(trkcalactE == 0.) & (trkcaltranE == 0.)] = -5.
    return muE.where(ntracks, -5.)
kNumuMuE = Var(kNumuMuE)

def kNumuHadE(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)

  hdr_df = tables['rec.hdr']
  runs = hdr_df.assign(run=hdr_df.index.get_level_values('run'))['run']
  
  hadvisE = tables['rec.energy.numu']['hadtrkE'] + tables['rec.energy.numu']['hadcalE']
  hadvisE.name = 'hadvisE'
  ntracks = (tables['rec.trk.kalman']['ntracks'] != 0)
  
  if (det == detector.kFD).agg(np.all) and not det.empty:
    ismc = tables['rec.hdr']['ismc']
    if not ismc.agg(np.all) and not ismc.empty:
      periods = runs.apply(lambda x: GetPeriod(x))
      hadvisE[periods > 2] = 0.9949*hadvisE[periods > 2]
      hadvisE[periods <= 2] = 0.9844*hadvisE[periods <= 2]
  
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

def NusScale(isRHC, det):
  if (isRHC and det == detector.kFD): return 1.18
  if (isRHC and det == detector.kND): return 1.15
  if (not isRHC and det == detector.kFD): return 1.2
  if (not isRHC and det == detector.kND): return 1.11
  else: return -5.

def kNusEnergy(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)
  cale = kCaloE(tables)
  df = pd.concat([det, isRHC, cale], axis=1)
  return df.apply(lambda x: \
                 NusScale(x['isRHC'], x['det'])*x['calE'], axis = 1)
kNusEnergy = Var(kNusEnergy)

kClosestSlcTime = Var(lambda tables: tables['rec.slc']['closestslicetime'])
kClosestSlcMinDist = Var(lambda tables: tables['rec.slc']['closestslicemindist'])
kClosestSlcMinTop = Var(lambda tables: tables['rec.slc']['closestsliceminfromtop'])
