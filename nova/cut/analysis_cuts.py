import pandas as pd
import numpy as np

from pandana.core import Cut
from pandana.core.indices import KL
from nova.utils.misc import *
from nova.utils.enums import *
from nova.var.analysis_vars import *

kIsFD = kDetID == detector.kFD

###################################################################################
#
# Basic Cuts
#
###################################################################################

# Basic cosrej
kVeto = Cut(lambda tables: tables['rec.sel.veto']['keep'] == 1)

# Basic Reco Cuts
kHasVtx  = Cut(lambda tables: tables['rec.vtx']['nelastic'] > 0)
kHasPng  = Cut(lambda tables: tables['rec.vtx.elastic.fuzzyk']['npng'] > 0)

###################################################################################
#
# Nue Cuts
#
###################################################################################

#Data Quality

def kDibMaskHelper(l):
    mask = l[0]

    fp = l[1]
    fpmin = fp
    fpmax = fp

    lp = l[2]
    lpmin = lp
    lpmax = lp

    for i in range(fp, 14, 1):
        if mask[13-i] == '0':
            break
        else:
            fpmax = i

    for i in range(fp, -1, -1):
        if mask[13-i] == '0':
            break
        else:
            fpmin = i

    for i in range(lp, 14, 1):
        if mask[13-i] == '0':
            break
        else:
            lpmax = i

    for i in range(lp, -1, -1):
        if mask[13-i] == '0':
            break
        else:
            lpmin = i
    return (fpmin==lpmin) & (fpmax==lpmax) & (lpmax-fpmin+1>=4)

def kNueApplyMask(tables):
    mask = tables['rec.hdr']['dibmask']
    fp = tables['rec.slc']['firstplane']//64
    lp = tables['rec.slc']['lastplane']//64
    df = mask.apply(lambda x: bin(x)[2:].zfill(14))
    df = pd.concat([df,fp,lp],axis=1)
    return df.apply(kDibMaskHelper, axis=1)
kNueApplyMask = Cut(kNueApplyMask)

kNueDQ = (kHitsPerPlane < 8) & kHasVtx & kHasPng

kNueBasicPart = kIsFD & kNueDQ & kVeto & kNueApplyMask

# Presel
kNuePresel = (kNueEnergy > 1) & (kNueEnergy < 4) & \
    (kNHit > 30) & (kNHit < 150) & \
    (kLongestProng > 100) & (kLongestProng < 500)

kNueProngContainment = (kDistAllTop > 63) & (kDistAllBottom > 12) & \
    (kDistAllEast > 12) & (kDistAllWest > 12) & \
    (kDistAllFront > 18) & (kDistAllBack > 18)

kNueBackwardCut = ((kDistAllBack < 200) & (kSparsenessAsymm < -0.1)) | (kDistAllBack >= 200)

kNuePtPCut = (kPtP < 0.58) | ((kPtP >= 0.58) & (kPtP < 0.8) & (kMaxY < 590)) | ((kPtP >= 0.8) & (kMaxY < 350))

kNueCorePart = kNuePresel & kNueProngContainment & kNuePtPCut & kNueBackwardCut

kNueCorePresel = kNueCorePart & kNueBasicPart

# PID Selections
kNueCVNFHC = 0.84
kNueCVNRHC = 0.89

def kNueCVNCut(tables):
    df = kCVNe(tables)
    dfRHC = df[kRHC(tables)==1] >= kNueCVNRHC
    dfFHC = df[kRHC(tables)!=1] >= kNueCVNFHC

    return pd.concat([dfRHC, dfFHC])
kNueCVNCut = Cut(kNueCVNCut)

# Full FD Selection
kNueFD = kNueCVNCut & kNueCorePresel

def kNueNDFiducial(tables):
    check = tables['rec.vtx.elastic']['rec.vtx.elastic_idx'] == 0 
    df = tables['rec.vtx.elastic'][check]
    return (df['vtx.x'] > -100) & \
        (df['vtx.x'] < 160) & \
        (df['vtx.y'] > -160) & \
        (df['vtx.y'] < 100) & \
        (df['vtx.z'] > 150) & \
        (df['vtx.z'] < 900)
kNueNDFiducial = Cut(kNueNDFiducial)

def kNueNDContain(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    df_trans = df[['start.y','stop.y', 'start.x', 'stop.x']]
    df_long = df[['start.z', 'stop.z']]

    return ((df_trans.min(axis=1) > -170) & (df_trans.max(axis=1) < 170) & \
            (df_long.min(axis=1) > 100) & (df_long.max(axis=1) < 1225)).groupby(level=KL).agg(np.all)
kNueNDContain = Cut(kNueNDContain)

kNueNDFrontPlanes = Cut(lambda tables: tables['rec.sel.contain']['nplanestofront'] > 6)

kNueNDNHits = (kNHit >= 20) & (kNHit <= 200)

kNueNDEnergy = (kNueEnergy < 4.5)

kNueNDProngLength = (kLongestProng > 100) & (kLongestProng < 500)

kNueNDPresel = kNueDQ & kNueNDFiducial & kNueNDContain & kNueNDFrontPlanes & \
               kNueNDNHits & kNueNDEnergy & kNueNDProngLength

kNueNDCVNSsb = kNueNDPresel & kNueCVNCut



###################################################################################
#
# Numu Cuts
#
###################################################################################

def kNumuBasicQuality(tables):
    df_numutrkcce=tables['rec.energy.numu']['trkccE']
    df_remid=tables['rec.sel.remid']['pid']
    df_nhit=tables['rec.slc']['nhit']
    df_ncontplanes=tables['rec.slc']['ncontplanes']
    df_cosmicntracks=tables['rec.trk.cosmic']['ntracks']
    return(df_numutrkcce > 0) &\
               (df_remid > 0) &\
               (df_nhit > 20) &\
               (df_ncontplanes > 4) &\
               (df_cosmicntracks > 0)
kNumuBasicQuality = Cut(kNumuBasicQuality)

kNumuQuality = kNumuBasicQuality & (kCCE < 5.)

# FD 

kNumuProngsContainFD = (kDistAllTop > 60) & (kDistAllBottom > 12) & (kDistAllEast > 16) & \
                            (kDistAllWest > 12)  & (kDistAllFront > 18) & (kDistAllBack > 18)

def kNumuDibMaskHelper(l):
    mask = l[0]
    
    fd = l[1]//64
    ld = l[2]//64

    dmin = 0
    dmax = 13

    for i in range(fd, 14, 1):
        if mask[13-i] == '0':
            break
        else:
            dmax = i

    for i in range(fd, -1, -1):
        if mask[13-i] == '0':
            break
        else:
            dmin = i

    return ((l[1]-64*dmin) > 1) & ((64*(dmax+1)-l[2]-1) > 1) 

def kNumuOptimizedContainFD(tables):
    mask = tables['rec.hdr']['dibmask']
    fp = tables['rec.slc']['firstplane']
    lp = tables['rec.slc']['lastplane']
    df = mask.apply(lambda x: bin(x)[2:].zfill(14))
    df = pd.concat([df,fp,lp],axis=1)
    df = df.apply(kNumuDibMaskHelper, axis=1, result_type='reduce')

    df_containkalfwdcell = tables['rec.sel.contain']['kalfwdcell'] > 6
    df_containkalbakcell = tables['rec.sel.contain']['kalbakcell'] > 6
    df_containcosfwdcell = tables['rec.sel.contain']['cosfwdcell'] > 0 
    df_containcosbakcell = tables['rec.sel.contain']['cosbakcell'] > 7

    return df & df_containkalfwdcell & df_containkalbakcell & \
        df_containcosfwdcell & df_containkalbakcell
kNumuOptimizedContainFD = Cut(kNumuOptimizedContainFD)

kNumuContainFD = kNumuProngsContainFD & kNumuOptimizedContainFD 

kNumuNoPIDFD = kNumuQuality & kNumuContainFD

# ND
def kNumuContainND(tables):
    # check is a pandas.core.series.Series.
    # it will have a MultiIndex with names 'run', 'subrun', 'cycle', 'evt' and 'subevt'.
    check = tables['rec.vtx.elastic.fuzzyk.png']['rec.vtx.elastic_idx'] == 0

    shw_df = tables['rec.vtx.elastic.fuzzyk.png.shwlid'][check]
    shw_df_trans = shw_df[['start.y','stop.y', 'start.x', 'stop.x']]
    shw_df_long = shw_df[['start.z', 'stop.z']]
    no_shw = (tables['rec.vtx.elastic.fuzzyk']['nshwlid'] == 0)

    shw_contain = ((shw_df_trans.min(axis=1) >= -180.) & (shw_df_trans.max(axis=1) <= 180.) & \
             (shw_df_long.min(axis=1) >= 20.) & (shw_df_long.max(axis=1) <= 1525.)).groupby(level=KL).agg(np.all)
    shw_contain = (shw_contain | no_shw)

    trk_df = tables['rec.trk.kalman.tracks'][['start.z', 'stop.z', 'rec.trk.kalman.tracks_idx']]
    kalman_contain = (trk_df['rec.trk.kalman.tracks_idx'] == 0) | ((trk_df['start.z'] <= 1275) & (trk_df['stop.z'] <= 1275))
    kalman_contain = kalman_contain.groupby(level=KL).agg(np.all)

    df_ntracks = tables['rec.trk.kalman']['ntracks']
    df_remid = tables['rec.trk.kalman']['idxremid']
    df_firstplane = tables['rec.slc']['firstplane']
    df_lastplane = tables['rec.slc']['lastplane']
    
    first_trk = trk_df['rec.trk.kalman.tracks_idx'] == 0
    df_startz = trk_df[first_trk]['start.z']
    df_stopz  = trk_df[first_trk]['stop.z']
    
    df_containkalposttrans = tables['rec.sel.contain']['kalyposattrans']
    df_containkalfwdcellnd = tables['rec.sel.contain']['kalfwdcellnd']
    df_containkalbakcellnd = tables['rec.sel.contain']['kalbakcellnd']
    
    return (df_ntracks > df_remid) &\
           (df_firstplane > 1) &\
           (df_lastplane < 212) &\
           (df_containkalfwdcellnd > 5) &\
           (df_containkalbakcellnd > 10) &\
           (df_startz < 1100 ) & (( df_containkalposttrans < 55) | (df_stopz < 1275) ) &\
           shw_contain &\
           kalman_contain

kNumuContainND = Cut(kNumuContainND)

kNumuNCRej = Cut(lambda tables: tables['rec.sel.remid']['pid'] > 0.75)

kNumuNoPIDND = kNumuQuality & kNumuContainND

def kNumuPID(tables):
    return (tables['rec.sel.remid']['pid'] > 0.7) &\
           (tables['rec.sel.cvnProd3Train']['numuid'] > 0.7) &\
           (tables['rec.sel.cvn2017']['numuid'] > 0.1)
kNumuPID = Cut(kNumuPID)

kNumuCutND = kNumuQuality & kNumuContainND & kNumuPID

###################################################################################
#
# Nus Cuts
#
###################################################################################


# FD 

kNusFDContain = (kDistAllTop > 100) & (kDistAllBottom > 10) & \
    (kDistAllEast > 50) & (kDistAllWest > 50) & \
    (kDistAllFront > 50) & (kDistAllBack > 50)

kNusContPlanes = Cut(lambda tables: tables['rec.slc']['ncontplanes'] > 2)

kNusEventQuality = kHasVtx & kHasPng & \
                  (kHitsPerPlane < 8) & kNusContPlanes

kNusFDPresel = kNueApplyMask & kVeto & kNusEventQuality & kNusFDContain

kNusBackwardCut = ((kDistAllBack < 200) & (kSparsenessAsymm < -0.1)) | (kDistAllBack >= 200)

kNusEnergyCut = (kNusEnergy >= 0.5) & (kNusEnergy <= 20.)

kNusSlcTimeGap = (kClosestSlcTime > -150.) & (kClosestSlcTime < 50.)
kNusSlcDist = (kClosestSlcMinTop < 100) & (kClosestSlcMinDist < 500)
kNusShwPtp = ((kMaxY > 580) & (kPtP > 0.2)) | ((kMaxY > 540) & (kPtP > 0.4))

# Nus Cosrej Cuts use TMVA trained BDT natively
kNusNoPIDFD = (kNusFDPresel & kNusBackwardCut) & (~(kNusSlcTimeGap & kNusSlcDist)) & \
              (~kNusShwPtp) & kNusEnergyCut


# ND 

def kNusNDFiducial(tables):
    check = tables['rec.vtx.elastic']['rec.vtx.elastic_idx'] == 0 
    df = tables['rec.vtx.elastic'][check]
    return (df['vtx.x'] > -100) & \
        (df['vtx.x'] < 100) & \
        (df['vtx.y'] > -100) & \
        (df['vtx.y'] < 100) & \
        (df['vtx.z'] > 150) & \
        (df['vtx.z'] < 1000)
kNusNDFiducial = Cut(kNusNDFiducial)

kNusNDContain = (kDistAllTop > 25) & (kDistAllBottom > 25) & \
    (kDistAllEast > 25) & (kDistAllWest > 25) & \
    (kDistAllFront > 25) & (kDistAllBack > 25)

kNusNDPresel = kNusEventQuality & kNusNDFiducial & kNusNDContain
kNusNoPIDND = kNusNDPresel & (kPtP <= 0.8) & kNusEnergyCut
