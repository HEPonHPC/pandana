from PandAna.core.core import KL
from PandAna.weight.xsec_utils import *

import numpy as np
import pandas as pd

from PandAna.utils.enums import *

def kRescaleMAQE(tables, weight):
  correctionInSigma = (1.04 - 0.99) / 0.25
  
  df = tables['rec.mc.nu']

  genie_df = tables['rec.mc.nu.rwgt.genie']
  genie_plus1 = tables['rec.mc.nu.rwgt.genie']['plus1sigma']
  sel = (df['mode'] == mode.kQE) & \
        (df['iscc'] == 1) & \
        (genie_plus1.groupby(KL).agg('count') > genie.fReweightMaCCQE)
  genie_plus1 = genie_plus1[genie_df['rec.mc.nu.rwgt.genie_idx'] == genie.fReweightMaCCQE][sel]

  weight[sel] *= (1. + correctionInSigma*(genie_plus1 - 1.))
  return weight

def kFixNonres1Pi(tables, weight):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kDIS) & \
        (df['W2'] <=  1.7*1.7) & \
        (df['pdg'] >= 0) & \
        (df[['npiplus', 'npiminus', 'npizero']].sum(axis=1) == 1)
 
  # keeping typo in 2018 version for now. It should actually be 0.43 
  weight[sel] *= 0.41
  return weight

def kRescaleHighWDIS(tables, weight):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kDIS) & \
        (df['W2'] >= 1.7*1.7) & \
        (df['pdg'] >= 0)

  weight[sel] *= 1.1
  return weight

def kRPAWeightCCQE(tables, weight):
  rwgtCalc = RPAWeightCCQE_2017()
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kQE) & \
        (df['iscc'] == 1)
  
  q0 = (df['E']*df['y'])[sel]
  qmag = ((df['q2'] + (q0*q0)).pow(0.5))[sel]
  isAntiNu = (df['pdg'] < 0)[sel]
  params = pd.concat([q0, qmag, isAntiNu], axis=1)
  params.columns = ['q0', 'qmag', 'IsAntiNu']
  
  if not params.empty: 
    weight[sel] *= params.apply(lambda x: rwgtCalc.GetWeight(x), axis=1)
  return weight

def kRPAWeightRES(tables, weight):
  rwgtCalc = RPAWeightQ2_2017()
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kRes) & \
        (df['iscc'] == 1)

  q2 = df['q2'][sel]
  isAntiNu = (df['pdg'] < 0)[sel]
  params = pd.concat([q2, isAntiNu], axis=1)
  params.columns = ['q2', 'IsAntiNu']
  
  if not params.empty: 
    weight[sel] *= params.apply(lambda x: rwgtCalc.GetWeight(x), axis=1)
  return weight


def kEmpiricalMECWgt(tables, weight):
  rwgtCalc = EmpiricalMECWgt2018()
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kMEC)
  
  q0 = (df['E']*df['y'])[sel]
  qmag = ((df['q2'] + (q0*q0)).pow(0.5))[sel]
  isAntiNu = (df['pdg'] < 0)[sel]
  params = pd.concat([q0, qmag, isAntiNu], axis=1)
  params.columns = ['q0', 'qmag', 'IsAntiNu']
  
  if not params.empty: 
    weight[sel] *= params.apply(lambda x: rwgtCalc.GetWeight(x), axis=1)
  return weight

def kPPFXFluxCVWgt(tables, weight):
  sel = (tables['rec.mc']['nnu'] == 1)
  weight[sel] *= tables['rec.mc.nu.rwgt.ppfx']['cv']
  return weight

kXSecCVWgt2018 = [kRescaleMAQE, kRPAWeightCCQE, kRPAWeightRES, 
                  kFixNonres1Pi, kRescaleHighWDIS, kEmpiricalMECWgt]

kFluxCVWgt2018 = [kPPFXFluxCVWgt]

kCVWgt2018 = kXSecCVWgt2018 + kFluxCVWgt2018
