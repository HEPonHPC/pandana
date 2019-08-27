import pandas as pd
import numpy as np
import h5py

from PandAna.utils import *

class NuWeightDFWrapper():
  def __init__(self, df):
    self.df = df
    self.dim = (int)(len(self.df.columns)-1)/2

    self.xlow = df['xlow'].unique()
    self.xhigh = df['xhigh'].unique()
    if self.dim > 1:
      self.ylow = df['ylow'].unique()
      self.yhigh = df['yhigh'].unique()
  
  def GetNbinsX(self):
    nxlow = len(self.xlow)
    nxhigh = len(self.xhigh)
    assert nxlow == nxhigh
    return nxhigh
  
  def GetNbinsY(self):
    assert self.dim > 1
    nylow = len(self.ylow)
    nyhigh = len(self.yhigh)
    assert nylow == nyhigh
    return nyhigh

  def FindXBin(self, xval):
    if min(self.xlow) > xval:
      return 0
    if max(self.xhigh) <= xval:
      return self.GetNbinsX()+1
    else:
      xbin_arr = np.where((self.xlow <= xval) & (self.xhigh > xval))[0]
      assert len(xbin_arr) == 1
      return xbin_arr[0]+1
  
  def FindYBin(self, yval):
    assert self.dim > 1
    if min(self.ylow) > yval:
      return 0
    if max(self.yhigh) <= yval:
      return self.GetNbinsY()+1
    else:
      ybin_arr = np.where((self.ylow <= yval) & (self.yhigh > yval))[0]
      assert len(ybin_arr) == 1
      return ybin_arr[0]+1

  def GetBinContent(self, binx, biny=-1):
    if biny < 0: 
      assert self.dim == 1
      if binx < 1 or binx > self.GetNbinsX(): return 1.
      return self.df['weight'][binx-1]
    else:
      assert self.dim > 1
      if binx < 1 or biny < 1: return 1.
      if binx > self.GetNbinsX() or biny > self.GetNbinsY(): return 1.
      binidx = (binx-1)*self.GetNbinsY() + (biny-1)
      return self.df['weight'][binidx]
  
  def FindFirstBinAbove(self, threshold, axis):
    assert (axis == 1) or (axis < 3 and self.dim == 2)
    firstbin = self.df[self.df['weight'] > threshold].index[0]
    return (axis-1)*((firstbin % self.GetNbinsY())+1) + (2-axis)*((firstbin/self.GetNbinsX())+1)

  def GetValue(self, val):
    assert self.dim == 1
    return self.GetBinContent(self.FindXBin(val))

  def GetValueInRange(self, vals, maxrange=[-float("inf"), float("inf")], binranges=[[1,-1], [1,-1]]):
    assert self.dim == 2
    if binranges[0][1] < 0: binranges[0][1] = self.GetNbinsX()
    if binranges[1][1] < 0: binranges[1][1] = self.GetNbinsY()
    
    binx = self.FindXBin(vals[0])
    biny = self.FindYBin(vals[1])
    if binx > binranges[0][1]: binx = binranges[0][1]
    elif binx < binranges[0][0]: binx = binranges[0][0]
    if biny > binranges[1][1]: biny = binranges[1][1]
    elif biny < binranges[1][0]: biny = binranges[1][0]

    val = self.GetBinContent(binx, biny)
    if val < maxrange[0]: val = maxrange[0]
    if val > maxrange[1]: val = maxrange[1]
    return val
    
class NuWeightFromFile():
  def __init__(self, fnu, fnubar, forcenu=False):
    fnuh5 = h5py.File(fnu['file'], 'r')
    self.nu = NuWeightDFWrapper(
                pd.DataFrame(fnuh5.get(fnu['group']+'/block0_values')[()],
                columns=fnuh5.get(fnu['group']+'/block0_items')[()]))
    if not forcenu:
      fnubarh5 = h5py.File(fnubar['file'], 'r')
      self.nubar = NuWeightDFWrapper(
                    pd.DataFrame(fnubarh5.get(fnubar['group']+'/block0_values')[()],
                    columns=fnubarh5.get(fnubar['group']+'/block0_items')[()]))
      assert self.nu.dim == self.nubar.dim
    self.dim = self.nu.dim

  def GetWeight(self):
    return 1.

class RPAWeightCCQE_2017(NuWeightFromFile):
  def __init__(self):
    fnu = {}
    fnubar = {}
    fnu['file'] = FindPandAnaDir()+"/Data/xs/RPA2017.h5"
    fnubar['file'] = fnu['file']
    fnu['group'] = "RPA_CV_nu"
    fnubar['group'] = "RPA_CV_nubar"
    NuWeightFromFile.__init__(self, fnu, fnubar)

  def GetWeight(self, params):
    qmag = params['qmag']
    q0 = params['q0']
    isAntiNu = ('IsAntiNu' in params.keys()) and params['IsAntiNu']

    df = self.nu
    if isAntiNu: df = self.nubar
    minbin = df.FindFirstBinAbove(0, 2)
    val = df.GetValueInRange([qmag, q0],
                             [0., 2.],
                             [[1, df.GetNbinsX()],[minbin,df.GetNbinsY()]])
    if val == 0.: val = 1.
    return val

class RPAWeightQ2_2017(NuWeightFromFile):
  def __init__(self):
    fnu = {}
    fnubar = {}
    fnu['file'] = FindPandAnaDir()+"/Data/xs/RPA2017.h5"
    fnubar['file'] = fnu['file']
    fnu['group'] = "RPA_Q2_CV_nu"
    fnubar['group'] = "RPA_Q2_CV_nubar"
    NuWeightFromFile.__init__(self, fnu, fnubar)

  def GetWeight(self, params):
    q2 = params['q2']
    isAntiNu = ('IsAntiNu' in params.keys()) and params['IsAntiNu']
    
    df = self.nu
    if isAntiNu: df = self.nubar
    return df.GetValue(q2)

class EmpiricalMECWgt2018(NuWeightFromFile):
  def __init__(self):
    fnu = {}
    fnubar = {}
    fnu['file'] = FindPandAnaDir()+"/Data/xs/rw_empiricalMEC2018.h5"
    fnubar['file'] = fnu['file']
    fnu['group'] = "numu_mec_weights_smoothed"
    fnubar['group'] = "numubar_mec_weights_smoothed"
    NuWeightFromFile.__init__(self, fnu, fnubar)

  def GetWeight(self, params):
    qmag = params['qmag']
    q0 = params['q0']
    isAntiNu = ('IsAntiNu' in params.keys()) and params['IsAntiNu']

    df = self.nu
    if isAntiNu: df = self.nubar
    val = df.GetValueInRange([qmag, q0])
    if val < 0.: val = 0.
    return val
