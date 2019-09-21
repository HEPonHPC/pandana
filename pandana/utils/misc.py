import numpy as np
import os
from pandana.utils.enums import *
import sys

# Taken from https://cdcvs.fnal.gov/redmine/projects/novaart/repository/entry/trunk/CAFAna/Core/Utilities.cxx
BeamDirFD = np.array([-6.83271078e-05,  6.38772962e-02,  9.97957758e-01])
BeamDirND = np.array([-8.42393199e-04, -6.17395015e-02,  9.98091942e-01])

def FindpandanaDir(folder=None):
  pandana = os.environ['FW_RELEASE_BASE']+'/pandana'
  if folder:
    pandana = folder+'/pandana'
  assert os.path.isdir(pandana), "Cannot find directory : {}".format(pandana) 
  return pandana

def GetPeriod(run, det):
  if (det == detector.kFD):
    if (run <= 12941): return 0
    if (run <= 17139): return 1
    if (run <= 20752): return 2
    if (run <= 23419): return 3
    if (run <= 24613): return 4
    if (run <= 25412): return 5
    if (run <= 28036): return 6
    else: return 7
  if (det == detector.kND):
    if (run <  10377): return 0
    if (run <= 10407): return 1
    if (run <= 11228): return 2
    if (run <= 11628): return 3
    if (run <= 11925): return 4
    if (run <= 12086): return 5
    if (run <= 12516): return 6
    else: return 7

# stop all my pandana projects
def StopAllUserProjects():
  user = os.getenv('USER')
  if os.getenv('_CONDOR_SCRATCH_DIR'):
    print("Can only stop projects interactively")
    sys.exit(2)
  import samweb_client
  SAM = samweb_client.SAMWebClient(experiment='nova')
  projlist = SAM.listProjects(user=user, state='running')
  for proj in projlist:
    if (user+"_pandana_proj") in proj:
      SAM.stopProject(proj)
  return
