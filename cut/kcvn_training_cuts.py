from PandAna.core.core import *
from PandAna.var.kcvn_training_vars import *

#To change Purity cuts easily:

MuonPurityValue = 0.5
ElectronPurityValue = 0.4
ProtonPurityValue = 0.35
PionPurityValue = 0.35
GammaPurityValue = 0.5
NeutronPurityValue = 0.5
PiZeroPurityValue = 0.5


#Prong Based Cuts

kMuonPurCut = (((kPngPDG == 13) | (kPngPDG == -13)) & (kPngPur >= MuonPurityValue) & (kPngLabel == 1))
kElectronPurCut = (((kPngPDG == 11) | (kPngPDG == -11)) & (kPngPur >= ElectronPurityValue) & (kPngLabel == 0))
kProtonPurCut = (((kPngPDG == 2212) | (kPngPDG == -2212)) & (kPngPur >= ProtonPurityValue) & (kPngLabel == 2))
kPionPurCut = (((kPngPDG == 211) | (kPngPDG == -211)) & (kPngPur >= PionPurityValue) & (kPngLabel == 4))
kGammaPurCut = (((kPngPDG == 22) | (kPngPDG == -22)) & (kPngPur >= GammaPurityValue) & (kPngLabel == 6))
kNeutronPurCut = (((kPngPDG == 2112) | (kPngPDG == -2112)) & (kPngPur >= NeutronPurityValue) & (kPngLabel == 3))
kPiZeroPurCut = (((kPngPDG == 111) | (kPngPDG == -111)) & (kPngPur >= PiZeroPurityValue) & (kPngLabel == 5))

kPurityCuts = kMuonPurCut | kElectronPurCut | kProtonPurCut | kPionPurCut | kGammaPurCut | kNeutronPurCut | kPiZeroPurCut

kProngHitXCut = kPngNCellX > 2
kProngHitYCut = kPngNCellY > 2

kAllProngCuts = kPurityCuts & kProngHitYCut & kProngHitXCut

#Slice Based Cuts

kInterCut = kSliceInter > -5

kCosmicCut = kSliceInter >-1000

kAllSliceCuts = kVeto & kInterCut & kCosmicCut
