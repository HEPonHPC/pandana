class Enum():
  def __init__(self, labels, startfrom=0):
    self.labels = {}
    for index, label in enumerate(labels, startfrom):
      self.labels[label] = index
  def __getattr__(self, label):
    return self.labels[label]

genie = Enum([
    'kReweightNull', 

    # NCEL tweaking parameters:
    'fReweightMaNCEL',          # tweak Ma NCEL, affects dsigma(NCEL)/dQ2 both in shape and normalization
    'fReweightEtaNCEL',         # tweak NCEL strange axial form factor eta, affects dsigma(NCEL)/dQ2 both in shape and normalization
    # CCQE tweaking parameters:
    'fReweightNormCCQE',        # tweak CCQE normalization (energy independent)
    'fReweightNormCCQEenu',     # tweak CCQE normalization (maintains dependence on neutrino energy)
    'fReweightMaCCQEshape',     # tweak Ma CCQE, affects dsigma(CCQE)/dQ2 in shape only (normalized to constant integral)
    'fReweightMaCCQE',          # tweak Ma CCQE, affects dsigma(CCQE)/dQ2 both in shape and normalization
    'fReweightVecCCQEshape',    # tweak elastic nucleon form factors (BBA/default -> dipole) - shape only effect of dsigma(CCQE)/dQ2
    # Resonance neutrino-production tweaking parameters:
    'fReweightNormCCRES',       # tweak CCRES normalization
    'fReweightMaCCRESshape',    # tweak Ma CCRES, affects d2sigma(CCRES)/dWdQ2 in shape only (normalized to constant integral)
    'fReweightMvCCRESshape',    # tweak Mv CCRES, affects d2sigma(CCRES)/dWdQ2 in shape only (normalized to constant integral)
    'fReweightMaCCRES',         # tweak Ma CCRES, affects d2sigma(CCRES)/dWdQ2 both in shape and normalization
    'fReweightMvCCRES',         # tweak Mv CCRES, affects d2sigma(CCRES)/dWdQ2 both in shape and normalization
    
    'fReweightNormNCRES',       # tweak NCRES normalization
    'fReweightMaNCRESshape',    # tweak Ma NCRES, affects d2sigma(NCRES)/dWdQ2 in shape only (normalized to constant integral)
    'fReweightMvNCRESshape',    # tweak Mv NCRES, affects d2sigma(NCRES)/dWdQ2 in shape only (normalized to constant integral)
    'fReweightMaNCRES',         # tweak Ma NCRES, affects d2sigma(NCRES)/dWdQ2 both in shape and normalization
    'fReweightMvNCRES',         # tweak Mv NCRES, affects d2sigma(NCRES)/dWdQ2 both in shape and normalization
    
    # Coherent pion production tweaking parameters:
    'fReweightMaCOHpi',         # tweak Ma for COH pion production
    'fReweightR0COHpi',         # tweak R0 for COH pion production
    # Non-resonance background tweaking parameters:
    'fReweightRvpCC1pi',        # tweak the 1pi non-RES bkg in the RES region, for v+p CC
    'fReweightRvpCC2pi',        # tweak the 2pi non-RES bkg in the RES region, for v+p CC
    'fReweightRvpNC1pi',        # tweak the 1pi non-RES bkg in the RES region, for v+p NC
    'fReweightRvpNC2pi',        # tweak the 2pi non-RES bkg in the RES region, for v+p NC
    'fReweightRvnCC1pi',        # tweak the 1pi non-RES bkg in the RES region, for v+n CC
    'fReweightRvnCC2pi',        # tweak the 2pi non-RES bkg in the RES region, for v+n CC
    'fReweightRvnNC1pi',        # tweak the 1pi non-RES bkg in the RES region, for v+n NC
    'fReweightRvnNC2pi',        # tweak the 2pi non-RES bkg in the RES region, for v+n NC
    'fReweightRvbarpCC1pi',     # tweak the 1pi non-RES bkg in the RES region, for vbar+p CC
    'fReweightRvbarpCC2pi',     # tweak the 2pi non-RES bkg in the RES region, for vbar+p CC
    'fReweightRvbarpNC1pi',     # tweak the 1pi non-RES bkg in the RES region, for vbar+p NC
    'fReweightRvbarpNC2pi',     # tweak the 2pi non-RES bkg in the RES region, for vbar+p NC
    'fReweightRvbarnCC1pi',     # tweak the 1pi non-RES bkg in the RES region, for vbar+n CC
    'fReweightRvbarnCC2pi',     # tweak the 2pi non-RES bkg in the RES region, for vbar+n CC
    'fReweightRvbarnNC1pi',     # tweak the 1pi non-RES bkg in the RES region, for vbar+n NC
    'fReweightRvbarnNC2pi',     # tweak the 2pi non-RES bkg in the RES region, for vbar+n NC
    # DIS tweaking parameters - applied for DIS events with (Q2>Q2o, W>Wo), 
    # typically Q2okReweight =1GeV^2, WokReweight =1.7-2.0GeV
    'fReweightAhtBY',           # tweak the Bodek-Yang model parameter A_{ht} - incl. both shape and normalization effect
    'fReweightBhtBY',           # tweak the Bodek-Yang model parameter B_{ht} - incl. both shape and normalization effect
    'fReweightCV1uBY',          # tweak the Bodek-Yang model parameter CV1u - incl. both shape and normalization effect
    'fReweightCV2uBY',          # tweak the Bodek-Yang model parameter CV2u - incl. both shape and normalization effect
    'fReweightAhtBYshape',      # tweak the Bodek-Yang model parameter A_{ht} - shape only effect to d2sigma(DIS)/dxdy
    'fReweightBhtBYshape',      # tweak the Bodek-Yang model parameter B_{ht} - shape only effect to d2sigma(DIS)/dxdy
    'fReweightCV1uBYshape',     # tweak the Bodek-Yang model parameter CV1u - shape only effect to d2sigma(DIS)/dxdy
    'fReweightCV2uBYshape',     # tweak the Bodek-Yang model parameter CV2u - shape only effect to d2sigma(DIS)/dxdy
    'fReweightNormDISCC',       # tweak the inclusive DIS CC normalization (not currently working in genie)
    'fReweightRnubarnuCC',      # tweak the ratio of \sigma(\bar\nu CC) / \sigma(\nu CC) (not currently working in genie)
    'fReweightDISNuclMod',      # tweak DIS nuclear modification (shadowing, anti-shadowing, EMC).  Does not appear to be working in GENIE at the moment
    #
    'fReweightNC',              #
    
    
    #
    # Hadronization (free nucleon target)
    # 
    
    'fReweightAGKY_xF1pi',      # tweak xF distribution for low multiplicity (N + pi) DIS f/s produced by AGKY
    'fReweightAGKY_pT1pi',      # tweak pT distribution for low multiplicity (N + pi) DIS f/s produced by AGKY
    
    
    #
    # Medium-effects to hadronization
    # 
    
    'fReweightFormZone',        # tweak formation zone
    
    
    #
    # Intranuclear rescattering systematics.
    # There are 2 sets of parameters:
    # - parameters that control the total rescattering probability, P(total)
    # - parameters that control the fraction of each process (`fate'), given a total rescat. prob., P(fate|total)
    # These parameters are considered separately for pions and nucleons.
    #
    
    'fReweightMFP_pi',          # tweak mean free path for pions
    'fReweightMFP_N',           # tweak mean free path for nucleons
    'fReweightFrCEx_pi',        # tweak charge exchange probability for pions, for given total rescattering probability
    'fReweightFrElas_pi',       # tweak elastic   probability for pions, for given total rescattering probability
    'fReweightFrInel_pi',       # tweak inelastic probability for pions, for given total rescattering probability
    'fReweightFrAbs_pi',        # tweak absorption probability for pions, for given total rescattering probability
    'fReweightFrPiProd_pi',     # tweak pion production probability for pions, for given total rescattering probability
    'fReweightFrCEx_N',         # tweak charge exchange probability for nucleons, for given total rescattering probability
    'fReweightFrElas_N',        # tweak elastic    probability for nucleons, for given total rescattering probability
    'fReweightFrInel_N',        # tweak inelastic  probability for nucleons, for given total rescattering probability
    'fReweightFrAbs_N',         # tweak absorption probability for nucleons, for given total rescattering probability
    'fReweightFrPiProd_N',      # tweak pion production probability for nucleons, for given total rescattering probability
    
    #
    # Nuclear model
    # 
    
    'fReweightCCQEPauliSupViaKF',   #
    'fReweightCCQEMomDistroFGtoSF', #
    
    #
    # Resonance decays
    # 
    
    'fReweightBR1gamma',         # tweak Resonance -> X + gamma branching ratio, eg Delta+(1232) -> p gamma
    'fReweightBR1eta',           # tweak Resonance -> X + eta   branching ratio, eg N+(1440) -> p eta
    'fReweightTheta_Delta2Npi',  # distort pi angular distribution in Delta -> N + pi

    #
    # Alternative approach to CCQE form factors (z-expansion)
    #

    'fReweightZNormCCQE',        # tweak Z-expansion CCQE normalization (energy independent)
    'fReweightZExpA1CCQE',       # tweak Z-expansion coefficient 1, affects dsigma(CCQE)/dQ2 both in shape and normalization
    'fReweightZExpA2CCQE',       # tweak Z-expansion coefficient 2, affects dsigma(CCQE)/dQ2 both in shape and normalization
    'fReweightZExpA3CCQE',       # tweak Z-expansion coefficient 3, affects dsigma(CCQE)/dQ2 both in shape and normalization
    'fReweightZExpA4CCQE',       # tweak Z-expansion coefficient 4, affects dsigma(CCQE)/dQ2 both in shape and normalization
    'fReweightAxFFCCQEshape'     # tweak axial nucleon form factors (dipole -> z-expansion) - shape only effect of dsigma(CCQE)/dQ2
])

mode = Enum([
    'kUnknownMode',
    'kQE',
    'kRes',
    'kDIS',
    'kCoh',
    'kCohElastic',
    'kElectronScattering',
    'kIMDAnnihilation',
    'kInverseBetaDecay',
    'kGlashowResonance',
    'kAMNuGamma',
    'kMEC',
    'kDiffractive',
    'kEM',
    'kWeakMix'
], -1)

detector = Enum([
    'kUnknownDetector',
    'kND',
    'kFD'
])

trainint = Enum([
    'kNumuQE',
    'kNumuRes',
    'kNumuDIS',
    'kNumuOther',
    'kNueQE',
    'kNueRes',
    'kNueDIS',
    'kNueOther',
    'kNutauQE',
    'kNutauRes',
    'kNutauDIS',
    'kNutauOther',
    'kNuElectronElastic',
    'kNC',
    'kCosmic',
    'kOther',
    'kNIntType'
])
