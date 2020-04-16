import numpy as np

from pandana.utils.enums import *
from nova.utils.misc import GetPeriod

# For the Numu Energy Estimator
class SplineFit():
  def __init__(self, spline_spec):
    self.spline = spline_spec
    self.nstitch = len(self.spline)//2 - 1

    self.slopes = self.spline[3::2]
    self.x0 = self.spline[2::2]

    self.intercepts = [0]*self.nstitch
    self.InitIntercept()

  def InitIntercept(self):
    prev_intercept = self.spline[0]
    prev_slope = self.spline[1]
    for i in range(self.nstitch):
      self.intercepts[i] = prev_intercept + (prev_slope-self.slopes[i])*self.x0[i]
      prev_intercept = self.intercepts[i]
      prev_slope = self.slopes[i]
  
  def __call__(self, var):
    if var <= 0.:
      return 0.
    stitchpos = np.where([i <= var for i in self.x0])[0]
    if len(stitchpos):
      return self.slopes[stitchpos[-1]]*var + self.intercepts[stitchpos[-1]]
    else:
      return self.spline[1]*var + self.spline[0]
    
kSplineProd4MuonFDp1 = SplineFit([
  1.264631942673353215e-01, 2.027211173807817457e-01,
  8.774231219753678701e+00, 2.163481314097564778e-01
])

kSplineProd4HadFDp1 = SplineFit([
  7.281506207958776677e-02, 5.441509444740477708e-01,
  5.995673938377335532e-02, 2.016769249502490702e+00
])

kSplineProd4MuonFDp2 = SplineFit([
  1.257629394848571724e-01, 2.036422681279472513e-01,
  9.855445529033460161e+00, 2.195160029549789171e-01
])

kSplineProd4HadFDp2 = SplineFit([
  5.877366209096868133e-02, 1.519704739772891111e+00,
  7.898381593661769895e-02, 2.169042213902584670e+00,
  4.771203681401176011e-01, 1.694250867119319715e+00,
  7.539010791835750736e-01, 2.059894991370703199e+00
])

kSplineProd4MuonFDp3 = SplineFit([
  1.258258375981389232e-01, 2.033924315007327455e-01,
  9.466540687161570489e+00, 2.184781804887378220e-01
])

kSplineProd4HadFDp3 = SplineFit([
  5.673223776636682203e-02, 1.465342397469045377e+00,
  8.044448532774936544e-02, 2.106105448794447277e+00,
  4.350000000342279516e-01, 1.829072531499642107e+00
])

kSplineProd4MuonFDp4 = SplineFit([
  1.463359246892861343e-01, 1.972096058268091312e-01,
  4.248366183914889405e+00, 2.072087518392393413e-01,
  1.048696659085294414e+01, 2.187994552957214234e-01
])

kSplineProd4HadFDp4 = SplineFit([
  4.249458417893325901e-02, 2.193901299460967902e+00,
  1.049999999339812778e-01, 1.981193414403220387e+00,
  4.492042726487709969e-01, 1.540708309857498071e+00,
  6.193806271369780569e-01, 2.032400170863909228e+00
])

kSplineProd4MuonFDp5 = SplineFit([
  1.303357512360951986e-01, 2.024876303943876354e-01,
  9.173839619568841641e+00, 2.184865210120549850e-01
])

kSplineProd4HadFDp5 = SplineFit([
  5.513045392968107805e-02, 1.534780212240209885e+00,
  8.119825924952998875e-02, 2.102191592086820382e+00,
  4.829923244287704365e-01, 1.655956208512852301e+00,
  7.849999988138458562e-01, 2.030853659350569718e+00
])

kSplineProd4ActNDp3 = SplineFit([
  1.514900417035001112e-01, 1.941290659171270860e-01,
  3.285152850349305265e+00, 2.027000969328388302e-01,
  5.768910882104949955e+00, 2.089391482903513730e-01
])

kSplineProd4CatNDp3 = SplineFit([
  8.720084706388187001e-03, 5.529278858789209439e-01,
  2.270042802448197783e+00, 1.711916184621219417e+00,
  2.307938644096652947e+00, 3.521795029684806622e-01
])

kSplineProd4HadNDp3 = SplineFit([
  6.390888289071572359e-02, 1.416435376103045707e+00,
  4.714208143879788232e-02, 2.144449735801436052e+00,
  2.402894748598130015e-01, 2.453526979604870206e+00,
  4.581240238035835244e-01, 1.730599464528853160e+00
])

kSplineProd4ActNDp4 = SplineFit([
  1.534847032701298630e-01, 1.939572586752992267e-01,
  3.076153165048225446e+00, 2.016367496419133321e-01,
  5.139959732881990817e+00, 2.085728450137292189e-01
])

kSplineProd4CatNDp4 = SplineFit([
  6.440384628800144284e-02, 1.663238457993588826e-01,
  1.466666666666678887e-01, 5.572409982576314036e-01,
  2.270441314113081255e+00, 2.043488369726459641e+00,
  2.299237514383191794e+00, 3.673229067047741880e-01
])

kSplineProd4HadNDp4 = SplineFit([
  4.743190085416526536e-02, 2.510494758751641520e+00,
  9.795449140279462175e-02, 1.981710272551915564e+00
])

def GetSpline(run, det, isRHC, comp):
  period = GetPeriod(run, det)
  if (period==1) and (det == detector.kFD):
    return {"muon" : kSplineProd4MuonFDp1, "had" : kSplineProd4HadFDp1}[comp]
  if (period==2) and (det == detector.kFD):
    return {"muon" : kSplineProd4MuonFDp2, "had" : kSplineProd4HadFDp2}[comp]
  if (period==3) and (det == detector.kFD):
    return {"muon" : kSplineProd4MuonFDp3, "had" : kSplineProd4HadFDp3}[comp]
  if (period>=4) and (det == detector.kFD) and isRHC:
    return {"muon" : kSplineProd4MuonFDp4, "had" : kSplineProd4HadFDp4}[comp]
  if (period>=5) and (det == detector.kFD) and not isRHC:
    return {"muon" : kSplineProd4MuonFDp5, "had" : kSplineProd4HadFDp5}[comp]
  if (det == detector.kND) and isRHC:
    return {"act" : kSplineProd4ActNDp4, "cat": kSplineProd4CatNDp4, "had": kSplineProd4HadNDp4}[comp]
  if (det == detector.kND) and not isRHC:
    return {"act" : kSplineProd4ActNDp3, "cat": kSplineProd4CatNDp3, "had": kSplineProd4HadNDp3}[comp]
