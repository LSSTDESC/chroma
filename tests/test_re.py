import numpy as np
import lmfit

import _mypath
import chroma

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.0)
    gparam.add('y0', value=0.0)
    gparam.add('n', value=0.5)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0)
    gparam.add('gmag', value=0.0)
    gparam.add('phi', value=0.0)
    return gparam

s_engine = chroma.ImageEngine.GalSimSEngine(size=53)
galtool = chroma.GalTools.SGalTool(s_engine)
gparam = fiducial_galaxy()

gparam['n'].value = 0.5
print 'testing sqrt(Ixx + Iyy) / hlr for n=0.5'
print 'analytic value is 1.20112'
print np.sqrt(s_engine.get_uncvl_r2(gparam, pixsize=1./7))
print

gparam['n'].value = 1.0
print 'testing sqrt(Ixx + Iyy) / hlr for n=1.0'
print 'analytic value is 1.45947'
print np.sqrt(s_engine.get_uncvl_r2(gparam, pixsize=1./7))
print

gparam['n'].value = 4.0
print 'testing sqrt(Ixx + Iyy) / hlr for n=4.0'
print 'analytic value is 4.65611'
print np.sqrt(s_engine.get_uncvl_r2(gparam, pixsize=1./7))
