import lmfit

import _mypath
import chroma

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.0)
    gparam.add('y0', value=0.0)
    gparam.add('n', value=4.0)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0)
    gparam.add('gmag', value=0.0)
    gparam.add('phi', value=0.0)
    return gparam

s_engine = chroma.ImageEngine.GalSimSEngine()
gparam = fiducial_galaxy()
galtool = chroma.GalTools.SGalTool(s_engine)
overim = s_engine.get_uncvl_image(gparam, pixsize=1./7)
print s_engine.get_uncvl_r2(gparam)
