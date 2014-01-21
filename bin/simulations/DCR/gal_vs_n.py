# displays images and profiles for Sersic galaxies with
# the same second moment radius, but different Sersic
# indices

import numpy
import lmfit

import _mypath
import chroma

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.0)
    gparam.add('y0', value=0.0)
    gparam.add('n', value=0.5, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.0)
    gparam.add('phi', value=0.0)
    return gparam

gparam= fiducial_galaxy()
gparam['n'].value = 0.5

s_engine = chroma.ImageEngine.GalSimSEngine(size=15, oversample_factor=7)
stool = chroma.GalTools.SGalTool(s_engine)

gparam = stool.set_uncvl_r2(gparam, (0.27/0.2)**2)
im05 = s_engine.get_uncvl_image(gparam, pixsize=1./7)

gparam['n'].value = 1.0
gparam = stool.set_uncvl_r2(gparam, (0.27/0.2)**2)
im10 = s_engine.get_uncvl_image(gparam, pixsize=1./7)

gparam['n'].value = 4.0
gparam = stool.set_uncvl_r2(gparam, (0.27/0.2)**2)
im40 = s_engine.get_uncvl_image(gparam, pixsize=1./7)

import matplotlib.pyplot as plt
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
vmin = im10.min()
vmax = im10.max()

ax1.imshow(im05, vmin=vmin, vmax=vmax)
ax2.imshow(im10, vmin=vmin, vmax=vmax)
ax3.imshow(im40, vmin=vmin, vmax=vmax)
ax4.plot(im05.sum(axis=0))
ax5.plot(im10.sum(axis=0))
ax6.plot(im40.sum(axis=0))
ax4.set_ylim(bottom=0)
ax5.set_ylim(bottom=0)
ax6.set_ylim(bottom=0)
ax4.set_xlim([0,15*7])
ax5.set_xlim([0,15*7])
ax6.set_xlim([0,15*7])
plt.show()
