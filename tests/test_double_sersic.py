import _mypath
import chroma
import galsim

monochromaticPSF = galsim.Moffat(fwhm=0.7, beta=3.0)
PSF = galsim.ChromaticAtmosphere(monochromaticPSF, base_wavelength=500.0,
                                 zenith_angle=45.0*galsim.degrees,
                                 parallactic_angle=0.0*galsim.degrees,
                                 alpha=-0.2)
stamp_size = 32
pixel_scale = 0.2
SED1 = chroma.SED('../data/SEDs/CWW_E_ext.ascii')
SED2 = chroma.SED('../data/SEDs/CWW_Im_ext.ascii')
bandpass = chroma.Bandpass('../data/filters/LSST_g.dat')

dst = chroma.DoubleSersicTool(PSF, stamp_size, pixel_scale, SED1=SED1, SED2=SED2,
                              bandpass=bandpass)
gparam = dst.default_galaxy()
im = dst.get_image(gparam)
psfs = dst.get_PSF_image()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(im.array)
plt.show()
