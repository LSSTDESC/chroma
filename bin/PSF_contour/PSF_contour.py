import numpy
import galsim
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import _mypath
import chroma

def get_PSF_image(PSF, size=15, pixsize=1.0):
    PSF_image = galsim.ImageD(int(round(size/pixsize)), int(round(size/pixsize)))
    PSF.draw(image=PSF_image, dx=pixsize)
    return PSF_image.array

def PSF_contour(filter_name, SED, z, seeing, zenith=30*numpy.pi/180):
    PSF_model = chroma.PSF_model.GSAtmSeeingPSF

    data_dir = '../../data/'
    filter_file = data_dir+'filters/LSST_{}.dat'.format(filter_name)
    SED_file = data_dir+'SEDs/{}.ascii'.format(SED)
    wave, photons = chroma.utils.get_photons(SED_file, filter_file, z)
    PSF = PSF_model(wave, photons, zenith=zenith, moffat_FWHM_500=seeing)

    PSF_image = get_PSF_image(PSF, pixsize=1./7)
    PSF_image /= PSF_image.max()

    X = numpy.arange(0, 14.99, 1./7)
    Y = X

    #levels = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5]
    levels = numpy.arange(-5.0, 0.01, 1./3)
    cmap = cm.gist_rainbow

    plt.figure(figsize=(5,5))
    c1 = plt.imshow(numpy.log10(PSF_image), extent=[0, 15-1./7, 0, 15 - 1./7])
    c2 = plt.contour(X, Y, numpy.log10(PSF_image), levels, colors='k')
    plt.show()

if __name__ == '__main__':
    # PSF_contour('r', 'CWW_E_ext', 1.25, 2.5, zenith=60./180 * numpy.pi)
    PSF_contour('r', 'ukg5v', 1.25, 2.5, zenith=60./180 * numpy.pi)
