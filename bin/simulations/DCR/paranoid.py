import numpy
import matplotlib.pyplot as plt
import scipy.integrate
import galsim

import _mypath
import chroma

def my_imshow(my_img,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%5e @ [%4i, %4i]' % (my_img[y, x], x, y)
        except IndexError:
            return ''
    img = ax.imshow(my_img,**kwargs)
    ax.format_coord = format_coord
    return img

def paranoid():
    s_engine = chroma.ImageEngine.GalSimSEngine(size=31)
    galtool = chroma.GalTools.SGalTool(s_engine)
    PSF_ellip = 0.0
    PSF_phi = 0.0
    data_dir = '../../data/'
    filter_file = data_dir+'filters/LSST_r.dat'
    gal_SED_file = data_dir+'SEDs/CWW_E_ext.ascii'
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    # thin out stellar spectrum
    swave = swave[::50]
    sphotons = sphotons[::50]
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = chroma.PSF_model.GSAtmPSF(swave, sphotons, zenith=30.0 * numpy.pi / 180.0)
    star_PSF2 = chroma.PSF_model.GSSeeingPSF(swave, sphotons)

    PSF_overim = s_engine.get_PSF_image(star_PSF, pixsize=1./7)
    PSF2_overim = s_engine.get_PSF_image(star_PSF2, pixsize=1./7)

    f = plt.figure()
    ax1 = f.add_subplot(121)
    my_imshow(numpy.log10(abs(PSF_overim/PSF_overim.max())), ax=ax1,
              extent=[0, PSF_overim.shape[0], 0, PSF_overim.shape[1]],
              vmin=-5, vmax=0)
    ax2 = f.add_subplot(122)
    my_imshow(numpy.log10(abs(PSF2_overim/PSF2_overim.max())), ax=ax2,
              extent=[0, PSF2_overim.shape[0], 0, PSF2_overim.shape[1]],
              vmin=-5, vmax=0)
    plt.show()

if __name__ == '__main__':
    paranoid()
