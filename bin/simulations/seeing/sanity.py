import numpy
import scipy.integrate
import galsim
import time

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

def seeing(wave, seeing_500):
    '''Return chromatic Kolmogorov seeing wrt seeing at 500 nm'''
    # Kolmogorov turbulence predicts $seeing \propto \lambda^{-0.2}$
    return seeing_500 * (wave / 500)**(-0.2)

def seeing_PSF(wave, photons, moffat_beta=2.5, moffat_FWHM_500=3.5):
    ''' Create galsim.SBProfile object representing PSF given specific SED.'''
    photons /= scipy.integrate.simps(photons, wave) # normalize SED
    mpsfs = [] # store monochromatic PSFs here
    for w, p, in zip(wave, photons):
        FWHM = seeing(w, moffat_FWHM_500)
        monochrome_PSF = galsim.Moffat(flux=p, fwhm=FWHM, beta=moffat_beta)
        mpsfs.append(monochrome_PSF)
    PSF = galsim.Add(mpsfs)
    return PSF

def interpolated_seeing_PSF(size, over, *args, **kwargs):
    PSF = seeing_PSF(*args, **kwargs)
    im = galsim.ImageD(size*over+1,size*over+1)
    PSF.draw(image=im, dx=1.0/over)
    PSF = galsim.InterpolatedImage(im, dx=1.0/over)
    return PSF

def main():
    import matplotlib.pyplot as plt
    size = 15 # size of PSF postage stamp in pixels
    over = 3 # amount by which to oversample PSF for InterpolatedImage
    data_dir = '../../data/'
    filter_file = data_dir+'filters/LSST_r.dat'
    gal_SED_file = data_dir+'SEDs/CWW_E_ext.ascii'
    z = 0.0
    wave, photons = chroma.utils.get_photons(gal_SED_file, filter_file, z)

    t0 = time.time()
    hlrs = [0.27, 0.30, 0.33, 0.36]
    PSF = seeing_PSF(wave, photons)
    for hlr in hlrs:
        gal = galsim.Sersic(n=4.0, half_light_radius=hlr, flux=1.0)
        gal.applyShift(0.3, 0.1)
        gal.applyShear(g=0.2, beta=0.0 * galsim.radians)
        pix = galsim.Pixel(1.0)

        thumb = galsim.Convolve([PSF, gal, pix])
        im = galsim.ImageD(size, size)
        thumb.draw(image=im, dx=1.0)
        im_array = im.array
        im_array /= im_array.max()
        log_im = numpy.log10(im_array)
    t1 = time.time()
    print 'seeing_PSF took: {}'.format(t1-t0)

    PSF2 = interpolated_seeing_PSF(size, over, wave, photons)
    for hlr in hlrs:
        gal2 = galsim.Sersic(n=4.0, half_light_radius=0.2899, flux=1.0)
        gal2.applyShift(0.3, 0.1)
        gal2.applyShear(g=0.2, beta=0.0 * galsim.radians)
        pix2 = galsim.Pixel(1.0)

        im2 = galsim.ImageD(size, size)
        thumb2 = galsim.Convolve([PSF2, gal2, pix2])
        thumb2.draw(image=im2, dx=1.0)
        im_array2 = im2.array
        im_array2 /= im_array2.max()
        log_im2 = numpy.log10(im_array2)
    t2 = time.time()
    print 'interpolated_seeing_PSF took: {}'.format(t2-t1)

    f = plt.figure()
    ax1 = f.add_subplot(131)
    my_imshow(im_array, ax=ax1)
    ax1.set_title('seeing_PSF')
    ax2 = f.add_subplot(132)
    my_imshow(im_array2, ax=ax2)
    ax2.set_title('interpolated_seeing_PSF')
    ax3 = f.add_subplot(133)
    my_imshow((im_array - im_array2)/im_array, ax=ax3)
    ax3.set_title('difference')
    plt.show()

    print ((im_array - im_array2)/im_array)[1:-1,1:-1].mean()

if __name__ == '__main__':
    main()
