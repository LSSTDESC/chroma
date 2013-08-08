import numpy
import scipy.integrate
import galsim
import time

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

def simple_SED():
    # wave = numpy.arange(550.0, 700.0, 0.1) # r-band
    # wave = numpy.arange(700.0, 850.0, 0.1) # i-band
    wave = numpy.arange(550.0, 850.0, 0.1) # super wide r+i-band
    # slowly falling spectrum
    photons = 1.0 * (wave / 685.0)**(-0.1)
    return wave, photons

def main():
    import matplotlib.pyplot as plt
    size = 61 # size of PSF postage stamp in pixels
    over = 7 # amount by which to oversample PSF for InterpolatedImage
    wave, photons = simple_SED()

    gal = galsim.Sersic(n=4.0, half_light_radius=0.2899, flux=1.0)
    gal.applyShift(0.3, 0.1)
    gal.applyShear(g=0.2, beta=0.0 * galsim.radians)
    pix = galsim.Pixel(1.0)

    t0 = time.time()
    PSF = seeing_PSF(wave, photons)
    thumb = galsim.Convolve([PSF, gal, pix])
    im = galsim.ImageD(size, size)
    thumb.draw(image=im, dx=1)
    im_array = im.array
    im_array /= im_array.max()
    log_im = numpy.log10(im_array)
    t1 = time.time()
    print 'seeing_PSF took: {}'.format(t1-t0)

    PSF2 = interpolated_seeing_PSF(size, over, wave, photons)
    im2 = galsim.ImageD(size, size)
    thumb.draw(image=im2, dx=1)
    im_array2 = im2.array
    im_array2 /= im_array2.max()
    log_im2 = numpy.log10(im_array2)
    t2 = time.time()
    print 'interpolated_seeing_PSF took: {}'.format(t2-t1)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(log_im, extent=[0, size * 0.2, 0, size * 0.2])
    X = numpy.linspace(0.0, size * 0.2, log_im.shape[0])
    Y = X
    ax.contour(X, Y, log_im, levels=numpy.arange(-5, 0, 0.5), colors='k')
    plt.show()

if __name__ == '__main__':
    main()
