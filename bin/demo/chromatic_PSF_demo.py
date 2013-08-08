import numpy
import scipy.integrate
import galsim

def air_refractive_index(wave):
    # refractive index in air at STP
    sigma_squared = 1.0 / (wave * 1.e-3)**2
    n = (64.328 + (29498.1 / (146.0 - sigma_squared))
         + (255.4 / (41.0 - sigma_squared))) * 1.e-6 + 1.0
    return n

def refrac(wave, zenith):
    n_squared = air_refractive_index(wave)**2
    r0 = (n_squared - 1.0) / (2.0 * n_squared)
    return r0 * numpy.tan(zenith)

def seeing(wave, seeing_500):
    # Kolmogorov turbulence predicts $seeing \propto \lambda^{-0.2}$
    return seeing_500 * (wave / 500)**(-0.2)

def chromatic_PSF(wave, photons, zenith, size, over,
                  plate_scale=0.2, moffat_beta=2.5, moffat_FWHM_500=0.7):

    R = refrac(wave, zenith)
    # normalize refraction to 685nm (right between filters r & i)
    R685 = refrac(685., zenith)
    R_pixels = (R - R685) * 3600 * 180 / numpy.pi / plate_scale # radians -> pixels
    # normalize SED
    photons /= scipy.integrate.simps(photons, wave)
    # predict the centroid to be here
    print 'integral centroid :{:8.5f}'.format(scipy.integrate.simps(photons * R_pixels, wave))

    mpsfs = [] # store monochromatic PSFs here
    for w, p, Rp in zip(wave, photons, R_pixels):
        FWHM = seeing(w, moffat_FWHM_500 / plate_scale)
        monochrome_PSF = galsim.Moffat(flux=p, fwhm=FWHM, beta=moffat_beta)
        # add in refraction
        monochrome_PSF.applyShift(0.0, Rp)
        mpsfs.append(monochrome_PSF)
    PSF = galsim.Add(mpsfs)
    print 'galsim.Add centroid :{:8.5f}'.format(PSF.centroid().y)

    im = galsim.ImageD(size*over+1,size*over+1)
    PSF.draw(image=im, dx=1.0/over)
    PSF = galsim.InterpolatedImage(im, dx=1.0/over)
    print 'galsim.InterpolatedImage centroid :{:8.5f}'.format(PSF.centroid().y)
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
    zenith = 60 * numpy.pi / 180
    PSF = chromatic_PSF(wave, photons, zenith, size, over)
    im = galsim.ImageD(size*over, size*over)
    PSF.draw(image=im, dx=1./7)
    im_array = im.array
    im_array /= im_array.max()
    log_im = numpy.log10(im_array)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(log_im, extent=[0, size * 0.2, 0, size * 0.2])
    X = numpy.linspace(0.0, size * 0.2, log_im.shape[0])
    Y = X
    ax.contour(X, Y, log_im, levels=numpy.arange(-5, 0, 0.5), colors='k')
    plt.show()

if __name__ == '__main__':
    main()
