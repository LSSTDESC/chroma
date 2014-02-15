#list of arbitrary numbers:
# GSEuclidePSFInt, GSAtmSeeingPSF:
#   the oversample factor is 7 and the (over)pixsize is (7 * 15)**2
# GSAtmPSF, GSGaussAtmPSF:
#   the oversample factor is 21 and the (over)pixsize is (21 * 15)**2

import hashlib

import galsim
import scipy
import numpy
import astropy.utils.console

import chroma

def GSEuclidPSF(wave, photons, ellipticity=0.0, phi=0.0):
    ''' Returns a Galsim SBProfile object representing the SED-dependent Euclid PSF as described in
    Voigt+12.
    '''
    hlr = lambda wave: 0.7 * (wave / 520.0)**0.6 # pixels
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    for w, p in zip(wave, photons):
        mpsfs.append(galsim.Gaussian(flux=p, half_light_radius=hlr(w)))
    PSF = galsim.Add(mpsfs)
    beta = phi * galsim.radians
    PSF.applyShear(g=ellipticity, beta=beta)
    return PSF

def GSEuclidPSFInt(wave, photons, ellipticity=0.0, phi=0.0):
    ''' Returns a Galsim SBProfile object representing the SED-dependent Euclid PSF as described in
    Voigt+12.  To make life faster, caches the result in a single 7 times oversampled image, instead
    of carrying around a sum of thousands of Gaussians.
    '''
    hlr = lambda wave: 0.7 * (wave / 520.0)**0.6 # pixels
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    for w, p in zip(wave, photons):
        mpsfs.append(galsim.Gaussian(flux=p, half_light_radius=hlr(w)))
    PSF = galsim.Add(mpsfs)
    beta = phi * galsim.radians
    PSF.applyShear(g=ellipticity, beta=beta)
    im = galsim.ImageD(105, 105) #arbitrary numbers!
    PSF.draw(image=im, scale=1.0/7)
    PSF = galsim.InterpolatedImage(im, scale=1.0/7)
    return PSF

def GSAtmPSF(wave, photons,
             pixel_scale=0.2, moffat_beta=2.5, moffat_FWHM=3.5,
             moffat_ellip=0.0, moffat_phi=0.0, **kwargs):
    ''' Returns a Galsim SBProfile object PB12-type differential chromatic refraction PSF by
    convolving a Moffat PSF with a DCR kernel in the zenith (y) direction.'''
    # get photon density binned by refraction angle
    R, angle_dens = chroma.wave_dens_to_angle_dens(wave, photons, **kwargs)
    # need to take out the huge zenith angle dependence, normalize to whatever the
    # refraction is at 685 nm
    R685 = chroma.atm_refrac(685.0, **kwargs)
    pixels = (R - R685) * 3600 * 180 / numpy.pi / pixel_scale # radians -> pixels
    sort = numpy.argsort(pixels)
    pixels = pixels[sort]
    angle_dens = angle_dens[sort]
    # now sample a size of 15 pixels oversampled by a factor of 21 => 315 subpixels
    pixmin = -7.5 + 1./42
    pixmax = 7.5 - 1./42
    y = numpy.linspace(pixmin, pixmax, 315) # coords of subpixel centers
    yboundaries = numpy.concatenate([numpy.array([y[0] - 1./42]), y + 1./42])
    yunion = numpy.union1d(pixels, yboundaries)
    angle_dens_interp = numpy.interp(yunion, pixels, angle_dens, left=0.0, right=0.0)
    PSFim = galsim.ImageD(315, 315)
    for i in range(315):
        w = numpy.logical_and(yunion >= yboundaries[i], yunion <= yboundaries[i+1])
        PSFim.array[i,157] = scipy.integrate.simps(angle_dens_interp[w], yunion[w])
    aPSF = galsim.InterpolatedImage(PSFim, scale=1.0/21, flux=1.0)
    mPSF = galsim.Moffat(beta=moffat_beta, fwhm=moffat_FWHM)
    mPSF.applyShear(g=moffat_ellip, beta=moffat_phi * galsim.radians)
    PSF = galsim.Convolve([aPSF, mPSF])
    return PSF

def GSAtmPSF2(wave, photons, pixel_scale=0.2, moffat_beta=2.5, moffat_FWHM=3.5,
              moffat_ellip=0.0, moffat_phi=0.0, **kwargs):
    ''' Returns a Galsim SBProfile object PB12-type differential chromatic refraction PSF by
    convolving a Moffat PSF with a DCR kernel in the zenith (y) direction.'''
    R = chroma.atm_refrac(wave, **kwargs)
    R685 = chroma.atm_refrac(685.0, **kwargs)
    R_pixels = (R - R685) * 3600 * 180 / numpy.pi / pixel_scale
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    for w, p, Rp in zip(wave, photons, R_pixels):
        psf1 = galsim.Moffat(flux=p*0.1, fwhm=moffat_FWHM, beta=moffat_beta)
        psf1.applyShift(0.0, Rp)
        mpsfs.append(psf1)
    PSF = galsim.Add(mpsfs)
    beta = moffat_phi * galsim.radians
    PSF.applyShear(g=moffat_ellip, beta=beta)
    im = galsim.ImageD(288, 288) #arbitrary numbers!
    PSF.draw(image=im, scale=1./7)
    PSF = galsim.InterpolatedImage(im, scale=1./7)
    return PSF

def GSAtmPSF3(wave, photons, pixel_scale=0.2, moffat_beta=2.5, moffat_FWHM=3.5,
              moffat_ellip=0.0, moffat_phi=0.0, **kwargs):
    ''' Returns a Galsim SBProfile object PB12-type differential chromatic refraction PSF by
    convolving a Moffat PSF with a DCR kernel in the zenith (y) direction.'''
    R = chroma.atm_refrac(wave, **kwargs)
    R685 = chroma.atm_refrac(685.0, **kwargs)
    R_pixels = (R - R685) * 3600 * 180 / numpy.pi / pixel_scale
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    im = galsim.ImageD(288, 288, scale=1./7)
    for w, p, Rp in zip(wave, photons, R_pixels):
        psf1 = galsim.Moffat(flux=p*0.1, fwhm=moffat_FWHM, beta=moffat_beta)
        psf1.applyShift(0.0, Rp)
        psf1.draw(image=im, add_to_image=True)
    PSF = galsim.InterpolatedImage(im, scale=1./7)
    return PSF

# Hope is to refactor such that can either give galsim chromatic galaxy and chromatic PSF, or
# give galsim monochrome galaxy and effective PSF.

def GSGaussAtmPSF(wave, photons,
                  pixel_scale=0.2, FWHM=3.5,
                  gauss_phi=0.0, gauss_ellip=0.0, **kwargs):
    ''' Returns a Galsim SBProfile object PB12-type differential chromatic refraction PSF by
    convolving a Gaussian PSF with a DCR kernel in the zenith (y) direction.'''
    # get photon density binned by refraction angle
    R, angle_dens = chroma.wave_dens_to_angle_dens(wave, photons, **kwargs)
    # need to take out the huge zenith angle dependence:
    # normalize to whatever the refraction is at 685 nm
    R685 = chroma.atm_refrac(685.0, **kwargs)
    pixels = (R - R685) * 3600 * 180 / numpy.pi / pixel_scale # degrees -> pixels
    sort = numpy.argsort(pixels)
    pixels = pixels[sort]
    angle_dens = angle_dens[sort]
    # now sample a size of 15 pixels oversampled by a factor of 21 => 315 subpixels
    pixmin = -7.5 + 1./42
    pixmax = 7.5 - 1./42
    y = numpy.linspace(pixmin, pixmax, 315) # coords of subpixel centers
    yboundaries = numpy.concatenate([numpy.array([y[0] - 1./42]), y + 1./42])
    yunion = numpy.union1d(pixels, yboundaries)
    angle_dens_interp = numpy.interp(yunion, pixels, angle_dens, left=0.0, right=0.0)
    PSFim = galsim.ImageD(315,315)
    for i in range(315):
        w = numpy.logical_and(yunion >= yboundaries[i], yunion <= yboundaries[i+1])
        PSFim.array[i,157] = scipy.integrate.simps(angle_dens_interp[w], yunion[w])
    aPSF = galsim.InterpolatedImage(PSFim, scale=1.0/21, flux=1.0)
    gPSF = galsim.Gaussian(fwhm=FWHM)
    gPSF.applyShear(g=gauss_ellip, beta=gauss_phi * galsim.radians)
    PSF = galsim.Convolve([aPSF, gPSF])
    return PSF

def GSAtmSeeingPSF(wave, photons, pixel_scale=0.2, moffat_beta=2.5, moffat_FWHM_500=3.5,
                   moffat_ellip=0.0, moffat_phi=0.0, **kwargs):
    ''' Returns a Galsim SBProfile object representing an atmospheric chromatic PSF characterized by
    both DCR and seeing chromatic effects.'''
    R = chroma.atm_refrac(wave, **kwargs)
    R685 = chroma.atm_refrac(685.0, **kwargs)
    R_pixels = (R - R685) * 3600 * 180 / numpy.pi / pixel_scale
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    for w, p, Rp in zip(wave, photons, R_pixels):
        fwhm = moffat_FWHM_500 * (w / 500.0)**(-0.2)
        psf1 = galsim.Moffat(flux=p, fwhm=fwhm, beta=moffat_beta)
        psf1.applyShift(0.0, Rp)
        mpsfs.append(psf1)
    PSF = galsim.Add(mpsfs)
    beta = moffat_phi * galsim.radians
    PSF.applyShear(g=moffat_ellip, beta=beta)
    im = galsim.ImageD(288, 288) #arbitrary numbers!
    PSF.draw(image=im, scale=1./7)
    PSF = galsim.InterpolatedImage(im, scale=1./7)
    return PSF

def GSSeeingPSF(wave, photons, moffat_beta=2.5, moffat_FWHM_500=3.5,
                moffat_ellip=0.0, moffat_phi=0.0):
    ''' Returns a Galsim SBProfile object representing an atmospheric chromatic PSF characterized by
    both DCR and seeing chromatic effects.'''
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    for w, p in zip(wave, photons):
        fwhm = moffat_FWHM_500 * (w / 500.0)**(-0.2)
        psf1 = galsim.Moffat(flux=p, fwhm=fwhm, beta=moffat_beta)
        mpsfs.append(psf1)
    PSF = galsim.Add(mpsfs)
    PSF.applyShear(g=moffat_ellip, beta=moffat_beta * galsim.radians)
    im = galsim.ImageD(124, 124) #arbitrary numbers!  support up to 41x41 stamp
    PSF.draw(image=im, scale=1./3)
    PSF = galsim.InterpolatedImage(im, scale=1./3)
    return PSF
