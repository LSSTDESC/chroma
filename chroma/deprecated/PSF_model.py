import hashlib

import galsim
import scipy
import numpy

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
    R685 = chroma.get_refraction(685.0, **kwargs)
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
    R = chroma.get_refraction(wave, **kwargs)
    R685 = chroma.get_refraction(685.0, **kwargs)
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
    R = chroma.get_refraction(wave, **kwargs)
    R685 = chroma.get_refraction(685.0, **kwargs)
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


def GSGaussAtmPSF(wave, photons,
                  pixel_scale=0.2, FWHM=3.5,
                  gauss_phi=0.0, gauss_ellip=0.0, **kwargs):
    ''' Returns a Galsim SBProfile object PB12-type differential chromatic refraction PSF by
    convolving a Gaussian PSF with a DCR kernel in the zenith (y) direction.'''
    # get photon density binned by refraction angle
    R, angle_dens = chroma.wave_dens_to_angle_dens(wave, photons, **kwargs)
    # need to take out the huge zenith angle dependence:
    # normalize to whatever the refraction is at 685 nm
    R685 = chroma.get_refraction(685.0, **kwargs)
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
    R = chroma.get_refraction(wave, **kwargs)
    R685 = chroma.get_refraction(685.0, **kwargs)
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


class VoigtEuclidPSF(object):
    '''Class to handle the Euclid-like chromatic PSF defined in the Voigt+12 color gradient paper.'''
    def __init__(self, wave, photons, ellipticity=0.0, phi=0.0, y0=0.0, x0=0.0):
        '''Initialize a EuclidPSF instance.

        Arguments
        ---------
        wave -- wavelengths in nm of spectrum
        photons -- d(photons)/d(lambda) spectrum.  Normalization doesn't matter.
        ellipticity -- defined as (a-b)/(a+b) in Voigt+12. (default 0.0)
        phi -- position angle of ellipticity CCW from x-axis. (default 0.0)
        x0, y0 -- center of PSF

        Returns
        -------
        A callable class instance.
        '''
        self.wave = wave
        self.photons = photons / scipy.integrate.trapz(photons, wave)
        self.ellipticity = ellipticity
        self.phi = phi
        self.y0 = y0
        self.x0 = x0
        self._monochromatic_PSFs = None
        self.key = self.hash()

    @staticmethod
    def _rp(wave):
        '''Effective radius of PSF at given wavelength -- Equation 2 from Voigt+12'''
        rp0 = 0.7
        wave0 = 520.0
        return rp0 * (wave / wave0)**0.6

    def _monochromatic_PSF(self, y0, x0, wave, ellipticity, phi, norm):
        '''Returns the Voigt+12 Euclid Gaussian PSF for particular wavelength.'''
        n = 0.5 # Gaussian
        return chroma.Sersic.Sersic(y0, x0, n,
                                    r_e=self._rp(wave), flux=norm, gmag=ellipticity, phi=phi)

    def _load_monochromatic_PSFs(self):
        # create all the monochromatic Gaussians (as Sersics) at initialization and store for later
        print "Loading PSF"
        self._monochromatic_PSFs = []
        with chroma.ProgressBar(len(self.wave)) as bar:
            for wav, phot in zip(self.wave, self.photons):
                self._monochromatic_PSFs.append(
                    self._monochromatic_PSF(self.y0, self.x0, wav, self.ellipticity, self.phi, phot))
                bar.update()

    def hash(self):
        '''Make object parameters hashable so a sophisticated calling class won't need to regenerate
        monochromatic PSFs or PSF images unnecessarily if it sees it already has a psf image for
        the same PSF params in its database (see VoigtImageFactory).  Hashing by input parameters is
        better than hashing by the instance ID since more than one instance can be created with the
        same parameters but will have different IDs.

        Somewhat experimental, since I'm no expert on md5 hashes and there are some warnings on
        the intartubze about hashing numpy.ndarray objects (which is why I tupled them)...  but
        seems to work right now...
        '''
        m = hashlib.md5()
        m.update(str((self.y0, self.x0, self.ellipticity, self.phi)))
        m.update(str(tuple(self.wave)))
        m.update(str(tuple(self.photons)))
        return m.hexdigest()

    def psfcube(self, y, x):
        '''Evaluate monochromatic PSFs to make cube of y, x, lambda'''
        if self._monochromatic_PSFs is None:
            self._load_monochromatic_PSFs()
        if isinstance(y, int) or isinstance(y, float):
            y1 = numpy.array([y])
            x1 = numpy.array([x])
        if isinstance(y, list) or isinstance(y, tuple):
            y1 = numpy.array(y)
            x1 = numpy.array(x)
        if isinstance(y, numpy.ndarray):
            y1 = y
            x1 = x
        shape = list(y1.shape)
        shape.append(len(self._monochromatic_PSFs))
        psfcube = numpy.empty(shape, dtype=numpy.float64)
        print "Evaluating PSF"
        with chroma.ProgressBar(len(self._monochromatic_PSFs)) as bar:
            for i, mpsf in enumerate(self._monochromatic_PSFs):
                psfcube[..., i] = mpsf(y1,x1)
                bar.update()
        return psfcube

    def __call__(self, y, x):
        '''Integrate the psfcube over the lambda direction to get the PSF.'''
        psfcube = self.psfcube(y, x)
        print "Integrating over wavelengths"
        return scipy.integrate.trapz(psfcube, self.wave)


class AtmDispPSF(object):
    def __init__(self, wave, photons, pixel_scale=0.2, xloc=0.0, **kwargs):
        self.wave = wave
        self.photons = photons
        self.pixel_scale = pixel_scale
        self.kwargs = kwargs
        self.xloc = xloc
        self.key = self.hash()

    def hash(self):
        m = hashlib.md5()
        m.update(str(tuple(self.wave)))
        m.update(str(tuple(self.photons)))
        m.update(str(self.pixel_scale))
        m.update(str(self.xloc))
        keys = self.kwargs.keys()
        keys.sort()
        for key in keys:
            m.update(str((key, self.kwargs[key])))
        return m.hexdigest()

    def __call__(self, y, x):
        if isinstance(y, int) or isinstance(y, float):
            y1 = numpy.array([y])
            x1 = numpy.array([x])
        if isinstance(y, list) or isinstance(y, tuple):
            y1 = numpy.array(y)
            x1 = numpy.array(x)
        if isinstance(y, numpy.ndarray):
            y1 = y
            x1 = x
        R, angle_dens = chroma.wave_dens_to_angle_dens(self.wave, self.photons, **self.kwargs)
        R685 = chroma.get_refraction(685.0, **self.kwargs)
        pixels = (R - R685) * 3600 * 180 / numpy.pi / self.pixel_scale
        sort = numpy.argsort(pixels)
        pixels = pixels[sort]
        angle_dens = angle_dens[sort]
        PSF = numpy.interp(y, pixels, angle_dens, left=0.0, right=0.0)
        minx = abs(self.xloc - x).min()
        assert minx < 1.e-10
        PSF *= (abs(self.xloc - x) < 1.e-10)
        return PSF


class ConvolvePSF(object):
    def __init__(self, PSFs, factor=3):
        self.PSFs = PSFs
        self.factor = factor
        self.key = self.hash()

    def hash(self):
        m = hashlib.md5()
        for PSF in self.PSFs:
            m.update(PSF.key)
        return m.hexdigest()

    @staticmethod
    def _rebin(a, shape):
        '''Bin down image a to have final size given by shape.

        I think I stole this from stackoverflow somewhere...
        '''
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    def __call__(self, y, x):
        # first compute `factor` oversampled coordinates from y, x, which are assumed to be in a
        # grid and uniformly spaced (any way to relax this assumption?)
        if isinstance(y, int) or isinstance(y, float):
            y1 = numpy.array([y])
            x1 = numpy.array([x])
        if isinstance(y, list) or isinstance(y, tuple):
            y1 = numpy.array(y)
            x1 = numpy.array(x)
        if isinstance(y, numpy.ndarray):
            y1 = y
            x1 = x
        nx = x.shape[1]
        ny = y.shape[0]
        dx = (x.max() - x.min())/(nx - 1.0)
        dy = (y.max() - y.min())/(ny - 1.0)
        x0 = x.min() - 0.5 * dx
        y0 = y.min() - 0.5 * dy
        x1 = x.max() + 0.5 * dx
        y1 = y.max() + 0.5 * dy
        dsubx = dx / self.factor
        dsuby = dy / self.factor
        xsub = numpy.linspace(x0 + dsubx/2.0, x1 - dsubx/2.0, nx * self.factor)
        ysub = numpy.linspace(y0 + dsubx/2.0, y1 - dsubx/2.0, ny * self.factor)
        xsub, ysub = numpy.meshgrid(xsub, ysub)

        over = self.PSFs[0](ysub, xsub)
        for PSF in self.PSFs[1:]:
            over = scipy.signal.fftconvolve(over, PSF(ysub, xsub), mode='same')
        return self._rebin(over, x.shape)


class VoigtAtmPSF(object):
    def __init__(self, wave, photons,
                 pixel_scale=0.2, moffat_beta=2.5, moffat_FWHM=3.5, **kwargs):
        self.wave = wave
        self.photons = photons
        self.aPSF = AtmDispPSF(wave, photons, pixel_scale=pixel_scale, **kwargs)
        self.mPSF = MoffatPSF(0.0, 0.0, moffat_beta, FWHM=moffat_FWHM,
                              gmag=0.0, phi=0.0, flux=1.0)
        self.cPSF = ConvolvePSF([self.aPSF, self.mPSF])
        self.key = self.cPSF.key

    def __call__(self, y, x):
        return self.cPSF(y, x)


class VoigtGaussAtmPSF(object):
    def __init__(self, wave, photons, aPSF_kwargs=None, gPSF_kwargs=None):
        self.wave = wave
        self.photons = photons
        self.aPSF = AtmDispPSF(wave, photons, **aPSF_kwargs)
        self.gPSF = chroma.Sersic.Sersic(0.0, 0.0, 0.5, **mPSF_kwargs)
        self.cPSF = ConvolvePSF([self.aPSF, self.gPSF])
        self.key = self.cPSF.key

    def __call__(self, y, x):
        return self.cPSF(y, x)
