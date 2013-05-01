import hashlib

import galsim
import scipy
import numpy
import astropy.utils.console

import chroma

def GSEuclidPSF(wave, photons, ellipticity=0.0, phi=0.0):
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
    hlr = lambda wave: 0.7 * (wave / 520.0)**0.6 # pixels
    mpsfs = []
    photons /= scipy.integrate.simps(photons, wave)
    for w, p in zip(wave, photons):
        mpsfs.append(galsim.Gaussian(flux=p, half_light_radius=hlr(w)))
    PSF = galsim.Add(mpsfs)
    beta = phi * galsim.radians
    PSF.applyShear(g=ellipticity, beta=beta)
    im = galsim.ImageD(105, 105)
    PSF.draw(image=im, dx=1.0/7)
    PSF = galsim.InterpolatedImage(im, dx=1.0/7)
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
        return chroma.SBProfile.Sersic(y0, x0, n,
                                       r_e=self._rp(wave), flux=norm, gmag=ellipticity, phi=phi)

    def _load_monochromatic_PSFs(self):
        # create all the monochromatic Gaussians (as Sersics) at initialization and store for later
        print "Loading PSF"
        self._monochromatic_PSFs = []
        with astropy.utils.console.ProgressBar(len(self.wave)) as bar:
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
        with astropy.utils.console.ProgressBar(len(self._monochromatic_PSFs)) as bar:
            for i, mpsf in enumerate(self._monochromatic_PSFs):
                psfcube[..., i] = mpsf(y1,x1)
                bar.update()
        return psfcube

    def __call__(self, y, x):
        '''Integrate the psfcube over the lambda direction to get the PSF.'''
        psfcube = self.psfcube(y, x)
        print "Integrating over wavelengths"
        return scipy.integrate.trapz(psfcube, self.wave)

class MoffatPSF(object):
    def __init__(self, y0, x0, beta, # required
                 flux=None, # required right now, eventually allow peak as alternative?
                 C11=None, C12=None, C22=None, # one possibility for size/ellipticity
                 a=None, b=None, phi=None, # another possibility for size/ellipticity
                 # if neither of the above two triplets is provided, then one of the following size
                 # parameters must be provided
                 FWHM=None, alpha=None,
                 # if specifying ellipticity in polar units (including phi above), then
                 # one of the following three params is required
                 b_over_a=None, emag=None, gmag=None,
                 # if specifying ellipticity in complex components, then one of the following pairs
                 # is required
                 e1=None, e2=None,
                 g1=None, g2=None):
        self.y0 = y0
        self.x0 = x0
        self.beta = beta
        self.flux = flux

        if C11 is not None and C12 is not None and C22 is not None:
            self.C11 = C11
            self.C12 = C12
            self.C22 = C22
            # want to keep some additional bookkeepping parameters around as well...
            one_over_a_squared = 0.5 * (C11 + C22 + numpy.sqrt((C11 - C22)**2 + 4.0 * C12**2))
            one_over_b_squared = C11 + C22 - one_over_a_squared
            # there's degeneracy between a, b and phi at this point so enforce a > b
            if one_over_a_squared > one_over_b_squared:
                one_over_a_squared, one_over_b_squared = one_over_b_squared, one_over_a_squared
            self.a = numpy.sqrt(1.0 / one_over_a_squared)
            self.b = numpy.sqrt(1.0 / one_over_b_squared)
            self.alpha = numpy.sqrt(self.a * self.b)
            self.phi = 0.5 * numpy.arctan2(2.0 * C12 / (one_over_a_squared - one_over_b_squared),
                                           (C11 - C22) / (one_over_a_squared - one_over_b_squared))

        else:
            # goal for this block is to determine a, b, phi
            # first check the direct case
            if a is not None and b is not None and phi is not None:
                self.a = a
                self.b = b
                self.phi = phi
                self.alpha = numpy.sqrt(a * b)
            else: # now check a hierarchy of size & ellip possibilities
                # first the size must be either FWHM or r_e
                if FWHM is not None:
                    self.alpha = FWHM / (2.0 * numpy.sqrt(2.0**(1.0/self.beta) - 1.0))
                else:
                    assert alpha is not None, "need to specify a size parameter"
                    self.alpha = alpha
                # goal here is to determine the axis ratio b_over_a, and position angle phi
                if phi is not None: # must be doing a polar decomposition
                    self.phi = phi
                    if gmag is not None:
                        b_over_a = (1.0 - gmag)/(1.0 + gmag)
                    elif emag is not None:
                        b_over_a = numpy.sqrt((1.0 - emag)/(1.0 + emag))
                    else:
                        assert b_over_a is not None, "need to specify ellipticity magnitude"
                else: #doing a complex components decomposition
                    if g1 is not None and g2 is not None:
                        self.phi = 0.5 * numpy.arctan2(g2, g1)
                        gmag = numpy.sqrt(g1**2.0 + g2**2.0)
                        b_over_a = (1.0 - gmag)/(1.0 + gmag)
                    else:
                        assert e1 is not None and e2 is not None, "need to specify ellipticty"
                        self.phi = 0.5 * numpy.arctan2(e2, e1)
                        emag = numpy.sqrt(e1**2.0 + e2**2.0)
                        b_over_a = numpy.sqrt((1.0 - emag)/(1.0 + emag))

                self.a = self.alpha / numpy.sqrt(b_over_a)
                self.b = self.alpha * numpy.sqrt(b_over_a)
            cph = numpy.cos(self.phi)
            sph = numpy.sin(self.phi)
            self.C11 = (cph/self.a)**2 + (sph/self.b)**2
            self.C12 = 0.5 * (1.0/self.a**2 - 1.0/self.b**2) * numpy.sin(2.0 * self.phi)
            self.C22 = (sph/self.a)**2 + (cph/self.b)**2

        det = self.C11 * self.C22 - self.C12**2.0
        self.norm = self.flux * (self.beta - 1.0) / (numpy.pi / numpy.sqrt(abs(det)))

        self.key = self.hash()

    def hash(self):
        m = hashlib.md5()
        m.update(str((self.x0, self.y0, self.beta)))
        m.update(str((self.C11, self.C12, self.C22)))
        m.update(str(self.flux))
        return m.hexdigest()

    def __call__(self, y, x):
        xp = x - self.x0
        yp = y - self.y0
        base = 1.0 + self.C11 * xp**2.0 + 2.0 * self.C12 * xp * yp + self.C22 * yp**2.0
        return self.norm * base**(-self.beta)

class AtmDispPSF(object):
    def __init__(self, wave, photons, plate_scale=0.2, xloc=0.0, **kwargs):
        self.wave = wave
        self.photons = photons
        self.plate_scale = plate_scale
        self.kwargs = kwargs
        self.xloc = xloc
        self.key = self.hash()

    def hash(self):
        m = hashlib.md5()
        m.update(str(tuple(self.wave)))
        m.update(str(tuple(self.photons)))
        m.update(str(self.plate_scale))
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
        R685 = chroma.atm_refrac(685.0, **self.kwargs)
        pixels = (R - R685) * 206265 / self.plate_scale
        sort = numpy.argsort(pixels)
        pixels = pixels[sort]
        angle_dens = angle_dens[sort]
        angle_dens /= scipy.integrate.simps(angle_dens, pixels)
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
    def __init__(self, wave, photons, aPSF_kwargs=None, mPSF_kwargs=None):
        self.wave = wave
        self.photons = photons
        self.aPSF = AtmDispPSF(wave, photons, **aPSF_kwargs)
        self.mPSF = MoffatPSF(0.0, 0.0, **mPSF_kwargs)
        self.cPSF = ConvolvePSF([self.aPSF, self.mPSF])
        self.key = self.cPSF.key

    def __call__(self, y, x):
        return self.cPSF(y, x)

def GSAtmPSF(wave, photons, aPSF_kwargs=None, mPSF_kwargs=None)
    R, angle_dens = chroma.wave_dens_to_angle_dens(wave, photons, **kwargs)
    R685 = chroma.atm_refrac(685.0, **aPSF_kwargs)
    pixels = (R - R685) * 206265 / plate_scale
    sort = numpy.argsort(pixels)
    pixels = pixels[sort]
    angle_dens = angle_dens[sort]
    angle_dens /= scipy.integrate.simps(angle_dens, pixels)
    #should really be integrating next step, not interpolating
    PSFim = numpy.interp(y, pixels, angle_dens, left=0.0, right=0.0)
    aPSF = galsim.InterpolatedImage(im, dx=1.0/7)
    mPSF = galsim.Moffat(beta=mPSF_kwargs['beta'],
                         fwhm=mPSF_kwargs['FWHM'],
                         flux=mPSF_kwargs['flux'])
    PSF = galsim.Convolve([aPSF, mPSF])
    return PSF
