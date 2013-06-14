import numpy
import scipy

class Sersic(object):
    ''' Class for handling all things related to Sersic surface brightness profiles.  The main usage
    is to evaluate the surface brightness at a given location, but also provides routines to convert
    peak surface brightness to integrated flux, for example.  Initialization is meant to be very
    flexible, with many options for specifying the ellipticity and size, and two ways to specify
    the amplitude of the Sersic.  The created object is callable, with the argument being a
    (y,x) position and the output being the surface brightness at that position.
    '''

    def __init__(self,
                 y0, x0, n, # required
                 peak=None, flux=None, # one of these is required
                 C11=None, C12=None, C22=None, # one possibility for size/ellipticity
                 a=None, b=None, phi=None, # another possibility for size/ellipticity
                 # if neither of the above two triplets is provided, then one of the following size
                 # parameters must be provided
                 FWHM=None, r_e=None,
                 # if specifying ellipticity in polar units (together with phi above), then
                 # one of the following three is required
                 b_over_a=None, emag=None, gmag=None,
                 # if specifying ellipticity in complex components, then one of the following pairs
                 # is required
                 e1=None, e2=None,
                 g1=None, g2=None):
        ''' Create a Sersic object.

        Arguments
        ---------
        y0, x0 -- center of profile
        n -- Sersic index
        r_e -- the effective radius (half-light radius) (for a circular profile)
        peak -- peak surface brightness
        flux -- integrated (over all space) surface brightness
        b_over_a -- minor axis / major axis ratio
        phi -- position angle measured anticlockwise from postive x axis (radians)
        C11, C12, C22 -- elements of the C-matrix as defined in Voigt+12 for example
        a, b -- the semimajor and semiminor halflight axes of distribution
                normalized such that for a circular profile, a * b = r_e**2
        FWHM -- the full-width at half maximum (for a circular profile)
        emag -- ellipticity in units of (a^2 - b^2)/(a^2 + b^2)
        gmag -- ellipticity in units of (a - b)/(a +b)
        e1, e2 -- cartesian representation of ellipticity also specifiable by (emag, 2*phi)
        g1, g2 -- cartesian representation of ellipticity also specifiable by (gmag, 2*phi)

        Read the comments below to figure out the hierarchy of all these possible specifications.
        '''

        # required things...
        self.y0 = y0
        self.x0 = x0
        self.n = n
        self.kappa = self.compute_kappa(n)

        # sort through the size/ellipticity possibilities:

        # most direct specification for internal representation is components of the C matrix
        # as specified in Voigt+10, Voigt+12, or Semboloni+12 for example.
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
            self.r_e = numpy.sqrt(self.a * self.b)
            self.phi = 0.5 * numpy.arctan2(2.0 * C12 / (one_over_a_squared - one_over_b_squared),
                                           (C11 - C22) / (one_over_a_squared - one_over_b_squared))
        else:
            # goal for this block is to determine a, b, phi
            # first check the direct case
            if a is not None and b is not None and phi is not None:
                self.a = a
                self.b = b
                self.phi = phi
                self.r_e = numpy.sqrt(a * b)
            else: # now check a hierarchy of size & ellip possibilities
                # first the size must be either FWHM or r_e
                if FWHM is not None:
                    self.r_e = 0.5 * FWHM * (self.kappa / numpy.log(2.0))**n
                else:
                    assert r_e is not None, "need to specify a size parameter"
                    self.r_e = r_e
                # goal here is to determine the axis ratio b_over_a, and position angle phi
                if phi is not None:  #must be doing a polar decomposition
                    self.phi = phi
                    if gmag is not None:
                        b_over_a = (1.0 - gmag)/(1.0 + gmag)
                    elif emag is not None:
                        b_over_a = numpy.sqrt((1.0 - emag)/(1.0 + emag))
                    else:
                        assert b_over_a is not None, "need to specify ellipticity magnitude"
                else: # doing a complex components decomposition
                    if g1 is not None and g2 is not None:
                        self.phi = 0.5 * numpy.arctan2(g2, g1)
                        gmag = numpy.sqrt(g1**2.0 + g2**2.0)
                        b_over_a = (1.0 - gmag)/(1.0 + gmag)
                    else:
                        assert e1 is not None and e2 is not None, "need to specify ellipticty"
                        self.phi = 0.5 * numpy.arctan2(e2, e1)
                        emag = numpy.sqrt(e1**2.0 + e2**2.0)
                        b_over_a = numpy.sqrt((1.0 - emag)/(1.0 + emag))

                self.a = self.r_e / numpy.sqrt(b_over_a)
                self.b = self.r_e * numpy.sqrt(b_over_a)
            cph = numpy.cos(self.phi)
            sph = numpy.sin(self.phi)
            self.C11 = (cph/self.a)**2 + (sph/self.b)**2
            self.C12 = 0.5 * (1.0/self.a**2 - 1.0/self.b**2) * numpy.sin(2.0 * self.phi)
            self.C22 = (sph/self.a)**2 + (cph/self.b)**2

        # last step is to determine normalization
        if peak is not None:
            self.peak = peak
            self.flux = self.compute_flux(n, self.r_e, peak)
        else:
            assert flux is not None, "need to specify amplitude"
            self.flux = flux
            self.peak = self.compute_peak(n, self.r_e, flux)

    def __call__(self, y, x):
        '''Return the surface brightness at (y,x)'''
        xp = x - self.x0
        yp = y - self.y0
        exponent = self.C11 * xp**2.0 + 2.0 * self.C12 * xp * yp + self.C22 * yp**2.0
        exponent **= 0.5 / self.n
        exponent *= -self.kappa
        return self.peak * numpy.exp(exponent)

    @staticmethod
    def compute_kappa(n):
        '''Compute Sersic exponent factor kappa from the Sersic index'''
        kguess = 1.9992 * n - 0.3271
        return scipy.optimize.newton(lambda k: scipy.special.gammainc(2.0 * n, k) - 0.5, kguess)

    @classmethod
    def compute_FWHM(cls, n, r_e, kappa=None):
        '''Compute the full-width at half maximum given input parameters.'''
        if kappa is None:
            kappa = cls.compute_kappa(n)
        return 2.0 * r_e * (numpy.log(2.0) / kappa)**n

    @classmethod
    def compute_flux(cls, n, r_e, peak, kappa=None):
        '''Compute flux integrated over all space given input parameters.'''
        if kappa is None:
            kappa = cls.compute_kappa(n)
        return (2.0 * numpy.pi) * peak * n * (kappa**(-2.0 * n)) * (r_e**2.0) \
          * scipy.special.gamma(2.0 * n)

    @classmethod
    def compute_peak(cls, n, r_e, flux, kappa=None):
        '''Compute peak surface brightness from integrated flux and Sersic params.'''
        if kappa is None:
            kappa = cls.compute_kappa(n)
        fluxnorm = cls.compute_flux(n, r_e, 1.0, kappa=kappa)
        return flux/fluxnorm
