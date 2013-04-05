import numpy as np
from scipy.special import gammainc, gamma
from scipy.optimize import newton

class SersicInputError(ValueError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Sersic:
    ''' Class for handling all things related to Sersic surface brightness profiles.  The main usage
    is to evaluate the surface brightness at a given location, but also provides routines to convert
    peak surface brightness to integrated flux, for example.  Initialization is meant to be very
    flexible, with many options for specifying the ellipticity and size, and two ways to specify
    the amplitude of the Sersic.  The created object is callable, with the argument being a
    (y,x) position and the output being the surface brightness at that position.
    '''

    def __init__(self,
                 y0, x0, n, #required
                 peak=None, flux=None, #one of these is required
                 C11=None, C12=None, C22=None, #one possibility for size/ellipticity
                 a=None, b=None, phi=None, #another possibility for size/ellipticity
                 #if neither of the above two triplets is provided, then one of the following size
                 #parameters must be provided
                 FWHM=None, r_e=None,
                 #if specifying ellipticity in polar units (together with phi above), then
                 #one of the following three is required
                 b_over_a=None, emag=None, gmag=None,
                 #if specifying ellipticity in complex components, then one of the following pairs
                 #is required
                 e1=None, e2=None,
                 g1=None, g2=None):
        ''' Create a Sersic object.

        Arguments
        ---------
        y0, x0 -- center of profile
        n -- Sersic index
        r_e -- the effective radius (half-light radius)
        peak -- peak surface brightness
        axis_ratio -- minor axis / major axis
        phi -- position angle measured anticlockwise from postive x axis
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
            # want to keep some bookkeepping parameters around though...
            one_over_a_squared = 0.5 * (C11 + C22 + np.sqrt((C11 - C22)**2 + 4 * C12**2))
            self.a = np.sqrt(1. / one_over_a_squared)
            one_over_b_squared = C11 + C22 - one_over_a_squared
            self.b = np.sqrt(1. / one_over_b_squared)
            self.r_e = np.sqrt(self.a * self.b)
            twophi = np.arctan2(2 * C12 / (one_over_a_squared - one_over_b_squared),
                                (C11 - C22) / (one_over_a_squared - one_over_b_squared))
            self.phi = twophi/2.0 # radians
        else:
            # goal for this block is to determine a, b, phi
            # first check the direct case
            if a is not None and b is not None and phi is not None:
                self.a = a
                self.b = b
                self.phi = phi
                self.r_e = np.sqrt(a * b)
            else: # now check a hierarchy of size + ellip
                # first the size
                if FWHM is not None:
                    self.r_e = 0.5 * FWHM * (self.kappa / np.log(2.0))**n
                else:
                    assert r_e is not None, "need to specify a size parameter"
                    self.r_e = r_e
                # goal here is to determine the axis ratio b_over_a, and phi
                if phi is not None: #doing a polar decomposition
                    self.phi = phi
                    if gmag is not None:
                        b_over_a = (1.0 - gmag)/(1.0 + gmag)
                    elif emag is not None:
                        b_over_a = np.sqrt((1.0 - emag)/(1.0 + emag))
                    else:
                        assert b_over_a is not None, "need to specify ellipticity magnitude"
                else: #doing a complex components decomposition
                    if g1 is not None and g2 is not None:
                        self.phi = 0.5 * np.arctan2(g2, g1)
                        gmag = np.sqrt(g1**2.0 + g2**2.0)
                        b_over_a = (1.0 - gmag)/(1.0 + gmag)
                    else:
                        assert e1 is not None and e2 is not None, "need to specify ellipticty"
                        self.phi = 0.5 * np.arctan2(e2, e1)
                        emag = np.sqrt(e1**2.0 + e2**2.0)
                        b_over_a = np.sqrt((1.0 - emag)/(1.0 + emag))

                self.a = self.r_e / np.sqrt(b_over_a)
                self.b = self.r_e * np.sqrt(b_over_a)
            cph = np.cos(self.phi)
            sph = np.sin(self.phi)
            self.C11 = (cph/self.a)**2 + (sph/self.b)**2
            self.C12 = 0.5 * (1.0/self.a**2 - 1.0/self.b**2) * np.sin(2.0 * self.phi)
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
        ''' Return the surface brightness at (y,x).'''
        xp = x - self.x0
        yp = y - self.y0
        exponent = self.C11*xp**2 + 2*self.C12*xp*yp + self.C22*yp**2
        exponent = exponent**(0.5/self.n)
        exponent *= -self.kappa
        return self.peak * np.exp(exponent)

    @staticmethod
    def compute_kappa(n):
        '''Compute Sersic exponent factor kappa from the Sersic index'''
        kguess = 1.9992*n - 0.3271
        return newton(lambda k: gammainc(2*n, k) - 0.5, kguess)

    @classmethod
    def compute_FWHM(cls, n, r_e, kappa=None):
        '''Compute the full-width at half maximum given input parameters.'''
        if kappa is None:
            kappa = cls.compute_kappa(n)
        return 2.0 * r_e * (np.log(2.0) / kappa)**n

    @classmethod
    def compute_flux(cls, n, r_e, peak, kappa=None):
        '''Compute flux integrated over all space given input parameters.'''
        if kappa is None:
            kappa = cls.compute_kappa(n)
        return (2.0 * np.pi) * peak * n * (kappa**(-2.0 * n)) * (r_e**2.0) * gamma(2.0 * n)

    @classmethod
    def compute_peak(cls, n, r_e, flux, kappa=None):
        '''Compute peak surface brightness from integrated flux and Sersic params.'''
        if kappa is None:
            kappa = cls.compute_kappa(n)
        fluxnorm = cls.compute_flux(n, r_e, 1.0, kappa=kappa)
        return flux/fluxnorm

if __name__ == '__main__':
    '''Creating a test case for the various ways of initializing Sersics'''
    y0 = 0.1
    x0 = 0.3
    n = 1.5
    peak = 1.0

    # a, b, phi
    phi = 0.1
    a = 0.2
    b = 0.5
    s1 = Sersic(y0, x0, n, peak=peak, a=a, b=b, phi=phi)
    print 'It is an error if following 9 or so values do not all match'
    print s1(0.2, 2.1)

    # C11, C12, C22
    C11 = (np.cos(phi)**2.0 / a**2.0 + np.sin(phi)**2.0 / b**2.0)
    C22 = (np.sin(phi)**2.0 / a**2.0 + np.cos(phi)**2.0 / b**2.0)
    C12 = 0.5 * (1.0/a**2.0 - 1.0/b**2.0) * np.sin(2.0 * phi)
    s2 = Sersic(y0, x0, n, peak=peak, C11=C11, C12=C12, C22=C22)
    print s2(0.2, 2.1)

    # r_e, b_over_a, phi
    r_e = np.sqrt(a * b)
    b_over_a = b / a
    s3 = Sersic(y0, x0, n, peak=peak, r_e=r_e, b_over_a=b_over_a, phi=phi)
    print s3(0.2, 2.1)

    # r_e, emag, phi
    emag = (a**2.0 - b**2.0) / (a**2.0 + b**2.0)
    s5 = Sersic(y0, x0, n, peak=peak, r_e=r_e, emag=emag, phi=phi)
    print s5(0.2, 2.1)

    # r_e, gmag, phi
    gmag = (a - b) / (a + b)
    s6 = Sersic(y0, x0, n, peak=peak, r_e=r_e, gmag=gmag, phi=phi)
    print s6(0.2, 2.1)

    # r_e, e1, e2
    e1 = emag * np.cos(2.0 * phi)
    e2 = emag * np.sin(2.0 * phi)
    s7 = Sersic(y0, x0, n, peak=peak, r_e=r_e, e1=e1, e2=e2)
    print s7(0.2, 2.1)

    # r_e, g1, g2
    g1 = gmag * np.cos(2.0 * phi)
    g2 = gmag * np.sin(2.0 * phi)
    s8 = Sersic(y0, x0, n, peak=peak, r_e=r_e, g1=g1, g2=g2)
    print s8(0.2, 2.1)

    # FWHM instead of r_e
    FWHM = 2 * r_e * (np.log(2.0) / Sersic.compute_kappa(n))**n
    s4 = Sersic(y0, x0, n, peak=peak, r_e=r_e, b_over_a=b_over_a, phi=phi)
    print s4(0.2, 2.1)

    # flux instead of peak
    flux=Sersic.compute_flux(n, r_e, peak)
    s9 = Sersic(y0, x0, n, flux=flux, a=a, b=b, phi=phi)
    print s9(0.2, 2.1)

    # make sure have access to r_e
    print 'It is an error if following 9 or so values do not all match'
    print s1.r_e
    print s2.r_e
    print s3.r_e
    print s4.r_e
    print s5.r_e
    print s6.r_e
    print s7.r_e
    print s8.r_e
    print s9.r_e
