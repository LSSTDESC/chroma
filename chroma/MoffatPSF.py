import hashlib
import numpy as np

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
            one_over_a_squared = 0.5 * (C11 + C22 + np.sqrt((C11 - C22)**2 + 4.0 * C12**2))
            one_over_b_squared = C11 + C22 - one_over_a_squared
            # there's degeneracy between a, b and phi at this point so enforce a > b
            if one_over_a_squared > one_over_b_squared:
                one_over_a_squared, one_over_b_squared = one_over_b_squared, one_over_a_squared
            self.a = np.sqrt(1.0 / one_over_a_squared)
            self.b = np.sqrt(1.0 / one_over_b_squared)
            self.alpha = np.sqrt(self.a * self.b)
            self.phi = 0.5 * np.arctan2(2.0 * C12 / (one_over_a_squared - one_over_b_squared),
                                        (C11 - C22) / (one_over_a_squared - one_over_b_squared))

        else:
            # goal for this block is to determine a, b, phi
            # first check the direct case
            if a is not None and b is not None and phi is not None:
                self.a = a
                self.b = b
                self.phi = phi
                self.alpha = np.sqrt(a * b)
            else: # now check a hierarchy of size & ellip possibilities
                # first the size must be either FWHM or r_e
                if FWHM is not None:
                    self.alpha = FWHM / (2.0 * np.sqrt(2.0**(1.0/self.beta) - 1.0))
                else:
                    assert alpha is not None, "need to specify a size parameter"
                    self.alpha = alpha
                # goal here is to determine the axis ratio b_over_a, and position angle phi
                if phi is not None: # must be doing a polar decomposition
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

                self.a = self.alpha / np.sqrt(b_over_a)
                self.b = self.alpha * np.sqrt(b_over_a)
            cph = np.cos(self.phi)
            sph = np.sin(self.phi)
            self.C11 = (cph/self.a)**2 + (sph/self.b)**2
            self.C12 = 0.5 * (1.0/self.a**2 - 1.0/self.b**2) * np.sin(2.0 * self.phi)
            self.C22 = (sph/self.a)**2 + (cph/self.b)**2

        det = self.C11 * self.C22 - self.C12**2.0
        self.norm = self.flux * (self.beta - 1.0) / (np.pi / np.sqrt(abs(det)))

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
