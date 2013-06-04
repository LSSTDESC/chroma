import copy

import numpy
import scipy
import lmfit

import chroma.utils

class SGal(object):
    ''' Class to instantiate single-component Sersic galaxies.'''
    def __init__(self, gparam0, s_engine):
        self.gparam0 = gparam0
        self.s_engine = s_engine

    def set_cvl_FWHM(self, target_FWHM, PSF):
        gparam1 = copy.deepcopy(self.gparam0)
        gparam1['gmag'].value = 0.0
        gparam1['x0'].value = 0.0
        gparam1['y0'].value = 0.0
        def FWHM_gal(scale):
            gparam1['r_e'].value = self.gparam0['r_e'].value * scale
            return self.s_engine.galcvl_FWHM(gparam1, PSF)
        def f(scale):
            return FWHM_gal(scale) - FWHM_PSF
        scale = scipy.optimize.newton(f, 1.0)
        self.gparam0['r_e'].value *= scale

    def set_r2(self, target_r2):
        gparam1 = copy.deepcopy(self.gparam0)
        def r2_gal(scale):
            gparam1['r_e'].value = self.gparam0['r_e'].value * scale
            return self.s_engine.gal_r2(gparam1)
        scale = scipy.optimize.newton(lambda s: r2_gal(s) - target_r2, 1.0)
        self.gparam0['r_e'].value = self.gparam0['r_e'].value * scale

    def gen_init_param(self, gamma, beta):
        ''' Adjust bulge+disk parameters in self.gparam0 to reflect applied shear `gamma` and
        angle around the ring `beta` in a ring test.  Returned parameters are good both for
        creating the target image and for initializing the lmfit minimize routine.
        '''
        gparam1 = copy.deepcopy(self.gparam0)
        phi_ring = self.gparam0['phi'].value + beta/2.0
        # complex ellipticity
        c_ellip = self.gparam0['gmag'].value * \
          complex(numpy.cos(2.0 * phi_ring), numpy.sin(2.0 * phi_ring))
        # sheared complex ellipticity
        s_c_ellip = chroma.utils.shear_galaxy(c_ellip, gamma)
        s_gmag = abs(s_c_ellip)
        s_phi = numpy.angle(s_c_ellip) / 2.0
        # radius rescaling
        rescale = numpy.sqrt(1.0 - abs(gamma)**2.0)

        gparam1['y0'].value \
          = self.gparam0['y0'].value * numpy.sin(beta / 2.0) \
          + self.gparam0['x0'].value * numpy.cos(beta / 2.0)
        gparam1['x0'].value \
          = self.gparam0['y0'].value * numpy.cos(beta / 2.0) \
          - self.gparam0['x0'].value * numpy.sin(beta / 2.0)
        gparam1['gmag'].value = s_gmag
        gparam1['phi'].value = s_phi
        gparam1['r_e'].value = self.gparam0['r_e'].value * rescale
        return gparam1

    def gen_target_image(self, gamma, beta, PSF):
        ''' Generate a target "truth" image for ring test.

        Arguments
        ---------
        gamma -- the input shear for the ring test.  Complex number.
        beta -- angle around the ellipticity ring.
        '''
        gparam1 = self.gen_init_param(gamma, beta)
        return self.s_engine.get_image(gparam1, PSF)
