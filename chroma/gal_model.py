import copy

import numpy
import scipy
import lmfit

import chroma.utils

class BDGal(object):
    ''' Class to instantiate bulge+disk galaxies.'''
    def __init__(self, gparam0, bd_engine):
        '''
        Arguments
        ---------
        gparam0 -- lmfit.Parameters object describing bulge+disk galaxy.  Params for each of the
                   bulge and disk components include:
                       `x0`, `y0` -- center of Sersic
                       `r_e` -- half light radius of Sersic
                       `gmag` -- magnitude of ellipticity
                       `phi` -- position angle of ellipticity
                       `flux` -- flux of component
                   Bulge params carry a `b_` prefix, disk params carry a `d_` prefix.
        bd_engine -- image creation engine.  Possible instances are located in ImageEngine.py
        '''
        self.gparam0 = gparam0
        self.bd_engine = bd_engine

    def get_image(self, bulge_PSF, disk_PSF):
        return self.bd_engine.get_image(self.gparam0, bulge_PSF, disk_PSF)

    def get_uncvl_image(self):
        return self.bd_engine.get_uncvl_image(self.gparam0)

    def get_ring_params(self, gamma, beta):
        ''' Adjust bulge+disk parameters in self.gparam0 to reflect applied shear `gamma` and
        angle around the ring `beta` in a ring test.  Returned parameters are good both for
        creating the target image and for initializing the lmfit minimize routine.
        '''
        gparam1 = copy.deepcopy(self.gparam0)
        b_phi_ring = self.gparam0['b_phi'].value + beta/2.0
        d_phi_ring = self.gparam0['d_phi'].value + beta/2.0
        # bulge complex ellipticity
        b_c_ellip = self.gparam0['b_gmag'].value * \
          complex(numpy.cos(2.0 * b_phi_ring), numpy.sin(2.0 * b_phi_ring))
        # bulge sheared complex ellipticity
        b_s_c_ellip = chroma.utils.shear_galaxy(b_c_ellip, gamma)
        b_s_gmag = abs(b_s_c_ellip)
        b_s_phi = numpy.angle(b_s_c_ellip) / 2.0
        # disk complex ellipticity
        d_c_ellip = self.gparam0['d_gmag'].value * \
          complex(numpy.cos(2.0 * d_phi_ring), numpy.sin(2.0 * d_phi_ring))
        # disk sheared complex ellipticity
        d_s_c_ellip = chroma.utils.shear_galaxy(d_c_ellip, gamma)
        d_s_gmag = abs(d_s_c_ellip)
        d_s_phi = numpy.angle(d_s_c_ellip) / 2.0
        # radius rescaling
        rescale = numpy.sqrt(1.0 - abs(gamma)**2.0)

        gparam1['b_y0'].value \
          = self.gparam0['b_y0'].value * numpy.sin(beta / 2.0) \
          + self.gparam0['b_x0'].value * numpy.cos(beta / 2.0)
        gparam1['b_x0'].value \
          = self.gparam0['b_y0'].value * numpy.cos(beta / 2.0) \
          - self.gparam0['b_x0'].value * numpy.sin(beta / 2.0)
        gparam1['d_y0'].value \
          = self.gparam0['d_y0'].value * numpy.sin(beta / 2.0) \
          + self.gparam0['d_x0'].value * numpy.cos(beta / 2.0)
        gparam1['d_x0'].value \
          = self.gparam0['d_y0'].value * numpy.cos(beta / 2.0) \
          - self.gparam0['d_x0'].value * numpy.sin(beta / 2.0)
        gparam1['b_gmag'].value = b_s_gmag
        gparam1['d_gmag'].value = d_s_gmag
        gparam1['b_phi'].value = b_s_phi
        gparam1['d_phi'].value = d_s_phi
        gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * rescale
        gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * rescale
        return gparam1

    def circularize(self):
        gparam1 = copy.deepcopy(self.gparam0)
        gparam1['b_gmag'].value = 0.0
        gparam1['d_gmag'].value = 0.0
        return BDGal(gparam1, self.bd_engine)

    def get_FWHM(self, bulge_PSF, disk_PSF):
        return self.bd_engine.get_FWHM(self.gparam0, bulge_PSF, disk_PSF)

    def set_FWHM(self, FWHM, bulge_PSF, disk_PSF):
        gparam1 = copy.deepcopy(self.gparam0)
        def test_FWHM(scale):
            gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            return self.bd_engine.get_FWHM(gparam1, bulge_PSF, disk_PSF)
        def resid(scale):
            return test_FWHM(scale) - FWHM
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
        return BDGal(gparam1, self.bd_engine)

    def set_AHM(self, AHM, bulge_PSF, disk_PSF):
        gparam1 = copy.deepcopy(self.gparam0)
        def test_AHM(scale):
            gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            return self.bd_engine.get_AHM(gparam1, bulge_PSF, disk_PSF)
        def resid(scale):
            return test_AHM(scale) - AHM
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return BDGal(gparam1, self.bd_engine)

    def set_r2(self, r2, bulge_PSF, disk_PSF):
        gparam1 = copy.deepcopy(self.gparam0)
        def test_r2(scale):
            gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            return self.bd_engine.get_r2(gparam1, bulge_PSF, disk_PSF)
        def resid(scale):
            return test_r2(scale) - r2
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return BDGal(gparam1, self.bd_engine)

    def set_uncvl_r2(self, r2):
        gparam1 = copy.deepcopy(self.gparam0)
        def r2_gal(scale):
            gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            return self.bd_engine.gal_uncvl_r2(gparam1)
        def resid(scale):
            return r2_gal(s) - r2
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return BDGal(gparam1, self.bd_engine)

class SGal(object):
    ''' Class to instantiate single-component Sersic galaxies.'''
    def __init__(self, gparam0, s_engine):
        self.gparam0 = gparam0
        self.s_engine = s_engine

    def set_circ_FWHM(self, target_FWHM, PSF):
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

    def set_uncvl_r2(self, target_r2):
        gparam1 = copy.deepcopy(self.gparam0)
        def r2_gal(scale):
            gparam1['r_e'].value = self.gparam0['r_e'].value * scale
            return self.s_engine.gal_uncvl_r2(gparam1)
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
