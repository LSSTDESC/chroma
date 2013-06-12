import copy

import numpy
import scipy
import lmfit

import chroma.utils

class BDGalTools(object):
    ''' Class to instantiate bulge+disk galaxies.'''
    def __init__(self, bd_engine):
        '''
        Arguments
        ---------
        bd_engine -- image creation engine.  Possible instances are located in ImageEngine.py
        '''
        self.bd_engine = bd_engine

    def get_ring_params(self, gparam0, gamma, beta):
        ''' Adjust bulge+disk parameters in gparam0 to reflect applied shear `gamma` and
        angle around the ring `beta` in a ring test.  Returned parameters are good both for
        creating the target image and for initializing the lmfit minimize routine.
        '''
        gparam1 = copy.deepcopy(gparam0)
        b_phi_ring = gparam0['b_phi'].value + beta/2.0
        d_phi_ring = gparam0['d_phi'].value + beta/2.0
        # bulge complex ellipticity
        b_c_ellip = gparam0['b_gmag'].value * \
          complex(numpy.cos(2.0 * b_phi_ring), numpy.sin(2.0 * b_phi_ring))
        # bulge sheared complex ellipticity
        b_s_c_ellip = chroma.utils.shear_galaxy(b_c_ellip, gamma)
        b_s_gmag = abs(b_s_c_ellip)
        b_s_phi = numpy.angle(b_s_c_ellip) / 2.0
        # disk complex ellipticity
        d_c_ellip = gparam0['d_gmag'].value * \
          complex(numpy.cos(2.0 * d_phi_ring), numpy.sin(2.0 * d_phi_ring))
        # disk sheared complex ellipticity
        d_s_c_ellip = chroma.utils.shear_galaxy(d_c_ellip, gamma)
        d_s_gmag = abs(d_s_c_ellip)
        d_s_phi = numpy.angle(d_s_c_ellip) / 2.0
        # radius rescaling
        rescale = numpy.sqrt(1.0 - abs(gamma)**2.0)

        gparam1['b_y0'].value \
          = gparam0['b_y0'].value * numpy.sin(beta / 2.0) \
          + gparam0['b_x0'].value * numpy.cos(beta / 2.0)
        gparam1['b_x0'].value \
          = gparam0['b_y0'].value * numpy.cos(beta / 2.0) \
          - gparam0['b_x0'].value * numpy.sin(beta / 2.0)
        gparam1['d_y0'].value \
          = gparam0['d_y0'].value * numpy.sin(beta / 2.0) \
          + gparam0['d_x0'].value * numpy.cos(beta / 2.0)
        gparam1['d_x0'].value \
          = gparam0['d_y0'].value * numpy.cos(beta / 2.0) \
          - gparam0['d_x0'].value * numpy.sin(beta / 2.0)
        gparam1['b_gmag'].value = b_s_gmag
        gparam1['d_gmag'].value = d_s_gmag
        gparam1['b_phi'].value = b_s_phi
        gparam1['d_phi'].value = d_s_phi
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * rescale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * rescale
        return gparam1

    def circularize(self, gparam0):
        gparam1 = copy.deepcopy(gparam0)
        gparam1['b_gmag'].value = 0.0
        gparam1['d_gmag'].value = 0.0
        return gparam1

    def set_FWHM(self, gparam0, FWHM, bulge_PSF, disk_PSF):
        gparam1 = copy.deepcopy(gparam0)
        def test_FWHM(scale):
            gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
            return self.bd_engine.get_FWHM(gparam1, bulge_PSF, disk_PSF)
        def resid(scale):
            return test_FWHM(scale) - FWHM
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return gparam1

    def set_AHM(gparam0, AHM, bulge_PSF, disk_PSF):
        gparam1 = copy.deepcopy(gparam0)
        def test_AHM(scale):
            gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
            return self.bd_engine.get_AHM(gparam1, bulge_PSF, disk_PSF)
        def resid(scale):
            return test_AHM(scale) - AHM
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return gparam1

    def set_r2(self, gparam0, r2, bulge_PSF, disk_PSF):
        gparam1 = copy.deepcopy(gparam0)
        def test_r2(scale):
            gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
            return self.bd_engine.get_r2(gparam1, bulge_PSF, disk_PSF)
        def resid(scale):
            return test_r2(scale) - r2
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return gparam1

    def set_uncvl_r2(self, gparma0, r2):
        gparam1 = copy.deepcopy(gparam0)
        def r2_gal(scale):
            gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
            return self.bd_engine.gal_uncvl_r2(gparam1)
        def resid(scale):
            return r2_gal(s) - r2
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['b_r_e'].value = gparam0['b_r_e'].value * scale
        gparam1['d_r_e'].value = gparam0['d_r_e'].value * scale
        return gparam1

class SGal(object):
    ''' Class to instantiate single-component Sersic galaxies.'''
    def __init__(self, s_engine):
        self.s_engine = s_engine

    def get_ring_params(self, gparam0, gamma, beta):
        gparam1 = copy.deepcopy(gparam0)
        phi_ring = gparam0['phi'].value + beta/2.0
        # complex ellipticity
        c_ellip = gparam0['gmag'].value * \
          complex(numpy.cos(2.0 * phi_ring), numpy.sin(2.0 * phi_ring))
        # sheared complex ellipticity
        s_c_ellip = chroma.utils.shear_galaxy(c_ellip, gamma)
        s_gmag = abs(s_c_ellip)
        s_phi = numpy.angle(s_c_ellip) / 2.0
        # radius rescaling
        rescale = numpy.sqrt(1.0 - abs(gamma)**2.0)

        gparam1['y0'].value \
          = gparam0['y0'].value * numpy.sin(beta / 2.0) \
          + gparam0['x0'].value * numpy.cos(beta / 2.0)
        gparam1['x0'].value \
          = gparam0['y0'].value * numpy.cos(beta / 2.0) \
          - gparam0['x0'].value * numpy.sin(beta / 2.0)
        gparam1['gmag'].value = s_gmag
        gparam1['phi'].value = s_phi
        gparam1['r_e'].value = gparam0['r_e'].value * rescale
        return gparam1

    def circularize(self, gparam0):
        gparam1 = copy.deepcopy(gparam0)
        gparam1['gmag'].value = 0.0
        return gparam1

    def set_FWHM(self, gparam0, FWHM, PSF):
        gparam1 = copy.deepcopy(gparam0)
        def test_FWHM(scale):
            gparam1['r_e'].value = gparam0['r_e'].value * scale
            return self.s_engine.get_FWHM(gparam1, PSF)
        def resid(scale):
            return test_FWHM(scale) - FWHM
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['r_e'].value = gparam0['r_e'] * scale
        return gparam1

    def set_AHM(self, gparam0, AHM, PSF):
        gparam1 = copy.deepcopy(gparam0)
        def test_AHM(scale):
            gparam1['r_e'].value = gparam0['r_e'].value * scale
            return self.s_engine.get_AHM(gparam1, PSF)
        def resid(scale):
            return test_AHM(scale) - AHM
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['r_e'].value = gparam0['r_e'] * scale
        return gparam1

    def set_r2(self, gparam0, r2, PSF):
        gparam1 = copy.deepcopy(gparam0)
        def test_r2(scale):
            gparam1['r_e'].value = gparam0['r_e'].value * scale
            return self.s_engine.get_r2(gparam1, PSF)
        def resid(scale):
            return test_r2(scale) - r2
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['r_e'].value = gparam0['r_e'] * scale
        return gparam1

    def set_uncvl_r2(self, gparam0, r2):
        gparam1 = copy.deepcopy(gparam0)
        def test_r2(scale):
            gparam1['r_e'].value = gparam0['r_e'].value * scale
            return self.s_engine.get_uncvl_r2(gparam1, PSF)
        def resid(scale):
            return test_r2(scale) - r2
        scale = scipy.optimize.newton(resid, 1.0)
        gparam1['r_e'].value = gparam0['r_e'] * scale
        return gparam1