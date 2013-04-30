import copy

import numpy
import scipy
import lmfit

import chroma.utils

class BDGal(object):
    def __init__(self, gparam0, wave, bulge_photons, disk_photons,
                 PSF_model=None, PSF_kwargs=None,
                 bd_engine=None):
        self.gparam0 = gparam0
        self.wave = wave
        self.bulge_photons = bulge_photons / scipy.integrate.simps(bulge_photons, wave)
        self.disk_photons = disk_photons / scipy.integrate.simps(disk_photons, wave)
        self.PSF_model = PSF_model
        self.PSF_kwargs = PSF_kwargs
        self.composite_photons = self.bulge_photons * self.gparam0['b_flux'].value \
          + self.disk_photons * self.gparam0['d_flux'].value
        self.bd_engine = bd_engine
        self.build_PSFs()

    def build_PSFs(self):
        self.bulge_PSF = self.PSF_model(self.wave, self.bulge_photons, **self.PSF_kwargs)
        self.disk_PSF = self.PSF_model(self.wave, self.disk_photons, **self.PSF_kwargs)
        self.composite_PSF = self.PSF_model(self.wave, self.composite_photons, **self.PSF_kwargs)

    def build_circ_PSF(self):
        PSF_kwargs2 = copy.deepcopy(self.PSF_kwargs)
        PSF_kwargs2['ellipticity']=0.0
        PSF_kwargs2['phi']=0.0
        self.circ_PSF = self.PSF_model(self.wave, self.composite_photons, **PSF_kwargs2)

    def set_FWHM_ratio(self, rpg):
        '''Set the effective radii of the bulge+disk galaxy specified in `self.gparam0` such that the
        ratio of the FWHM of the PSF-convolved galaxy image is `rpg` times the FWHM of the PSF
        itself.  The galaxy is circularized and centered at the origin for this computation
        (ellip -> 0.0) and (x0, y0 -> 0.0, 0.0), and the PSF derived from the composite spectrum and
        set to be circular.
        '''
        FWHM_PSF = self.bd_engine.PSF_FWHM(self.circ_PSF)
        gparam1 = copy.deepcopy(self.gparam0)
        gparam1['b_gmag'].value = 0.0
        gparam1['b_x0'].value = 0.0
        gparam1['b_y0'].value = 0.0
        gparam1['d_gmag'].value = 0.0
        gparam1['d_x0'].value = 0.0
        gparam1['d_y0'].value = 0.0
        def FWHM_gal(scale):
            gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            return self.bd_engine.bdcvl_FWHM(gparam1, self.circ_PSF, self.circ_PSF)
        def f(scale):
            return FWHM_gal(scale) - rpg * FWHM_PSF
        scale = scipy.optimize.newton(f, 1.0)
        self.gparam0['b_r_e'].value *= scale
        self.gparam0['d_r_e'].value *= scale

    def gen_target_image(self, gamma, beta):
        gparam1 = self.gen_init_param(gamma, beta)
        return self.bd_engine.bd_image(gparam1, self.bulge_PSF, self.disk_PSF)

    def gen_init_param(self, gamma, beta):
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

    def measure_ellip(self, target_image, init_param):
        def resid(param):
            im = self.bd_engine.bd_image(param, self.composite_PSF, self.composite_PSF)
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['d_gmag'].value
        phi = result.params['d_phi'].value
        c_ellip = gmag * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip
