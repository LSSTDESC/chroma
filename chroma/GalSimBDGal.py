import copy

import numpy
from scipy.integrate import simps
from scipy.optimize import newton
from lmfit import minimize, report_errors
import galsim

import chroma

def GSEuclidPSF(wave, photons, ellipticity=0.0, phi=0.0):
    hlr = lambda wave: 0.7 * (wave / 520.0)**0.6 # pixels
    mpsfs = []
    photons /= simps(photons, wave)
    for w, p in zip(wave, photons):
        mpsfs.append(galsim.Gaussian(flux=p, half_light_radius=hlr(w)))
    PSF = galsim.Add(mpsfs)
    beta = phi * galsim.radians
    PSF.applyShear(g=ellipticity, beta=beta)
    return PSF

class GalSimBDGal(object):
    def __init__(self, gparam0, wave, bulge_photons, disk_photons, PSF_ellip, PSF_phi):
        self.gparam0 = gparam0
        self.wave = wave
        self.bulge_photons = bulge_photons / simps(bulge_photons, wave)
        self.disk_photons = disk_photons / simps(disk_photons, wave)
        self.PSF_ellip = PSF_ellip
        self.PSF_phi = PSF_phi

        self.composite_photons = self.bulge_photons * self.gparam0['b_flux'].value \
          + self.disk_photons * self.gparam0['d_flux'].value
        self.build_PSFs()

    def build_PSFs(self):
        self.bulge_PSF = GSEuclidPSF(self.wave, self.bulge_photons,
                                     ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.disk_PSF = GSEuclidPSF(self.wave, self.disk_photons,
                                    ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.composite_PSF = GSEuclidPSF(self.wave, self.composite_photons,
                                         ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.circ_PSF = GSEuclidPSF(self.wave, self.composite_photons,
                                    ellipticity=0.0, phi=0.0)

    def set_FWHM_ratio(self, rpg):
        '''Set the effective radii of the bulge+disk galaxy specified in `self.gparam0` such that the
        ratio of the FWHM of the PSF-convolved galaxy image is `rpg` times the FWHM of the PSF
        itself.  The galaxy is circularized and centered at the origin for this computation
        (ellip -> 0.0) and (x0, y0 -> 0.0, 0.0), and the PSF derived from the composite spectrum and
        set to be circular.
        '''
        PSF_im = galsim.ImageD(119,119)
        self.circ_PSF.draw(image=PSF_im, dx=1.0/7)
        FWHM_PSF = chroma.utils.FWHM(PSF_im.array, scale=7.0)
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
            image = self.gal_overimage(gparam1, self.circ_PSF, self.circ_PSF)
            return chroma.utils.FWHM(image, scale=7.0)
        def f(scale):
            return FWHM_gal(scale) - rpg * FWHM_PSF
        scale = newton(f, 1.0)
        self.gparam0['b_r_e'].value *= scale
        self.gparam0['d_r_e'].value *= scale

    def gal_overimage(self, gparam, b_PSF, d_PSF):
        '''Compute oversampled galaxy image.  Similar to `gal_image()`.

        Useful for computing FWHM of galaxy image at higher resolution than available from just
        `gal_image()`.
        '''
        bulge = galsim.Sersic(n=gparam['b_n'].value, half_light_radius=gparam['b_r_e'].value)
        bulge.applyShift(gparam['b_x0'].value, gparam['b_y0'].value)
        bulge.applyShear(g=gparam['b_gmag'].value, beta=gparam['b_phi'].value * galsim.radians)
        bulge.setFlux(gparam['b_flux'].value)
        disk = galsim.Sersic(n=gparam['d_n'].value, half_light_radius=gparam['d_r_e'].value)
        disk.applyShift(gparam['d_x0'].value, gparam['d_y0'].value)
        disk.applyShear(g=gparam['d_gmag'].value, beta=gparam['d_phi'].value * galsim.radians)
        disk.setFlux(gparam['d_flux'].value)
        bulge_cvl = galsim.Convolve(bulge, self.bulge_PSF)
        disk_cvl = galsim.Convolve(disk, self.disk_PSF)
        gal = bulge_cvl + disk_cvl
        gal_overim = galsim.ImageD(119,119)
        gal.draw(image=gal_overim, dx=1./7)
        return gal_overim.array

    def gen_target_image(self, gamma, beta):
        gparam1 = self.gen_init_param(gamma, beta)
        return self.gal_image(gparam1, self.bulge_PSF, self.disk_PSF)

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

    def gal_image(self, gparam, bulge_PSF, disk_PSF):
        '''Use galsim to make a galaxy image from params in gparam and using the bulge and disk
        PSFs `bulge_PSF` and `disk_PSF`.

        Arguments
        ---------
        gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
                  `b_` prefix for bulge, `d_` prefix for disk.
                  Suffixes are all init arguments for the Sersic object.

        Note that you can specify the composite PSF `c_PSF` for both bulge and disk PSF when using
        during ringtest fits.
        '''
        bulge = galsim.Sersic(n=gparam['b_n'].value, half_light_radius=gparam['b_r_e'].value)
        bulge.applyShift(gparam['b_x0'].value, gparam['b_y0'].value)
        bulge.applyShear(g=gparam['b_gmag'].value, beta=gparam['b_phi'].value * galsim.radians)
        bulge.setFlux(gparam['b_flux'].value)
        disk = galsim.Sersic(n=gparam['d_n'].value, half_light_radius=gparam['d_r_e'].value)
        disk.applyShift(gparam['d_x0'].value, gparam['d_y0'].value)
        disk.applyShear(g=gparam['d_gmag'].value, beta=gparam['d_phi'].value * galsim.radians)
        disk.setFlux(gparam['d_flux'].value)
        bulge_cvl = galsim.Convolve(bulge, self.bulge_PSF)
        disk_cvl = galsim.Convolve(disk, self.disk_PSF)
        gal = bulge_cvl + disk_cvl
        gal_im = galsim.ImageD(15, 15)
        gal.draw(image=gal_im, dx=1.0)
        return gal_im.array

    def measure_ellip(self, target_image, init_param):
        def resid(param):
            im = self.gal_image(param, self.composite_PSF, self.composite_PSF)
            return (im - target_image).flatten()
        result = minimize(resid, init_param)
        gmag = result.params['d_gmag'].value
        phi = result.params['d_phi'].value
        c_ellip = gmag * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip
