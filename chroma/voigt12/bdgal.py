import copy

import numpy
from scipy.integrate import simps
from scipy.optimize import newton
from lmfit import minimize, report_errors

import chroma
from EuclidPSF import EuclidPSF

class bdgal(object):
    def __init__(self, gparam0, wave, bulge_photons, disk_photons, PSF_ellip, PSF_phi, im_fac):
        self.gparam0 = gparam0
        self.wave = wave
        self.bulge_photons = bulge_photons/simps(bulge_photons, wave)
        self.disk_photons = disk_photons/simps(disk_photons, wave)
        self.PSF_ellip = PSF_ellip
        self.PSF_phi = PSF_phi
        self.im_fac = im_fac

        self.composite_photons = self.bulge_photons * self.gparam0['b_flux'].value \
          + self.disk_photons * self.gparam0['d_flux'].value
        self.build_PSFs()

    def build_PSFs(self):
        ''' Build bulge, disk, and composite PSFs from SEDs.'''
        self.bulge_PSF = EuclidPSF(self.wave, self.bulge_photons,
                                   ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.disk_PSF = EuclidPSF(self.wave, self.disk_photons,
                                  ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.composite_PSF = EuclidPSF(self.wave, self.composite_photons,
                                       ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.circ_PSF = EuclidPSF(self.wave, self.composite_photons,
                                  ellipticity=0.0, phi=0.0)

    def set_FWHM_ratio(self, rpg):
        '''Set the effective radii of the bulge+disk galaxy specified in `gparam` such that the
        ratio of the FWHM of the PSF-convolved galaxy image is `rpg` times the FWHM of the PSF
        itself. The galaxy is circularized and centered at the origin for this computation
        (ellip -> 0.0) and (x0, y0 -> 0.0, 0.0).  The supplied PSF `circ_c_PSF` is assumed to be
        circularly symmetric.
        '''
        FWHM_PSF = chroma.utils.FWHM(self.im_fac.get_PSF_image(self.circ_PSF),
                                     scale=self.im_fac.oversample_factor)
        gparam2 = copy.deepcopy(self.gparam0)
        gparam2['b_gmag'].value = 0.0
        gparam2['b_x0'].value = 0.0
        gparam2['b_y0'].value = 0.0
        gparam2['d_gmag'].value = 0.0
        gparam2['d_x0'].value = 0.0
        gparam2['d_y0'].value = 0.0
        def FWHM_gal(scale):
            gparam2['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam2['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            image = self.gal_overimage(gparam2, self.circ_PSF, self.circ_PSF)
            return chroma.utils.FWHM(image, scale=self.im_fac.oversample_factor)
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
        bulge = chroma.SBProfile.Sersic(gparam['b_y0'].value,
                                        gparam['b_x0'].value,
                                        gparam['b_n'].value,
                                        r_e=gparam['b_r_e'].value,
                                        gmag=gparam['b_gmag'].value,
                                        phi=gparam['b_phi'].value,
                                        flux=gparam['b_flux'].value)
        disk = chroma.SBProfile.Sersic(gparam['d_y0'].value,
                                       gparam['d_x0'].value,
                                       gparam['d_n'].value,
                                       r_e=gparam['d_r_e'].value,
                                       gmag=gparam['d_gmag'].value,
                                       phi=gparam['d_phi'].value,
                                       flux=gparam['d_flux'].value)
        return self.im_fac.get_overimage([(bulge, b_PSF), (disk, d_PSF)])

    def target_image_fn_generator(self):
        '''Return a function that can be passed to `ringtest()` which produces a target image as a
        function of applied shear `gamma` and angle along the ring `beta`.

        Arguments are passed through to `gal_image()`
        '''

        gen_init_param = self.init_param_generator()
        def f(gamma, beta):
            gparam1 = gen_init_param(gamma, beta)
            return self.gal_image(gparam1, self.bulge_PSF, self.disk_PSF)
        return f

    def init_param_generator(self):
        '''Return a function that can be passed to `ringtest()` which generates initial conditions
        for a fit to the "true image" using the incorrect PSF as a function of applied shear
        `gamma` and angle along the ring `beta`.
        '''
        # parameters which will change with applied shear gamma or angle along ring beta
        # extract their initial values here
        b_y0 = self.gparam0['b_y0'].value
        b_x0 = self.gparam0['b_x0'].value
        b_gmag = self.gparam0['b_gmag'].value
        b_phi = self.gparam0['b_phi'].value
        b_r_e = self.gparam0['b_r_e'].value
        d_y0 = self.gparam0['d_y0'].value
        d_x0 = self.gparam0['d_x0'].value
        d_gmag = self.gparam0['d_gmag'].value
        d_phi = self.gparam0['d_phi'].value
        d_r_e = self.gparam0['d_r_e'].value

        def gen_init_param(gamma, beta):
            gparam1 = copy.deepcopy(self.gparam0)
            b_phi_ring = b_phi + beta/2.0
            d_phi_ring = d_phi + beta/2.0
            # bulge complex ellipticity
            b_c_ellip = b_gmag * complex(numpy.cos(2.0 * b_phi_ring), numpy.sin(2.0 * b_phi_ring))
            # bulge sheared complex ellipticity
            b_s_c_ellip = chroma.utils.shear_galaxy(b_c_ellip, gamma)
            b_s_gmag = abs(b_s_c_ellip)
            b_s_phi = numpy.angle(b_s_c_ellip) / 2.0
            # disk complex ellipticity
            d_c_ellip = d_gmag * complex(numpy.cos(2.0 * d_phi_ring), numpy.sin(2.0 * d_phi_ring))
            # disk sheared complex ellipticity
            d_s_c_ellip = chroma.utils.shear_galaxy(d_c_ellip, gamma)
            d_s_gmag = abs(d_s_c_ellip)
            d_s_phi = numpy.angle(d_s_c_ellip) / 2.0
            # radius rescaling
            rescale = numpy.sqrt(1.0 - abs(gamma)**2.0)

            gparam1['b_y0'].value = b_y0 * numpy.sin(beta / 2.0) + b_x0 * numpy.cos(beta / 2.0)
            gparam1['b_x0'].value = b_y0 * numpy.cos(beta / 2.0) - b_x0 * numpy.sin(beta / 2.0)
            gparam1['d_y0'].value = d_y0 * numpy.sin(beta / 2.0) + d_x0 * numpy.cos(beta / 2.0)
            gparam1['d_x0'].value = d_y0 * numpy.cos(beta / 2.0) - d_x0 * numpy.sin(beta / 2.0)
            gparam1['b_gmag'].value = b_s_gmag
            gparam1['d_gmag'].value = d_s_gmag
            gparam1['b_phi'].value = b_s_phi
            gparam1['d_phi'].value = d_s_phi
            gparam1['b_r_e'].value = b_r_e * rescale
            gparam1['d_r_e'].value = d_r_e * rescale
            return gparam1
        return gen_init_param

    def gal_image(self, gparam, bulge_PSF, disk_PSF):
        '''Use image_factory `im_fac` to make a galaxy image from params in gparam and using
        the bulge and disk psfs `b_PSF` and `d_PSF`.

        Arguments
        ---------
        gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
                  `b_` prefix for bulge, `d_` prefix for disk.
                  Suffixes are all init arguments for the Sersic object.

        Note that you can specify the composite PSF `c_PSF` for both bulge and disk PSF when using
        during ringtest fits.
        '''
        bulge = chroma.SBProfile.Sersic(gparam['b_y0'].value,
                                        gparam['b_x0'].value,
                                        gparam['b_n'].value,
                                        r_e=gparam['b_r_e'].value,
                                        gmag=gparam['b_gmag'].value,
                                        phi=gparam['b_phi'].value,
                                        flux=gparam['b_flux'].value)
        disk = chroma.SBProfile.Sersic(gparam['d_y0'].value,
                                       gparam['d_x0'].value,
                                       gparam['d_n'].value,
                                       r_e=gparam['d_r_e'].value,
                                       gmag=gparam['d_gmag'].value,
                                       phi=gparam['d_phi'].value,
                                       flux=gparam['d_flux'].value)
        return self.im_fac.get_image([(bulge, bulge_PSF), (disk, disk_PSF)])

    def ellip_measurement_generator(self):
        '''Return a function that can be passed to `ringtest()` which computes the best-fit
        ellipticity of a sheared galaxy along the ring as a function of the "true" image
        `target_image` and some initial parameters for the fit.
        '''
        def measure_ellip(target_image, init_param):
            def resid(param):
                im = self.gal_image(param, self.composite_PSF, self.composite_PSF)
                return (im - target_image).flatten()
            result = minimize(resid, init_param)
            gmag = result.params['d_gmag'].value
            phi = result.params['d_phi'].value
            c_ellip = gmag * complex(numpy.cos(2.0  * phi), numpy.sin(2.0 * phi))
            return c_ellip
        return measure_ellip

# if __name__ == '__main__':
#     # Returns some objects that can be passed to ringtest for interactively checking that the code
#     # is working as expected.
#     from lmfit import Parameter, Parameters, Minimizer, minimize

#     gparam = Parameters()
#     # bulge
#     gparam.add('b_x0', value=2.1)
#     gparam.add('b_y0', value=3.3)
#     gparam.add('b_n', value=4.0, vary=False)
#     gparam.add('b_r_e', value=2.7)
#     gparam.add('b_flux', value=0.25)
#     gparam.add('b_gmag', value=0.4)
#     gparam.add('b_phi', value=0.0)
#     # disk
#     gparam.add('d_x0', expr='b_x0')
#     gparam.add('d_y0', expr='b_y0')
#     gparam.add('d_n', value=1.0, vary=False)
#     gparam.add('d_r_e', value=2.7 * 1.1)
#     gparam.add('d_flux', expr='1.0 - b_flux')
#     gparam.add('d_gmag', expr='b_gmag')
#     gparam.add('d_phi', expr='b_phi')

#     dummyfit = Minimizer(lambda x: 0, gparam)
#     dummyfit.prepare_fit()

#     filter_file = '../data/filters/voigt12_350.dat'
#     bulge_SED_file = '../data/SEDs/CWW_E_ext.ascii'
#     disk_SED_file = '../data/SEDs/CWW_Sbc_ext.ascii'

#     b_PSF, d_PSF, c_PSF, circ_c_PSF = build_PSFs(filter_file, 0.25, bulge_SED_file,
#                                                  disk_SED_file, 0.9, PSF_ellip=0.05)
#     # im_fac = VoigtImageFactory(size=51, oversample_factor=3)
#     im_fac = VoigtImageFactory()
#     set_FWHM_ratio(gparam, 1.4, circ_c_PSF, im_fac)
#     gen_target_image = target_image_fn_generator(gparam, b_PSF, d_PSF, im_fac)
#     gen_init_param = init_param_generator(gparam)
#     measure_ellip = ellip_measurement_generator(c_PSF, im_fac)
