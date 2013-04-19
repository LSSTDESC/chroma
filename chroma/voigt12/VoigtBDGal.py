import copy

import numpy
from scipy.integrate import simps
from scipy.optimize import newton
from lmfit import minimize, report_errors

import chroma.SBProfile
from chroma import BDGal
from EuclidPSF import EuclidPSF
from ImageFactory import ImageFactory

class VoigtBDGal(BDGal):
    def __init__(self, gparam0, wave, bulge_photons, disk_photons,
                 PSF_kwargs=None, im_fac=None):
        if im_fac is None:
            im_fac = ImageFactory()
        self.im_fac = im_fac
        self.oversample_factor = im_fac.oversample_factor
        super(VoigtBDGal, self).__init__(gparam0, wave, bulge_photons, disk_photons,
                                         PSF_model=EuclidPSF, PSF_kwargs=PSF_kwargs)

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

    def gal_image(self, gparam, bulge_PSF, disk_PSF):
        '''Use image_factory `self.im_fac` to make a galaxy image from params in gparam and using
        the bulge and disk psfs `bulge_PSF` and `disk_PSF`.

        Arguments
        ---------
        gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
                  `b_` prefix for bulge, `d_` prefix for disk.
                  Suffixes are all init arguments for the Sersic object.

        Note that you can specify the composite PSF `composite_PSF` for both bulge and disk PSF when
        using during ringtest fits.
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

    def PSF_image(self, PSF):
        return self.im_fac.get_PSF_image(self.circ_PSF)

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
