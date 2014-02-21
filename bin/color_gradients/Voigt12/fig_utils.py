import numpy
import scipy.integrate
import lmfit

import _mypath
import chroma

def fiducial_galaxy():
    '''Bulge + disk parameters of the fiducial galaxy described in Voigt+12.'''
    gparam = lmfit.Parameters()
    #bulge
    gparam.add('b_x0', value=0.1)
    gparam.add('b_y0', value=0.3)
    gparam.add('b_n', value=4.0, vary=False)
    gparam.add('b_r_e', value=1.1 * 1.1)
    gparam.add('b_flux', value=0.25)
    gparam.add('b_gmag', value=0.2)
    gparam.add('b_phi', value=0.0)
    #disk
    gparam.add('d_x0', expr='b_x0')
    gparam.add('d_y0', expr='b_y0')
    gparam.add('d_n', value=1.0, vary=False)
    gparam.add('d_r_e', value=1.1)
    gparam.add('d_flux', expr='1.0 - b_flux')
    gparam.add('d_gmag', expr='b_gmag')
    gparam.add('d_phi', expr='b_phi')
    #initialize constrained variables
    dummyfit = lmfit.Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()
    return gparam

def PSF_SEDs(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift):
    # get surviving photons given filter and SEDs
    bulge_wave, bulge_photons = chroma.utils.get_photons(bulge_SED_file, filter_file, redshift)
    disk_wave, disk_photons = chroma.utils.get_photons(disk_SED_file, filter_file, redshift)
    bulge_photons /= scipy.integrate.simps(bulge_photons, bulge_wave)
    disk_photons /= scipy.integrate.simps(disk_photons, disk_wave)
    composite_wave = numpy.union1d(bulge_wave, disk_wave)
    composite_wave.sort()
    composite_photons = numpy.interp(composite_wave, bulge_wave, bulge_photons,
                                     left=0.0, right=0.0) * gparam['b_flux'].value \
                        + numpy.interp(composite_wave, disk_wave, disk_photons,
                                       left=0.0, right=0.0) * gparam['d_flux'].value
    waves = bulge_wave, disk_wave, composite_wave
    photonses = bulge_photons, disk_photons, composite_photons
    return waves, photonses

def measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                        PSF_ellip, PSF_phi,
                        PSF_model, bd_engine):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''

    waves, photonses = PSF_SEDs(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift)
    bulge_wave, disk_wave, composite_wave = waves
    bulge_photons, disk_photons, composite_photons = photonses

    # construct PSFs from surviving photons
    bulge_PSF = PSF_model(bulge_wave, bulge_photons, ellipticity=PSF_ellip, phi=PSF_phi)
    disk_PSF = PSF_model(disk_wave, disk_photons, ellipticity=PSF_ellip, phi=PSF_phi)
    composite_PSF = PSF_model(composite_wave, composite_photons, ellipticity=PSF_ellip, phi=PSF_phi)
    circ_bulge_PSF = PSF_model(bulge_wave, bulge_photons)
    circ_disk_PSF = PSF_model(disk_wave, disk_photons)
    circ_composite_PSF = PSF_model(composite_wave, composite_photons)

    # create galaxy and adjust effective radii such that
    # FWHM(gal convolved with PSF) / FWHM(PSF) = 1.4
    # make size adjustment assuming circularized galaxy and PSFs
    bdtool = chroma.GalTools.BDGalTool(bd_engine)
    PSF_FWHM = bd_engine.get_PSF_FWHM(circ_composite_PSF)
    circ_gparam = bdtool.circularize(gparam)
    circ_gparam = bdtool.set_FWHM(circ_gparam, 1.4 * PSF_FWHM, circ_bulge_PSF, circ_disk_PSF)
    gparam['b_r_e'] = circ_gparam['b_r_e']
    gparam['d_r_e'] = circ_gparam['d_r_e']

    # generate target image using ringed gparam and PSFs
    def gen_target_image(gamma, beta):
        ring_gparam = bdtool.get_ring_params(gparam, gamma, beta)
        return bd_engine.get_image(ring_gparam, bulge_PSF, disk_PSF)

    # function to measure ellipticity of target_image by trying to match the pixels
    # but using the "wrong" PSF (the composite PSF for both bulge and disk).
    def measure_ellip(target_image, init_param):
        def resid(param):
            im = bd_engine.get_image(param, composite_PSF, composite_PSF)
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['d_gmag'].value
        phi = result.params['d_phi'].value
        c_ellip = gmag * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip

    def get_ring_params(gamma, beta):
        return bdtool.get_ring_params(gparam, gamma, beta)

    # Ring test for two values of gamma, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gen_target_image,
                                       get_ring_params,
                                       measure_ellip)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gen_target_image,
                                       get_ring_params,
                                       measure_ellip)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1

    return m, c
