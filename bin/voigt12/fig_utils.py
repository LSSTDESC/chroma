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

def measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                        PSF_ellip, PSF_phi,
                        PSF_model, bd_engine):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''
    wave, photons = chroma.utils.get_photons([bulge_SED_file, disk_SED_file],
                                             filter_file, redshift)
    bulge_photons, disk_photons = photons
    use=None
    PSF_kwargs = {'ellipticity':PSF_ellip, 'phi':PSF_phi}

    gal = chroma.BDGal(gparam, wave, bulge_photons, disk_photons,
                       PSF_model=PSF_model, PSF_kwargs=PSF_kwargs,
                       bd_engine=bd_engine)

    gal.build_circ_PSF()
    gal.set_FWHM_ratio(1.4)
    # Do ring test with two values of gamma_true: (0.0, 0.0) and (0.01, 0.02). With these two
    # simulations, one can solve the shear calibration equation for `m` and `c`.

    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gal.gen_target_image,
                                       gal.gen_init_param,
                                       gal.measure_ellip)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gal.gen_target_image,
                                       gal.gen_init_param,
                                       gal.measure_ellip)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1
    return m, c
