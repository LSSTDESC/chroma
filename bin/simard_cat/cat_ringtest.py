import sys

import numpy
import pyfits
import lmfit
import astropy.utils.console

import linear_spec
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

def get_surviving_photons(gparam, wave, throughput,
                          b_m, b_c, d_m, d_c):
    # lower, mid, and upper wavelengths of input catalog V, I filters
    V_L = 470.0
    V_M = 595.0
    V_U = 750.0

    I_L = 690.0
    I_M = 833.5
    I_U = 1000.0

    # create B/D spectra
    b_flux = b_c + b_m * wave
    b_flux[wave < V_M] = b_c + b_m * V_M
    b_flux[wave > I_M] = b_c + b_m * I_M

    d_flux = d_c + d_m * wave
    d_flux[wave < V_M] = d_c + d_m * V_M
    d_flux[wave > I_M] = d_c + d_m * I_M

    b_photons = b_flux * wave * throughput
    d_photons = d_flux * wave * throughput

    # composite spectrum normalized by B/D fluxes
    c_photons = b_photons * gparam['b_flux'].value + d_photons * gparam['d_flux'].value
    return b_photons, d_photons, c_photons

def measure_shear_calib(gparam,
                        wave, b_photons, d_photons, c_photons,
                        PSF_ellip, PSF_phi,
                        PSF_model, bd_engine):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''

    # setup PSF arguments
    aPSF_kwargs = {'zenith':numpy.pi * 25.0 / 180.0}
    mPSF_kwargs = {'gmag':PSF_ellip, 'phi':PSF_phi, 'beta':2.5, 'FWHM':3.0, 'flux':1.0}
    PSF_kwargs = {'aPSF_kwargs':aPSF_kwargs, 'mPSF_kwargs':mPSF_kwargs}

    # construct PSFs from surviving photons
    bulge_PSF = PSF_model(wave, b_photons, zenith = 30.0 * numpy.pi / 180.0)
    disk_PSF = PSF_model(wave, d_photons, zenith = 30.0 * numpy.pi / 180.0)
    composite_PSF = PSF_model(wave, c_photons, zenith = 30.0 * numpy.pi / 180.0)

    # create galaxy
    gal = chroma.gal_model.BDGal(gparam, bd_engine)

#    gal.set_cvl_r2((0.27/0.2)**2, bulge_PSF, disk_PSF)

    # wrapping galaxy gen_target_image using appropriate PSFs
    def gen_target_image(gamma, beta):
        return gal.gen_target_image(gamma, beta, bulge_PSF, disk_PSF)

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

    # Ring test for two values of gamma, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gen_target_image,
                                       gal.gen_init_param,
                                       measure_ellip, silent=True)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gen_target_image,
                                       gal.gen_init_param,
                                       measure_ellip, silent=True)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1

    return m, c


def ringtest():
    V_L = 470.0
    V_U = 750.0

    I_L = 690.0
    I_U = 1000.0

    simard_dir = '../../data/simard/'
    gal_phot = pyfits.getdata(simard_dir+'table3d.fits')
    bulge_phot = pyfits.getdata(simard_dir+'table3e.fits')
    disk_phot = pyfits.getdata(simard_dir+'table3f.fits')

    PSF_ellip = 0.05
    PSF_phi = 0.0

    filter_dir = '../../data/filters/'
    filter_file = filter_dir + 'LSST_r.dat'
    f_data = numpy.genfromtxt(filter_file)
    wave, throughput = f_data.T

    bd_engine = chroma.imgen.GalSimBDEngine()
    PSF_model = chroma.PSF_model.GSAtmPSF

    with astropy.utils.console.ProgressBar(len(bulge_phot)) as bar:
        for b, d, g in zip(bulge_phot, disk_phot, gal_phot):
            if b['DEEP-GSS'] != d['DEEP-GSS']:
                print 'ERROR ERROR ERROR'
                sys.exit()
            b_m, b_c = linear_spec.linear_spec(b['V606AB'], b['I814AB'])
            d_m, d_c = linear_spec.linear_spec(d['V606AB'], d['I814AB'])
            if (b_m * V_L + b_c < 0) or (b_m * I_U + b_c < 0):
                bar.update(); continue
            if (d_m * V_L + d_c < 0) or (d_m * I_U + d_c < 0):
                bar.update(); continue
            if (not numpy.isfinite(b_m) or not numpy.isfinite(b_c) or
                not numpy.isfinite(d_m) or not numpy.isfinite(d_c)):
                bar.update(); continue
            gparam = fiducial_galaxy()
            #set galaxy B/T from catalog, and ratio of r_e's from catalog
            b_flux = 10**(-0.4 * b['V606AB'])
            d_flux = 10**(-0.4 * d['V606AB'])
            total_flux = b_flux + d_flux
            gparam['b_flux'].value = b_flux/total_flux
            gparam['d_flux'].value = d_flux/total_flux
            gparam['b_r_e'].value = b['re814']/(d['rd814'] * numpy.log(2))
            gparam['d_r_e'].value = 1.0

            b_photons, d_photons, c_photons = get_surviving_photons(gparam, wave, throughput,
                                                                    b_m, b_c, d_m, d_c)
            m, c = measure_shear_calib(gparam,
                                       wave, b_photons, d_photons, c_photons,
                                       PSF_ellip, PSF_phi,
                                       PSF_model, bd_engine)
            bar.update()


if __name__ == '__main__':
    ringtest()
