import os

import numpy
import scipy.integrate
import lmfit
import astropy.utils.console

import _mypath
import chroma

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=0.5, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2)
    gparam.add('phi', value=0.0)
    return gparam

def make_PSF(wave, photons, PSF_ellip, PSF_phi, PSF_model):
    aPSF_kwargs = {'zenith':45.0 * numpy.pi / 180.0}
    gPSF_kwargs = {'gmag':PSF_ellip, 'phi':PSF_phi, 'beta':2.5, 'FWHM':3.0, 'flux':1.0}
    PSF_kwargs = {'aPSF_kwargs':aPSF_kwargs, 'gPSF_kwargs':gPSF_kwargs}
    PSF = PSF_model(wave, photons, **PSF_kwargs)
    return PSF

def measure_shear_calib(gparam, gal_PSF, star_PSF, s_engine):
    gal = chroma.gal_model.SGal(gparam, s_engine)
    def gen_target_image(gamma, beta):
        return gal.gen_target_image(gamma, beta, gal_PSF)

    # function to measure ellipticity of target_image by trying to match the pixels
    # but using the "wrong" PSF (from the stellar SED)
    def measure_ellip(target_image, init_param):
        def resid(param):
            im = s_engine.get_image(param, star_PSF)
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        c_ellip = gmag * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip

    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gen_target_image,
                                       gal.gen_init_param,
                                       measure_ellip, silent=True)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gen_target_image,
                                       gal.gen_init_param,
                                       measure_ellip, silent=True)
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1
    return m, c

def m_vs_rPSF_z():
    s_engine = chroma.ImageEngine.GalSimSEngine(size=41, oversample_factor=41)
    PSF_model = chroma.PSF_model.GSGaussAtmPSF

    PSF_ellip = 0.0
    PSF_phi = 0.0
    filter_file = '../../data/filters/LSST_r.dat'
    gal_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    star_SED_file = '../../data/SEDs/ukg5v.ascii'

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = make_PSF(swave, sphotons, PSF_ellip, PSF_phi, PSF_model)
    smom = chroma.disp_moments(swave, sphotons, zenith=45.0 * numpy.pi / 180)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/m_vs_rPSF_z.dat', 'w')


    r2s = (numpy.linspace(0.2, 0.5, 21) / 0.2)**2 #arcsec -> pixels
    zs = numpy.arange(0.0, 3.0, 0.03)

    with astropy.utils.console.ProgressBar(len(r2s)*len(zs)) as bar:
        for r2 in r2s:
            gparam = fiducial_galaxy()
            gal = chroma.gal_model.SGal(gparam, s_engine)
            gal.set_uncvl_r2(r2 / 0.2)
            for z in numpy.arange(0.0, 3.0, 0.03):
                gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, z)
                gphotons /= scipy.integrate.simps(gphotons, gwave)
                gal_PSF = make_PSF(gwave, gphotons, PSF_ellip, PSF_phi, PSF_model)
                m, c = measure_shear_calib(gparam, gal_PSF, star_PSF, s_engine)

                gmom = chroma.disp_moments(gwave, gphotons, zenith=45.0 * numpy.pi / 180)
                m_analytic = (smom[1] - gmom[1]) * 206265**2 / (0.27**2)
                fil.write('{} {} : {} {} {}\n'.format(z, r2, c, m, m_analytic))
                bar.update()

    fil.close()

if __name__ == '__main__':
    m_vs_rPSF_z()
