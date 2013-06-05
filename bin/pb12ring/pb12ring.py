import os
import sys
import copy

import numpy
import scipy.integrate
import lmfit

import _mypath
import chroma

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('r_e', value=0.058/0.2)
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

def m_vs_redshift(s_engine, PSF_model):
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
    fil = open('output/m_vs_redshift.dat', 'w')
    gparam = fiducial_galaxy()
    # normalize size to second moment (before PSF convolution)
    gal = chroma.gal_model.SGal(gparam, s_engine)
    gal.set_uncvl_r2((0.27/0.2)**2) ## (0.27 arcsec)^2 -> pixels^2
    gparam = gal.gparam0
    print gparam['n'].value
    print gparam['r_e'].value * 0.2 #pixels -> arcsec
    print
    print

    for z in numpy.arange(0.0, 2.01, 0.02):
        print z
        gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, z)
        gphotons /= scipy.integrate.simps(gphotons, gwave)
        gal_PSF = make_PSF(gwave, gphotons, PSF_ellip, PSF_phi, PSF_model)
        m, c = measure_shear_calib(gparam, gal_PSF, star_PSF, s_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])

        gmom = chroma.disp_moments(gwave, gphotons, zenith=45.0 * numpy.pi / 180)
        m_analytic = (smom[1] - gmom[1]) * 206265**2 / (0.27**2)
        print 'm_analytic: {:10g}'.format(m_analytic)
        fil.write('{} {} {} {}\n'.format(z, c, m, m_analytic))
    fil.close()

def main(argv):
    s_engine = chroma.imgen.GalSimSEngine(size=41, oversample_factor=41)
    PSF_model = chroma.PSF_model.GSGaussAtmPSF
    m_vs_redshift(s_engine, PSF_model)

if __name__ == '__main__':
    main(sys.argv)
