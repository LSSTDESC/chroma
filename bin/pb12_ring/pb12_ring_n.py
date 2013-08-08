import os
import sys
import copy

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
    gparam.add('n', value=4.0, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2)
    gparam.add('phi', value=0.0)
    return gparam

def measure_shear_calib(gparam, gal_PSF, star_PSF, s_engine):
    galtool = chroma.GalTools.SGalTool(s_engine)

    def gen_target_image(gamma, beta):
        ring_gparam = galtool.get_ring_params(gparam, gamma, beta)
        return s_engine.get_image(ring_gparam, gal_PSF)

    # function to measure ellipticity of target_image by trying to match the pixels
    # but using the "wrong" PSF (from the stellar SED)
    def measure_ellip(target_image, init_param):
        def resid(param):
            im = s_engine.get_image(param, star_PSF)
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        # print
        # print
        # print
        # print
        # print
        # print
        # lmfit.report_errors(result.params)
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        c_ellip = gmag * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip

    def get_ring_params(gamma, beta):
        return galtool.get_ring_params(gparam, gamma, beta)

    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gen_target_image,
                                       get_ring_params,
                                       measure_ellip, silent=True)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gen_target_image,
                                       get_ring_params,
                                       measure_ellip, silent=True)
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1
    return m, c

def calib_vs_redshift(filter_name, gal, star, n, zenith=30*numpy.pi/180):
    s_engine = chroma.ImageEngine.GalSimSEngine(size=31)
    PSF_model = chroma.PSF_model.GSAtmPSF
    PSF_ellip = 0.0
    PSF_phi = 0.0
    data_dir = '../../data/'
    filter_file = data_dir+'filters/LSST_{}.dat'.format(filter_name)
    gal_SED_file = data_dir+'SEDs/{}.ascii'.format(gal)
    star_SED_file = data_dir+'SEDs/{}.ascii'.format(star)

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    # swave = swave[::50]
    # sphotons = sphotons[::50]
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = PSF_model(swave, sphotons, zenith=zenith)
    smom = chroma.disp_moments(swave, sphotons, zenith=zenith)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    outfile = 'output/calib_vs_redshift.{}.{}.{}.{}.z{:02d}.dat'
    outfile = outfile.format(filter_name, gal, star, n, int(round(zenith*180/numpy.pi)))
    fil = open(outfile, 'w')
    gparam = fiducial_galaxy()
    gparam['n'].value = n
    #gparam['gmag'].value = 0.0
    galtool = chroma.GalTools.SGalTool(s_engine)

    # normalize size to second moment (before PSF convolution)
    print
    print
    print 'n: {}'.format(gparam['n'].value)
    print 'fiducial r_e: {}'.format(gparam['r_e'].value)
    print 'setting second moment radius to 0.27 arcseconds = 1.35 pixels'
    gparam = galtool.set_uncvl_r2(gparam, (0.27/0.2)**2) # (0.27 arcsec)^2 -> pixels^2
    print 'output r2: {}'.format(galtool.get_uncvl_r2(gparam))
    print 'output r: {}'.format(numpy.sqrt(galtool.get_uncvl_r2(gparam)))
    print 'output r_e:{}'.format(gparam['r_e'].value)

    # print 'retreiving subpixelized image of unconvolved circularized galaxy.'
    # gparam1 = copy.deepcopy(gparam)
    # gparam1['gmag'].value = 0.0
    # gim = s_engine.get_uncvl_image(gparam1, pixsize=1./7)
    # print 'computing quadrupole moment from image.'
    # mom = chroma.utils.moments(gim, pixsize=1./7)
    # print 'empirical second moment radius = {} pixels'.format(numpy.sqrt(mom[2]+mom[3]))
    # print 'r2/re = {}'.format(numpy.sqrt(mom[2]+mom[3])/gparam1['r_e'].value)

    zs = numpy.arange(0.0, 3.0, 0.03)
    with astropy.utils.console.ProgressBar(len(zs)) as bar:
        for z in zs:
            gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, z)
            # gwave = gwave[::50]
            # gphotons = gphotons[::50]
            gphotons /= scipy.integrate.simps(gphotons, gwave)
            gal_PSF = PSF_model(gwave, gphotons, zenith=zenith)
            gparam1 = copy.deepcopy(gparam)
            m, c = measure_shear_calib(gparam1, gal_PSF, star_PSF, s_engine)

            gmom = chroma.disp_moments(gwave, gphotons, zenith=zenith)
            m_analytic = - (gmom[1] - smom[1]) * (3600 * 180 / numpy.pi)**2 / (0.27**2)
            fil.write('{} {} {} {}\n'.format(z, c, m, m_analytic))
            bar.update()
        fil.close()

def main(argv):
    calib_vs_redshift('r', 'CWW_E_ext', 'ukg5v', 4.0, zenith=30*numpy.pi/180)
    calib_vs_redshift('r', 'CWW_E_ext', 'ukg5v', 1.0, zenith=30*numpy.pi/180)
    calib_vs_redshift('r', 'CWW_E_ext', 'ukg5v', 0.5, zenith=30*numpy.pi/180)

if __name__ == '__main__':
    main(sys.argv)
