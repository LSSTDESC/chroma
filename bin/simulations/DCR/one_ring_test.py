import copy
from argparse import ArgumentParser
import logging

import numpy as np
from scipy.integrate import simps
import lmfit

import _mypath
import chroma

data_dir = '../../../data/'

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
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
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        c_ellip = gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))
        return c_ellip

    def get_ring_params(gamma, beta):
        return galtool.get_ring_params(gparam, gamma, beta)

    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, args.nring,
                                       gen_target_image,
                                       get_ring_params,
                                       measure_ellip, silent=True)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, args.nring,
                                       gen_target_image,
                                       get_ring_params,
                                       measure_ellip, silent=True)
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1
    return m, c

def one_ring_test(args):
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger("one_ring_test")
    logger.info('Image settings')
    logger.info('--------------')

    s_engine = chroma.ImageEngine.GalSimSEngine(size=args.stamp_size)
    logger.info('Stamp size: {} pixels'.format(args.stamp_size))
    logger.info('Pixel scale: {} arcsec'.format(args.pixel_scale))

    galtool = chroma.GalTools.SGalTool(s_engine)
    PSF_model = chroma.PSF_model.GSAtmPSF #implements DCR, no chromatic seeing
    filter_file = data_dir+args.filter
    gal_SED_file = data_dir+args.galspec
    star_SED_file = data_dir+args.starspec
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    sphotons /= simps(sphotons, swave)
    logger.info('')
    logger.info('Spectra settings')
    logger.info('----------------')
    logger.info('Data directory: {}'.format(args.datadir))
    logger.info('Filter: {}'.format(args.filter))
    logger.info('Galaxy SED: {}'.format(args.galspec))
    logger.info('Galaxy redshift: {}'.format(args.redshift))
    logger.info('Star SED: {}'.format(args.starspec))

    star_PSF = PSF_model(swave, sphotons, zenith=args.zenith_angle * np.pi/180.,
                         moffat_beta=args.PSF_beta, moffat_FWHM=args.PSF_FWHM/args.pixel_scale,
                         moffat_phi=args.PSF_phi, moffat_ellip=args.PSF_ellip,
                         pixel_scale=args.pixel_scale)
    logger.info('')
    logger.info('PSF settings')
    logger.info('------------')
    logger.info('PSF beta: {}'.format(args.PSF_beta))
    logger.info('PSF phi: {}'.format(args.PSF_phi))
    logger.info('PSF ellip: {}'.format(args.PSF_ellip))
    logger.info('PSF FWHM: {} arcsec'.format(args.PSF_FWHM))

    smom = chroma.disp_moments(swave, sphotons, zenith=args.zenith_angle * np.pi/180.)

    gparam = fiducial_galaxy()
    gparam['n'].value = args.sersic_n
    gparam['x0'].value = args.gal_x0
    gparam['y0'].value = args.gal_y0
    gparam['gmag'].value = args.gal_ellip
    logger.info('')
    logger.info('Galaxy settings')
    logger.info('---------------')
    logger.info('Galaxy Sersic index: {}'.format(args.sersic_n))
    logger.info('Galaxy ellipticity: {}'.format(args.gal_ellip))
    logger.info('Galaxy x-offset: {} pixels'.format(args.gal_x0))
    logger.info('Galaxy y-offset: {} pixels'.format(args.gal_y0))
    logger.info('Galaxy r2: {} arcsec'.format(args.gal_r2))

    # normalize size to second moment (before PSF convolution)
    gparam = galtool.set_uncvl_r2(gparam, (args.gal_r2/args.pixel_scale)**2)

    gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, args.redshift)
    gphotons /= simps(gphotons, gwave)
    gal_PSF = PSF_model(gwave, gphotons, zenith=args.zenith_angle * np.pi/180.,
                        moffat_beta=args.PSF_beta, moffat_FWHM=args.PSF_FWHM/args.pixel_scale,
                        moffat_phi=args.PSF_phi, moffat_ellip=args.PSF_ellip,
                        pixel_scale=args.pixel_scale)
    gparam1 = copy.deepcopy(gparam)
    m, c = measure_shear_calib(gparam1, gal_PSF, star_PSF, s_engine)
    gmom = chroma.disp_moments(gwave, gphotons, zenith=args.zenith_angle * np.pi/180.)
    m_analytic = - (gmom[1] - smom[1]) * (3600 * 180 / np.pi)**2 / (args.gal_r2**2)

    logger.info('')
    logger.info('Results')
    logger.info('-------')
    logger.info('         {:>12s} {:>12s} {:>12s} {:>12s}'.format('m1','m2','c1','c2'))
    logger.info('analytic {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(m_analytic, m_analytic,
                                                                      m_analytic/2, 0.0))
    logger.info('ring     {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(m[0], m[1], c[0], c[1]))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--starspec', default='SEDs/ukg5v.ascii',
                        help="stellar spectrum to use when fitting (Default 'SEDs/ukg5v.ascii')")
    parser.add_argument('-g', '--galspec', default='SEDs/CWW_E_ext.ascii',
                        help="galactic spectrum used to create target image " +
                             "(Default 'SEDs/CWW_E_ext.ascii')")
    parser.add_argument('-z', '--redshift', type=float, default=0.0,
                        help="galaxy redshift (Default 0.0)")
    parser.add_argument('-f', '--filter', default='filters/LSST_r.dat',
                        help="filter for simulation (Default 'filters/LSST_r.dat')")
    parser.add_argument('--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('--datadir', default='../../../data/',
                        help="directory to find SED and filter files.")
    parser.add_argument('--PSF_beta', type=float, default=2.5,
                        help="Set beta parameter of PSF Moffat profile. (Default 2.5)")
    parser.add_argument('--PSF_FWHM', type=float, default=0.7,
                        help="Set FWHM of PSF in arcsec (Default 0.7).")
    parser.add_argument('--PSF_phi', type=float, default=0.0,
                        help="Set position angle of PSF in radians (Default 0.0).")
    parser.add_argument('--PSF_ellip', type=float, default=0.0,
                        help="Set ellipticity of PSF (Default 0.0)")
    parser.add_argument('-n', '--sersic_n', type=float, default=0.5,
                        help='Sersic index (Default 0.5)')
    parser.add_argument('--gal_ellip', type=float, default=0.3,
                        help="Set ellipticity of galaxy (Default 0.3)")
    parser.add_argument('--gal_x0', type=float, default=0.1,
                        help="Set galaxy center x-offset in pixels (Default 0.1)")
    parser.add_argument('--gal_y0', type=float, default=0.3,
                        help="Set galaxy center y-offset in pixels (Default 0.3)")
    parser.add_argument('--gal_r2', type=float, default=0.27,
                        help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    parser.add_argument('--nring', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--use_poly_r2', action='store_true',
                        help="Use polynomial in n to compute ratio r^2 / hlr when setting r^2." +
                             "Default is to iteratively draw, measure, and rescale r^2 in images" +
                             "until converged to input value.")

    # additional args:
    # stamp_size
    # galaxy redshift
    # n_ring
    # analytic vs simulation r^2 setting
    # r2 setting!
    args = parser.parse_args()

    one_ring_test(args)
