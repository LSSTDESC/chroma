import os
from argparse import ArgumentParser
import logging

import lmfit
import galsim
import numpy as np
import astropy.utils.console

import _mypath
import chroma

from pylab import *

def moments(data, pixsize=1.0):
    '''Compute first and second (quadrupole) moments of `data`.  Scales result by `pixsize` for
    non-unit width pixels.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    '''
    xs, ys = np.meshgrid(np.arange(data.shape[0], dtype=np.float64) * pixsize,
                            np.arange(data.shape[0], dtype=np.float64) * pixsize)
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs-xbar)**2).sum() / total
    Iyy = (data * (ys-ybar)**2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return xbar, ybar, Ixx, Iyy, Ixy

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('hlr', value=0.27)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)
    return gparam

def ringtest(gamma, n_ring, gen_target_image, gen_init_param, measure_ellip, silent=False):
    ''' Performs a shear calibration ringtest.

    Produces "true" images uniformly spread along a ring in ellipticity space using the supplied
    `gen_target_image` function.  Then tries to fit these images, (returning ellipticity estimates)
    using the supplied `measure_ellip` function with the fit initialized by the supplied
    `gen_init_param` function.

    The "true" images are sheared by `gamma` (handled by passing through to `gen_target_image`).
    Images are generated in pairs separated by 180 degrees on the ellipticity plane to minimize
    shape noise.

    Ultimately returns an estimate of the applied shear (`gamma_hat`), which can then be compared
    to the input shear `gamma` in an external function to estimate shear calibration parameters.
    '''

    betas = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    ellip0s = []
    ellip180s = []

    def work():
        print 'working'
        #measure ellipticity at beta along the ring
        target_image0 = gen_target_image(gamma, beta)
        init_param0 = gen_init_param(gamma, beta)
        ellip0 = measure_ellip(target_image0, init_param0)
        ellip0s.append(ellip0)

        #repeat with beta on opposite side of the ring (i.e. +180 deg)
        target_image180 = gen_target_image(gamma, beta + np.pi)
        init_param180 = gen_init_param(gamma, beta + np.pi)
        ellip180 = measure_ellip(target_image180, init_param180)
        ellip180s.append(ellip180)

        # print
        # print ellip0
        # print ellip180
        # print 0.5 * (ellip0 + ellip180)

    if not silent:
        with astropy.utils.console.ProgressBar(n_ring) as bar:
            for beta in betas:
                work()
                bar.update()
    else:
        for beta in betas:
            work()

    gamma_hats = [0.5 * (e0 + e1) for e0, e1 in zip(ellip0s, ellip180s)]
    gamma_hat = np.mean(gamma_hats)
    # print
    # print gamma_hat
    return gamma_hat

def measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF, pixel_scale, stamp_size):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''

    pix = galsim.Pixel(pixel_scale)

    target_tool = chroma.new_galtool.SersicTool(gal_SED, bandpass, PSF, stamp_size, pixel_scale)

    # generate target image using ringed gparam and PSFs
    def gen_target_image(gamma, beta):
        ring_shear = galsim.Shear(g1=gamma.real, g2=gamma.imag)
        target_image = target_tool.get_image(gparam, ring_beta=beta, ring_shear=ring_shear)
        # imshow(target_image.array)
        # title('new')
        # show()
        return target_image

    fit_tool = chroma.new_galtool.SersicTool(star_SED, bandpass, PSF, stamp_size, pixel_scale)

    def measure_ellip(target_image, init_param):
        def resid(param):
            image = fit_tool.get_image(param)
            # print
            # print param['x0'].value / pixel_scale
            # print param['y0'].value / pixel_scale
            # print param['n'].value
            # print param['hlr'].value / pixel_scale
            # print param['flux'].value
            # print param['gmag'].value
            # print param['phi'].value
            # print image.array.sum()
            # print abs(image.array - target_image.array).sum()
            return (image.array - target_image.array).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value

        # param = result.params
        # print
        # print param['x0'].value / pixel_scale
        # print param['y0'].value / pixel_scale
        # print param['n'].value
        # print param['hlr'].value / pixel_scale
        # print param['flux'].value
        # print param['gmag'].value
        # print param['phi'].value
        # print gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))
        # print result.chisqr
        # print
        # import ipdb; ipdb.set_trace()

        return gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))

    def get_ring_params(gamma, beta):
        return fit_tool.get_ring_params(gparam, beta, galsim.Shear(g1=gamma.real, g2=gamma.imag))

    # # DEBUG
    # # output first target image
    # image = gen_target_image(0.01 + 0.02j, 1.0)
    # print moments(image.array)
    # import astropy.io.fits as fits
    # fits.writeto('new_target0.fits', image.array, clobber=True)
    # import sys; sys.exit()
    # # DEBUG

    # # DEBUG
    # ring_beta = 1.2
    # ring_shear = galsim.Shear(g1=0.1, g2=0.2)
    # image = fit_tool.get_image(gparam, ring_beta=ring_beta, ring_shear=ring_shear)
    # image2 = fit_tool.get_image2(gparam, ring_beta=ring_beta, ring_shear=ring_shear)
    # m1 = moments(image.array)
    # m2 = moments(image2.array)
    # print m1
    # print m2
    # print [mm1 - mm2 for mm1, mm2 in zip(m1, m2)]
    # import sys; sys.exit()
    # # DEBUG


    # Ring test for two values of gamma, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = ringtest(gamma0, 3, gen_target_image, get_ring_params, measure_ellip, silent=True)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = ringtest(gamma1, 3, gen_target_image, get_ring_params, measure_ellip, silent=True)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1

    return m, c

def new_one_ring_test(args):
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger("one_ring_test")

    # build filter bandpass
    bandpass = chroma.Bandpass(args.datadir+args.filter)
    bandpass = bandpass.truncate(blue_limit=args.bluelim, red_limit=args.redlim)
    bandpass = bandpass.thin(args.thin)

    # build galaxy SED
    gal_SED = chroma.SED(args.datadir+args.galspec, flux_type='flambda')
    gal_SED = gal_SED.setRedshift(args.redshift)

    # build G5v star SED
    star_SED = chroma.SED(args.datadir+args.starspec)

    # scale SEDs
    gal_SED = gal_SED.setFlux(bandpass, 1.0)
    star_SED = star_SED.setFlux(bandpass, 1.0)

    logger.info('')
    logger.info('Spectra settings')
    logger.info('----------------')
    logger.info('Data directory: {}'.format(args.datadir))
    logger.info('Filter: {}'.format(args.filter))
    logger.info('Galaxy SED: {}'.format(args.galspec))
    logger.info('Galaxy redshift: {}'.format(args.redshift))
    logger.info('Star SED: {}'.format(args.starspec))

    # Define the PSF
    if args.gaussian:
        PSF685 = galsim.Gaussian(fwhm=args.PSF_FWHM)
    else:
        PSF685 = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    PSF685.applyShear(g=args.PSF_ellip, beta=args.PSF_phi * galsim.radians)
    PSF = galsim.ChromaticAtmosphere(PSF685, base_wavelength=685.0,
                                     zenith_angle=args.zenith_angle * galsim.degrees,
                                     alpha=0.0)
    logger.info('')
    logger.info('PSF settings')
    logger.info('------------')
    logger.info('PSF beta: {}'.format(args.PSF_beta))
    logger.info('PSF phi: {}'.format(args.PSF_phi))
    logger.info('PSF ellip: {}'.format(args.PSF_ellip))
    logger.info('PSF FWHM: {} arcsec'.format(args.PSF_FWHM))

    gparam = fiducial_galaxy()
    gparam['n'].value = args.sersic_n
    gparam['x0'].value = args.gal_x0 * args.pixel_scale
    gparam['y0'].value = args.gal_y0 * args.pixel_scale
    gparam['gmag'].value = args.gal_ellip
    logger.info('')
    logger.info('Galaxy settings')
    logger.info('---------------')
    logger.info('Galaxy Sersic index: {}'.format(args.sersic_n))
    logger.info('Galaxy ellipticity: {}'.format(args.gal_ellip))
    logger.info('Galaxy x-offset: {} pixels'.format(args.gal_x0))
    logger.info('Galaxy y-offset: {} pixels'.format(args.gal_y0))
    logger.info('Galaxy r2: {} arcsec'.format(args.gal_r2))

    gtool = chroma.new_galtool.SersicTool(gal_SED, bandpass, PSF,
                                          args.stamp_size, args.pixel_scale)
    gparam = gtool.set_uncvl_r2(gparam, (args.gal_r2)**2)

    # Measure shear bias
    m, c = measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF,
                               args.pixel_scale, args.stamp_size)

    analytic1 = star_SED.DCR_moment_shifts(bandpass, args.zenith_angle * np.pi / 180)
    analytic2 = gal_SED.DCR_moment_shifts(bandpass, args.zenith_angle * np.pi / 180)
    m_analytic = (analytic1[1] - analytic2[1]) * (3600 * 180 / np.pi)**2 / (args.gal_r2)**2

    logger.info('')
    logger.info('Results')
    logger.info('-------')
    logger.info('         {:>12s} {:>12s} {:>12s} {:>12s}'.format('m1','m2','c1','c2'))
    logger.info('analytic {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(m_analytic, m_analytic,
                                                                      m_analytic/2, 0.0))
    logger.info('ring     {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(m[0], m[1], c[0], c[1]))

    return m, c

def runme():
    class junk(object):
        pass
    args = junk()
    args.starspec = 'SEDs/ukg5v.ascii'
    args.galspec = 'SEDs/CWW_E_ext.ascii'
    args.redshift = 0.0
    args.filter = 'filters/LSST_r.dat'
    args.zenith_angle = 45.0
    args.datadir = '../../../data/'
    args.PSF_beta = 2.5
    args.PSF_FWHM = 0.7
    args.PSF_phi = 0.0
    args.PSF_ellip = 0.0
    args.sersic_n = 0.5
    args.gal_ellip = 0.3
    args.gal_x0 = 0.0
    args.gal_y0 = 0.0
    args.gal_r2 = 0.27
    args.gal_nring = 3
    args.pixel_scale = 0.2
    args.stamp_size = 31
    args.gaussian = True
    args.N = 100
    args.bluelim = 500
    args.redlim = 750
    args.thin = 150
    new_one_ring_test(args)

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
    parser.add_argument('--gal_x0', type=float, default=0.0,
                        help="Set galaxy center x-offset in pixels (Default 0.0)")
    parser.add_argument('--gal_y0', type=float, default=0.0,
                        help="Set galaxy center y-offset in pixels (Default 0.0)")
    parser.add_argument('--gal_r2', type=float, default=0.27,
                        help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    parser.add_argument('--nring', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--gaussian', action='store_true',
                        help="Use Gaussian PSF (Default Moffat)")
    parser.add_argument('--N', type=int, default=100,
                        help="Number of wavelength samples in intregrand (Default 100)")
    parser.add_argument('--bluelim', type=float, default=300)
    parser.add_argument('--redlim', type=float, default=1200)
    parser.add_argument('--thin', type=int, default=50)


    args = parser.parse_args()

    new_one_ring_test(args)
