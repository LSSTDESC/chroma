from argparse import ArgumentParser
import logging

import numpy as np
from scipy.integrate import simps
import lmfit

import galsim

import _mypath
import chroma

from pylab import *
import copy

def my_imshow(my_img,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%5e @ [%4i, %4i]' % (my_img[y, x], x, y)
        except IndexError:
            return ''
    img = ax.imshow(my_img,**kwargs)
    ax.format_coord = format_coord
    return img

def moffat_to_r2(fwhm, q, phi, beta):
    cph = np.cos(phi)
    sph = np.sin(phi)
    R = np.matrix([[cph,-sph],[sph,cph]])
    II = fwhm**2 / (8.0 * (2.0**(1.0/beta)-1.0) * (beta - 2.0)) * np.matrix([[1./q, 0.0], [0.0, q]])
    I = R*II*R.T
    return I[0,0] + I[1,1]

def moffat_to_I(fwhm, q, phi, beta):
    cph = np.cos(phi)
    sph = np.sin(phi)
    R = np.matrix([[cph,-sph],[sph,cph]])
    II = fwhm**2 / (8.0 * (2.0**(1.0/beta)-1.0) * (beta - 2.0)) * np.matrix([[1./q, 0.0], [0.0, q]])
    return R*II*R.T

def I_to_moffat(II, beta):
    phi = 0.5 * np.arctan2(2.0*II[0,1], II[0,0]-II[1,1])
    abschi = np.sqrt((II[0,0] - II[1,1])**2 + 4.0*II[0,1]**2)/(II[0,0]+II[1,1])
    q = np.sqrt((1.0 - abschi)/(1.0 + abschi))
    cph = np.cos(phi)
    sph = np.sin(phi)
    R = np.matrix([[cph,sph],[-sph,cph]])
    IIrot = R*II*R.T
    fwhm = np.sqrt(8.0*(2.0**(1.0/beta)-1.0)*(beta-2.0)*np.sqrt(IIrot[0,0]*IIrot[1,1]))
    return fwhm, q, phi

def modify_moffat(moffat, dR, dV):
    moffat['Ixx'].value = moffat['Ixx'].value + dV
    moffat['x0'].value = moffat['x0'].value + dR
    return moffat

def moffat_to_galsim(moffat):
    II = np.matrix([[moffat['Ixx'].value, moffat['Ixy'].value],
                    [moffat['Ixy'].value, moffat['Iyy'].value]])
    beta = moffat['beta'].value
    fwhm, q, phi = I_to_moffat(II, beta)
    obj = galsim.Moffat(fwhm=fwhm, beta=moffat['beta'].value, flux=moffat['amplitude'].value)
    obj.applyShear(g1=(1.0-q)/(1.0+q))
    obj.applyRotation(phi * galsim.radians)
    obj.applyShift(moffat['y0'].value, moffat['x0'].value)
    return obj

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
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)
    return gparam

def measure_shear_calib(gparam, gal_PSF, star_PSF, s_engine):
    galtool = chroma.GalTools.SGalTool(s_engine)

    def gen_target_image(gamma, beta):
        ring_gparam = galtool.get_ring_params(gparam, gamma, beta)
        target_image = s_engine.get_image(ring_gparam, gal_PSF)
        # imshow(target_image)
        # title('old')
        # show()
        return target_image

    # function to measure ellipticity of target_image by trying to match the pixels
    # but using the "wrong" PSF (from the stellar SED)
    def measure_ellip(target_image, init_param):
        def resid(param):
            im = s_engine.get_image(param, star_PSF)
            # print
            # print param['x0'].value
            # print param['y0'].value
            # print param['n'].value
            # print param['r_e'].value
            # print param['flux'].value
            # print param['gmag'].value
            # print param['phi'].value
            # print im.sum()
            # print abs(im - target_image).sum()
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        c_ellip = gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))

        # param = result.params
        # print
        # print param['x0'].value
        # print param['y0'].value
        # print param['n'].value
        # print param['r_e'].value
        # print param['flux'].value
        # print param['gmag'].value
        # print param['phi'].value
        # print c_ellip
        # print result.chisqr

        return c_ellip

    def get_ring_params(gamma, beta):
        return galtool.get_ring_params(gparam, gamma, beta)

    # # DEBUG
    # # output first target image
    # image = gen_target_image(0.01 + 0.02j, 1.0)
    # print moments(image)
    # import astropy.io.fits as fits
    # fits.writeto('target0.fits', image, clobber=True)
    # import sys; sys.exit()
    # # DEBUG

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
    PSF_model = chroma.PSF_model.GSAtmPSF
    filter_file = args.datadir+args.filter
    gal_SED_file = args.datadir+args.galspec
    star_SED_file = args.datadir+args.starspec
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    w = (swave >= args.bluelim) & (swave <= args.redlim)
    swave=swave[w]
    sphotons=sphotons[w]
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
    w = (gwave >= args.bluelim) & (gwave <= args.redlim)
    gwave = gwave[w]
    gphotons = gphotons[w]
    gphotons /= simps(gphotons, gwave)
    gal_PSF = PSF_model(gwave, gphotons, zenith=args.zenith_angle * np.pi/180.,
                        moffat_beta=args.PSF_beta, moffat_FWHM=args.PSF_FWHM/args.pixel_scale,
                        moffat_phi=args.PSF_phi, moffat_ellip=args.PSF_ellip,
                        pixel_scale=args.pixel_scale)
    gparam1 = copy.deepcopy(gparam)

    # This is where I want to fit a Moffat profile to the star PSF.
    star_PSF_image = s_engine.get_PSF_image(star_PSF)
    gal_PSF_image = s_engine.get_PSF_image(gal_PSF)
    star_mom = moments(star_PSF_image)
    moffat = lmfit.Parameters()
    moffat.add('x0', value=0.0)
    moffat.add('y0', value=0.0)
    moffat.add('Ixx', value=star_mom[2])
    moffat.add('Iyy', value=star_mom[3])
    moffat.add('Ixy', value=star_mom[4])
    moffat.add('beta', value=args.PSF_beta, min=1.1)
    moffat.add('amplitude', value=1.0)

    def moffat_to_image(moffat):
        obj = moffat_to_galsim(moffat)
        im = galsim.ImageD(args.stamp_size, args.stamp_size, scale=1.0)
        obj.draw(image=im)
        return im
    def starresid(m):
        mim = moffat_to_image(m)
        return (mim.array - star_PSF_image).flatten()
    starresult = lmfit.minimize(starresid, moffat)
    moffat2 = copy.deepcopy(moffat)
    def galresid(m):
        mim = moffat_to_image(m)
        return (mim.array - gal_PSF_image).flatten()
    galresult = lmfit.minimize(galresid, moffat2)

    gmom = chroma.disp_moments(gwave, gphotons, zenith=args.zenith_angle * np.pi/180.)
    m_analytic = - (gmom[1] - smom[1]) * (3600 * 180 / np.pi)**2 / (args.gal_r2**2)
    dV = (gmom[1] - smom[1]) * (3600 * 180 / np.pi)**2 / args.pixel_scale**2
    dR = (gmom[0] - smom[0]) * (3600 * 180 / np.pi) / args.pixel_scale

    starfit_PSF = moffat_to_galsim(starresult.params)
    starfit_PSF_image = s_engine.get_PSF_image(starfit_PSF)
    galfit_PSF = moffat_to_galsim(galresult.params)
    galfit_PSF_image = s_engine.get_PSF_image(galfit_PSF)
    moffat = modify_moffat(moffat, dR, -dV)
    shiftedstar_PSF = star_PSF.createShifted((0, dR))
    shiftedstar_PSF_image = s_engine.get_PSF_image(shiftedstar_PSF)
    galpredict_PSF = moffat_to_galsim(moffat)
    galpredict_PSF_image = s_engine.get_PSF_image(galpredict_PSF)


    print 'shiftedstar  ', moments(shiftedstar_PSF_image)
    print 'star         ', moments(star_PSF_image)
    print 'gal          ', moments(gal_PSF_image)
    print 'starfit      ', moments(starfit_PSF_image)
    print 'galfit       ', moments(galfit_PSF_image)
    print 'galpredict   ', moments(galpredict_PSF_image)

    figure(figsize=(10,4))
    subplot(131)
    my_imshow(np.log10(abs(gal_PSF_image-shiftedstar_PSF_image)), vmin=-10, vmax=-5)
    title('true galaxy - true star')
    subplot(132)
    my_imshow(np.log10(abs(gal_PSF_image-galpredict_PSF_image)), vmin=-10, vmax=-5)
    title('true galaxy - predicted galaxy')
    subplot(133)
    my_imshow(np.log10(abs(gal_PSF_image-galfit_PSF_image)), vmin=-10, vmax=-5)
    title('true galaxy - fitted galaxy')
    show()

    m, c = measure_shear_calib(gparam1, gal_PSF, galfit_PSF, s_engine)

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
    parser.add_argument('--bluelim', type=float, default=0.0)
    parser.add_argument('--redlim', type=float, default=1200.0)

    args = parser.parse_args()

    one_ring_test(args)
