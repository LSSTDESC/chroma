import os
from argparse import ArgumentParser
import logging

import lmfit
import galsim
import numpy as np

import _mypath
import chroma

def fiducial_galaxy():
    '''Setup lmfit.Parameters to represent a Single Sersic galaxy.
    '''
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('hlr', value=0.27)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)
    return gparam

def measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF, pixel_scale, stamp_size,
                        ring_n, galtool):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''

    pix = galsim.Pixel(pixel_scale)

    target_tool = galtool(gal_SED, bandpass, PSF, stamp_size, pixel_scale)
    # generate target image using ringed gparam and PSFs
    def gen_target_image(gamma, beta, diag=None):
        ring_shear = galsim.Shear(g1=gamma.real, g2=gamma.imag)
        target_image = target_tool.get_image(gparam, ring_beta=beta, ring_shear=ring_shear)
        return target_image

    fit_tool = galtool(star_SED, bandpass, PSF, stamp_size, pixel_scale)

    def measure_ellip(target_image, init_param, diag=None):
        def resid(param):
            image = fit_tool.get_image(param)
            return (image.array - target_image.array).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        return gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))

    def get_ring_params(gamma, beta):
        return fit_tool.get_ring_params(gparam, beta, galsim.Shear(g1=gamma.real, g2=gamma.imag))

    # Ring test for two values of gamma, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.ringtest(gamma0, ring_n, gen_target_image, get_ring_params, measure_ellip,
                                 silent=True)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.ringtest(gamma1, ring_n, gen_target_image, get_ring_params, measure_ellip,
                                 silent=True)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1

    return m, c

def ring_vs_z(args):
    """ Measure shear calibration parameters `m` and `c` as a function of redshift for a specific
    combination of star, galaxy, and PSF structural and spectral parameters.  Run
    `python ring_vs_z.py --help` for a list of available command line options.
    """
    dirname = os.path.dirname(args.outfile)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        filename=args.outfile,
                        filemode='w')
    logger = logging.getLogger("ring_vs_z")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # build filter bandpass
    bandpass = chroma.Bandpass(args.datadir+args.filter)
    bandpass = bandpass.createThinned(args.thin)
    PSF_wave = bandpass.effective_wavelength

    # build galaxy SED
    gal_SED0 = chroma.SED(args.datadir+args.galspec, flux_type='flambda')

    # build G5v star SED
    star_SED = chroma.SED(args.datadir+args.starspec)
    star_SED = star_SED.createWithFlux(bandpass, 1.0)

    logger.info('# ')
    logger.info('# General settings')
    logger.info('# ----------------')
    logger.info('# stamp size: {}'.format(args.stamp_size))
    logger.info('# pixel scale: {} arcsec/pixel'.format(args.pixel_scale))
    logger.info('# ring test angles: {}'.format(args.ring_n))

    logger.info('# ')
    logger.info('# Spectra settings')
    logger.info('# ----------------')
    logger.info('# Data directory: {}'.format(args.datadir))
    logger.info('# Filter effective wavelength: {}'.format(PSF_wave))
    logger.info('# Filter: {}'.format(args.filter))
    logger.info('# Thinning filter by factor: {}'.format(args.thin))
    logger.info('# Galaxy SED: {}'.format(args.galspec))
    logger.info('# Star SED: {}'.format(args.starspec))

    # Define the PSF
    if args.moffat:
        monoPSF = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    else:
        monoPSF = galsim.Gaussian(fwhm=args.PSF_FWHM)
    monoPSF.applyShear(g=args.PSF_ellip, beta=args.PSF_phi * galsim.radians)
    if not args.noDCR: #include DCR
        PSF = galsim.ChromaticAtmosphere(monoPSF, base_wavelength=685.0,
                                         zenith_angle=args.zenith_angle * galsim.degrees,
                                         alpha=args.alpha)
    else: # otherwise just include a powerlaw wavelength dependent FWHM
        PSF = galsim.ChromaticObject(monoPSF)
        PSF.applyDilation(lambda w:(w/685)**args.alpha)

    logger.info('# ')
    if args.moffat:
        logger.info('# Moffat PSF settings')
        logger.info('# -------------------')
        logger.info('# PSF beta: {}'.format(args.PSF_beta))
    else:
        logger.info('# Gaussian PSF settings')
        logger.info('# ---------------------')
    logger.info('# PSF phi: {}'.format(args.PSF_phi))
    logger.info('# PSF ellip: {}'.format(args.PSF_ellip))
    logger.info('# PSF FWHM: {} arcsec'.format(args.PSF_FWHM))
    logger.info('# PSF alpha: {}'.format(args.alpha))

    # Go ahead and calculate sqrt(r^2) for PSF here...
    # Ignoring corrections due to ellipticity for now.
    if args.moffat:
        r2_psf = args.PSF_FWHM * np.sqrt(2.0 /
                                         (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
    else:
        r2_psf = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))

    logger.info('# PSF sqrt(r^2): {}'.format(r2_psf))

    if not args.noDCR:
        logger.info('# ')
        logger.info('# Observation settings')
        logger.info('# --------------------')
        logger.info('# zenith angle: {} degrees'.format(args.zenith_angle))

    if args.slow:
        galtool = chroma.SersicTool
    else:
        galtool = chroma.SersicFastTool

    logger.info('# ')
    logger.info('# Galaxy settings')
    logger.info('# ---------------')
    logger.info('# Galaxy Sersic index: {}'.format(args.sersic_n))
    logger.info('# Galaxy ellipticity: {}'.format(args.gal_ellip))
    logger.info('# Galaxy x-offset: {} arcsec'.format(args.gal_x0))
    logger.info('# Galaxy y-offset: {} arcsec'.format(args.gal_y0))
    logger.info('# Galaxy sqrt(r2): {} arcsec'.format(args.gal_r2))

    logger.info('# ')
    logger.info('# Shear Calibration Results')
    logger.info('# -------------------------')
    logger.info(('#  {:>5s}'+' {:>9s}'*8).format('z', 'anltc m1', 'ring m1', 'anltc m2', 'ring m2',
                                                 'anltc c1', 'ring c1', 'anltc c2', 'ring c2'))

    zs = np.arange(0.0, 3.01, 0.03)
    for z in zs:
        gparam = fiducial_galaxy()
        gparam['n'].value = args.sersic_n
        gparam['x0'].value = args.gal_x0 * args.pixel_scale
        gparam['y0'].value = args.gal_y0 * args.pixel_scale
        gparam['gmag'].value = args.gal_ellip

        gal_SED = gal_SED0.createRedshifted(z)
        gal_SED = gal_SED.createWithFlux(bandpass, 1.0)

        gtool = galtool(gal_SED, bandpass, PSF, args.stamp_size, args.pixel_scale)
        gparam = gtool.set_uncvl_r2(gparam, args.gal_r2)

        # Measure shear bias
        m, c = measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF,
                                   args.pixel_scale, args.stamp_size, args.ring_n,
                                   galtool)

        # Now do the analytic part, which can be a little tricky.

        # First calculate \Delta V
        if not args.noDCR:
            dmom_DCR1 = star_SED.getDCRMomentShifts(bandpass, args.zenith_angle * np.pi / 180)
            dmom_DCR2 = gal_SED.getDCRMomentShifts(bandpass, args.zenith_angle * np.pi / 180)
            dV = (dmom_DCR2[1] - dmom_DCR1[1]) * (3600 * 180 / np.pi)**2
        else:
            dV = 0.0
        # Second calculate \Delta r^2 / r^2
        if args.alpha != 0.0:
            seeing1 = star_SED.getSeeingShift(bandpass, alpha=args.alpha)
            seeing2 = gal_SED.getSeeingShift(bandpass, alpha=args.alpha)
            dr2r2 = (seeing2 - seeing1)/seeing1
        else:
            dr2r2 = 0.0

        dIxx = (r2_psf**2/2.0) * dr2r2
        dIxy = 0.0
        dIyy = (r2_psf**2/2.0) * dr2r2
        dIyy += dV

        m1 = m2 = -(dIxx + dIyy) / args.gal_r2**2
        c1 = (dIxx-dIyy) / (2.0 * (args.gal_r2**2))
        c2 = dIxy / args.gal_r2**2

        logger.info(('   {:>5.2f}'+' {:>9.6f}'*8).format(z, m1, m[0], m2, m[1], c1, c[0], c2, c[1]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', default='../../data/',
                        help="directory to find SED and filter files. (Default:../../data/)")
    parser.add_argument('-s', '--starspec', default='SEDs/ukg5v.ascii',
                        help="stellar spectrum to use when fitting (Default: 'SEDs/ukg5v.ascii')")
    parser.add_argument('-g', '--galspec', default='SEDs/CWW_E_ext.ascii',
                        help="galactic spectrum used to create target image " +
                             "(Default: 'SEDs/CWW_E_ext.ascii')")
    parser.add_argument('-f', '--filter', default='filters/LSST_r.dat',
                        help="filter for simulation (Default: 'filters/LSST_r.dat')")
    parser.add_argument('--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default: 45.0)")
    parser.add_argument('--moffat', action='store_true',
                        help="Use Moffat PSF (Default: Gaussian )")
    parser.add_argument('--PSF_beta', type=float, default=2.5,
                        help="Set beta parameter of PSF Moffat profile. (Default: 2.5)")
    parser.add_argument('--PSF_FWHM', type=float, default=0.7,
                        help="Set FWHM of PSF in arcsec (Default: 0.7).")
    parser.add_argument('--PSF_phi', type=float, default=0.0,
                        help="Set position angle of PSF in radians (Default: 0.0).")
    parser.add_argument('--PSF_ellip', type=float, default=0.0,
                        help="Set ellipticity of PSF (Default: 0.0)")
    parser.add_argument('-n', '--sersic_n', type=float, default=0.5,
                        help='Sersic index (Default: 0.5)')
    parser.add_argument('--gal_ellip', type=float, default=0.3,
                        help="Set ellipticity of galaxy (Default: 0.3)")
    parser.add_argument('--gal_x0', type=float, default=0.0,
                        help="Set galaxy center x-offset in pixels (Default: 0.0)")
    parser.add_argument('--gal_y0', type=float, default=0.0,
                        help="Set galaxy center y-offset in pixels (Default: 0.0)")
    parser.add_argument('--gal_r2', type=float, default=0.27,
                        help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default: 0.27)")
    parser.add_argument('--ring_n', type=int, default=3,
                        help="Set number of angles in ring test (Default: 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default: 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default: 31)")
    parser.add_argument('--thin', type=int, default=10,
                        help="Reduce the wavelengths at which Bandpass is evaluted by factor"
                        +" (Default: 10).")
    parser.add_argument('--slow', action='store_true',
                        help="Use GalTool (somewhat more careful) instead of GalFastTool")
    parser.add_argument('--alpha', type=float, default=-0.2,
                        help="Power law index for chromatic seeing (Default: -0.2)")
    parser.add_argument('--noDCR', action='store_true',
                        help="Exclude differential chromatic refraction (DCR) in PSF."
                        +" (Default: include DCR)")
    parser.add_argument('--outfile', default='output/ring_vs_z.dat',
                        help="File to place output (Default: 'output/ring_vs_z.dat'")

    args = parser.parse_args()
    ring_vs_z(args)
