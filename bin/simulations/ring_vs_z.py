"""Perform a suite of ringtests for a specific combination of star, galaxy, and PSF structural and
spectral parameters while varying the redshift.  Use `python ring_vs_z.py --help` to see a list of
available command line options.

This procedure compares the analytic estimate for chromatic biases to that obtained by simulating
images and fitting their ellipticity either by fitting model parameters with least squares, or
using the Hirata-Seljak-Mandelbaum regaussianization PSF correction algorithm.
"""


import os
from argparse import ArgumentParser
import logging

import lmfit
import galsim
import numpy as np

import _mypath
import chroma

def measure_shear_calib(gparam, gen_target_image, fit_tool, measurer, nring=3):
    # This will serve as the function that returns an initial guess of the sheared and rotated
    # galaxy parameters.
    def get_ring_params(gamma, beta):
        return fit_tool.get_ring_params(gparam, beta, galsim.Shear(g1=gamma.real, g2=gamma.imag))

    # Do ring test for two values of the complex reduced shear `gamma`, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.ringtest(gamma0, nring, gen_target_image, get_ring_params, measurer,
                                 silent=True)
    # c is the same as the estimated reduced shear `gamma_hat` when the input reduced shear
    # is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.ringtest(gamma1, nring, gen_target_image, get_ring_params, measurer,
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
    bandpass = galsim.Bandpass(args.datadir+args.filter)

    # build star SED
    star_SED = galsim.SED(args.datadir+args.starspec)

    # Thin bandpass and spectra if requested
    if args.thin is not None:
        star_SED = star_SED.thin(args.thin)
        bandpass = bandpass.thin(args.thin)

    # Use effective wavelength to set FWHM
    PSF_wave = bandpass.effective_wavelength

    # scale SEDs
    # This probably isn't strictly required, but in general I think it's good to work with numbers
    # near one.
    star_SED = star_SED.withFlux(1.0, bandpass)

    # By default, use args.PSF_FWHM = 0.7.  However, override args.PSF_FWHM if
    # PSF_r2 is explicitly set.
    if args.PSF_r2 is not None:
        if args.moffat:
            args.PSF_FWHM = args.PSF_r2 / np.sqrt(
                2.0 / (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
        elif args.kolmogorov:
            # This line is wrong!!!  What is the relation b/n FWHM and r^2 for a Kolmogorov
            # profile?
            args.PSF_FWHM = args.PSF_r2 / np.sqrt(2.0/np.log(256.0))
        else: # default is Gaussian
            args.PSF_FWHM = args.PSF_r2 / np.sqrt(2.0/np.log(256.0))

    # Define the PSF
    if args.moffat:
        monochromaticPSF = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    elif args.kolmogorov:
        monochromaticPSF = galsim.Kolmogorov(lam_over_r0 = args.PSF_FWHM / 0.976)
    else:
        monochromaticPSF = galsim.Gaussian(fwhm=args.PSF_FWHM)
    monochromaticPSF = monochromaticPSF.shear(
            g=args.PSF_ellip, beta=args.PSF_phi * galsim.degrees)
    if not args.noDCR: #include DCR
        PSF = galsim.ChromaticAtmosphere(monochromaticPSF, base_wavelength=PSF_wave,
                                         zenith_angle=args.zenith_angle * galsim.degrees,
                                         parallactic_angle=args.parallactic_angle * galsim.degrees,
                                         alpha=args.alpha)
    else: # otherwise just include a powerlaw wavelength dependent FWHM
        PSF = galsim.ChromaticObject(monochromaticPSF)
        PSF = PSF.dilate(lambda w:(w/PSF_wave)**args.alpha)

    # Calculate sqrt(r^2) for PSF here...
    # Ignoring corrections due to ellipticity for now.
    if args.moffat:
        r2_PSF = args.PSF_FWHM * np.sqrt(
            2.0 / (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
    elif args.kolmogorov:
        # This line is wrong!!!  What is the relation b/n FWHM and r^2 for a Kolmogorov profile?
        r2_PSF = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))
    else: # default is Gaussian
        r2_PSF = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))

    offset = (args.image_x0, args.image_y0)

    logger.info('# ')
    logger.info('# General settings')
    logger.info('# ----------------')
    logger.info('# stamp size: {}'.format(args.stamp_size))
    logger.info('# pixel scale: {} arcsec/pixel'.format(args.pixel_scale))
    logger.info('# ring test angles: {}'.format(args.nring))

    logger.info('# ')
    logger.info('# Spectra settings')
    logger.info('# ----------------')
    logger.info('# Data directory: {}'.format(args.datadir))
    logger.info('# Filter effective wavelength: {}'.format(PSF_wave))
    logger.info('# Filter: {}'.format(args.filter))
    logger.info('# Thinning filter by factor: {}'.format(args.thin))
    logger.info('# Galaxy SED: {}'.format(args.galspec))
    logger.info('# Star SED: {}'.format(args.starspec))

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

    logger.info('# PSF sqrt(r^2): {}'.format(r2_PSF))

    if not args.noDCR:
        logger.info('# ')
        logger.info('# Observation settings')
        logger.info('# --------------------')
        logger.info('# zenith angle: {} degrees'.format(args.zenith_angle))

    logger.info('# ')
    logger.info('# Galaxy settings')
    logger.info('# ---------------')
    logger.info('# Galaxy Sersic index: {}'.format(args.sersic_n))
    logger.info('# Galaxy ellipticity: {}'.format(args.gal_ellip))
    if args.gal_convFWHM is not None:
        logger.info('# Galaxy PSF-convolved FWHM: {:6.3f} arcsec'.format(
                    args.gal_convFWHM))
    elif args.gal_HLR is not None:
        logger.info('# Galaxy HLR: {} arcsec'.format(args.gal_HLR))
    else:
        logger.info('# Galaxy sqrt(r2): {} arcsec'.format(args.gal_r2))

    logger.info('# ')
    logger.info('# Shear Calibration Results')
    logger.info('# -------------------------')
    logger.info(('#  {:>5s}'+' {:>9s}'*8).format('z', 'anltc m1', 'ring m1', 'anltc m2', 'ring m2',
                                                 'anltc c1', 'ring c1', 'anltc c2', 'ring c2'))

    zs = np.arange(args.zmin, args.zmax+0.001, args.dz)
    for z in zs:
        # build galaxy SED
        gal_SED = galsim.SED(args.datadir+args.galspec, flux_type='flambda')
        gal_SED = gal_SED.atRedshift(z)
        if args.thin is not None:
            gal_SED = gal_SED.thin(args.thin)
        gal_SED = gal_SED.withFlux(1.0, bandpass)

        offset = (args.image_x0, args.image_y0)
        fit_tool = chroma.SersicTool(PSF, args.stamp_size, args.pixel_scale, offset,
                                     star_SED, bandpass)
        target_tool = chroma.SersicTool(PSF, args.stamp_size, args.pixel_scale, offset,
                                        gal_SED, bandpass)
        if not args.slow:
            target_tool.use_effective_PSF()
            fit_tool.use_effective_PSF()

        # build galaxy
        gparam = target_tool.default_galaxy()
        gparam['n'].value = args.sersic_n
        gparam['g'].value = args.gal_ellip
        if args.gal_convFWHM is not None:
            gparam = target_tool.set_FWHM(gparam, args.gal_convFWHM)
        elif args.gal_HLR is not None:
            gparam['hlr'].value = args.gal_HLR
        else:
            gparam = target_tool.set_uncvl_r2(gparam, args.gal_r2)
        args.gal_r2 = target_tool.get_uncvl_r2(gparam)
        gal_fwhm, gal_fwhm_err = target_tool.compute_FWHM(gparam)

        #--------------------------------
        # Analytic estimate of shear bias

        # First calculate \Delta V
        if not args.noDCR:
            dmom_DCR1 = star_SED.calculateDCRMomentShifts(bandpass,
                                                          zenith_angle=(args.zenith_angle
                                                                        * galsim.degrees))
            dmom_DCR2 = gal_SED.calculateDCRMomentShifts(bandpass,
                                                         zenith_angle=(args.zenith_angle
                                                                       * galsim.degrees))
            # radians -> arcseconds
            Vstar = (dmom_DCR1[1] * (3600 * 180 / np.pi)**2)[1, 1]
            Vgal = (dmom_DCR2[1] * (3600 * 180 / np.pi)**2)[1, 1]
        else:
            Vstar = 0.0
            Vgal = 0.0
        dV = Vgal - Vstar
        # Second calculate \Delta r^2 / r^2
        if args.alpha != 0.0:
            seeing1 = star_SED.calculateSeeingMomentRatio(bandpass, alpha=args.alpha)
            seeing2 = gal_SED.calculateSeeingMomentRatio(bandpass, alpha=args.alpha)
            dr2r2 = (seeing2 - seeing1)/seeing1
            logger.debug("star seeing correction: {}".format(seeing1))
            logger.debug("galaxy seeing correction: {}".format(seeing2))
            r2byr2 = seeing2/seeing1
        else:
            dr2r2 = 0.0
            r2byr2 = 1.0

        # chromatic seeing correction
        dI_seeing = np.matrix(np.identity(2), dtype=float) * r2_PSF**2/2.0 * dr2r2
        # DCR correction.
        dI_DCR = np.matrix(np.zeros((2,2), dtype=float))
        dI_DCR[1,1] = dV
        c2p = np.cos(args.parallactic_angle * np.pi/180.0)
        s2p = np.sin(args.parallactic_angle * np.pi/180.0)
        R = np.matrix([[c2p, -s2p],
                       [s2p,  c2p]])
        dI_DCR = R * dI_DCR * R.T
        dI = dI_seeing + dI_DCR

        m1 = m2 = -float(dI.trace()) / args.gal_r2**2
        c1 = (dI[0,0] - dI[1,1]) / (2.0 * args.gal_r2**2)
        c2 = dI[0,1] / args.gal_r2**2

        if args.perturb:
            fit_tool.apply_perturbative_correction(r2byr2, Vstar, Vgal,
                                                   args.parallactic_angle * galsim.degrees)

        # Measure shear bias with ring test
        if args.hsm:
            measurer = chroma.HSMEllipMeasurer(fit_tool)
        else:
            measurer = chroma.LSTSQEllipMeasurer(fit_tool)
        gen_target_image = chroma.TargetImageGenerator(gparam, target_tool)
        def get_ring_params(gamma, beta):
            return fit_tool.get_ring_params(gparam, beta,
                                            galsim.Shear(g1=gamma.real, g2=gamma.imag))
        m, c = chroma.measure_shear_calib(gparam, gen_target_image, get_ring_params, measurer,
                                          nring=args.nring)

        logger.info(('   {:>5.2f}'+' {:>9.6f}'*8).format(z, m1, m[0], m2, m[1], c1, c[0], c2, c[1]))


if __name__ == '__main__':
    parser = ArgumentParser()

    # Input data file arguments
    parser.add_argument('--datadir', default='../../data/',
                        help="directory to find SED and filter files.")
    parser.add_argument('-s', '--starspec', default='SEDs/ukg5v.ascii',
                        help="stellar spectrum to use when fitting (Default 'SEDs/ukg5v.ascii')")
    parser.add_argument('-g', '--galspec', default='SEDs/KIN_Sa_ext.ascii',
                        help="galactic spectrum used to create target image " +
                             "(Default 'SEDs/KIN_Sa_ext.ascii')")
    parser.add_argument('-f', '--filter', default='filters/LSST_r.dat',
                        help="filter for simulation (Default 'filters/LSST_r.dat')")

    # Spectrum treatment arguments
    parser.add_argument('--zmin', type=float, default=0.0,
                        help="minimum galaxy redshift (Default 0.0)")
    parser.add_argument('--zmax', type=float, default=2.0,
                        help="maximum galaxy redshift (Default 2.0)")
    parser.add_argument('--dz', type=float, default=0.05,
                        help="delta galaxy redshift (Default 0.05)")
    parser.add_argument('--thin', type=float, default=1.e-8,
                        help="Thin spectra while retaining bandpass integral accuracy to this "
                        +"relative amount. (Default: 1.e-8).")

    # Observation input arguments
    parser.add_argument('-za', '--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('-q', '--parallactic_angle', default=0.0, type=float,
                        help="parallactic angle in degrees for differential chromatic refraction " +
                             "computation (Default 0.0)")

    # PSF structural arguments
    PSF_profile = parser.add_mutually_exclusive_group()
    PSF_profile.add_argument('--kolmogorov', action='store_true',
                             help="Use Kolmogorov PSF (Default Gaussian)")
    PSF_profile.add_argument('--moffat', action='store_true',
                             help="Use Moffat PSF (Default Gaussian)")
    parser.add_argument('--PSF_beta', type=float, default=2.5,
                        help="Set beta parameter of Moffat profile PSF. (Default 2.5)")
    PSF_size = parser.add_mutually_exclusive_group()
    PSF_size.add_argument('--PSF_FWHM', type=float, default=0.7,
                          help="Set FWHM of PSF in arcsec (Default 0.7).")
    PSF_size.add_argument('--PSF_r2', type=float,
                          help="Override PSF_FWHM with second moment radius sqrt(r^2).")
    parser.add_argument('--PSF_phi', type=float, default=0.0,
                          help="Set position angle of PSF in radians (Default 0.0).")
    parser.add_argument('--PSF_ellip', type=float, default=0.0,
                        help="Set ellipticity of PSF (Default 0.0)")

    # Galaxy structural arguments
    parser.add_argument('-n', '--sersic_n', type=float, default=0.5,
                        help='Sersic index (Default 0.5)')
    parser.add_argument('--gal_ellip', type=float, default=0.3,
                        help="Set ellipticity of galaxy (Default 0.3)")
    gal_size = parser.add_mutually_exclusive_group()
    gal_size.add_argument('--gal_r2', type=float, default=0.27,
                          help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    gal_size.add_argument('--gal_convFWHM', type=float,
                          help="Override gal_r2 by setting galaxy PSF-convolved FWHM.")
    gal_size.add_argument('--gal_HLR', type=float,
                          help="Override gal_r2 by setting galaxy half-light-radius.")

    # Simulation input arguments
    parser.add_argument('--nring', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--image_x0', type=float, default=0.0,
                        help="Image origin x-offset")
    parser.add_argument('--image_y0', type=float, default=0.0,
                        help="Image origin y-offset")
    parser.add_argument('--slow', action='store_true',
                        help="Use ChromaticSersicTool (somewhat more careful) instead of "
                            +"FastChromaticSersicTool.")

    # Physics arguments
    parser.add_argument('--alpha', type=float, default=-0.2,
                        help="Power law index for chromatic seeing (Default: -0.2)")
    parser.add_argument('--noDCR', action='store_true',
                        help="Exclude differential chromatic refraction (DCR) in PSF."
                        +" (Default: include DCR)")

    # Miscellaneous arguments
    parser.add_argument('--hsm', action='store_true',
                        help="Use HSM regaussianization to estimate ellipticity")
    parser.add_argument('--perturb', action='store_true',
                        help="Use PerturbFastChromaticSersicTool to estimate ellipticity")
    parser.add_argument('--quiet', action='store_true',
                        help="Don't print settings")
    parser.add_argument('--outfile', default='output/ring_vs_z.dat',
                        help="output filename.  (Default: 'output/ring_vs_z.dat')")

    # and run the program...
    args = parser.parse_args()
    ring_vs_z(args)
