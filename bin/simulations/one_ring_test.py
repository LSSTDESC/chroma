"""Perform a ring test for a specific combination of star, galaxy, and PSF structural and spectral
parameters.  Use `python one_ring_test.py --help` to see a list of available command line
options.

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
try:
    import astropy.io.fits as fits
except:
    import pyfits as fits

import _mypath
import chroma

def one_ring_test(args):
    """ Run a single ring test.  There are many configurable options here.  From the command-line,
    run `python one_ring_test.py --help` to see them.
    """
    # setup logging
    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger("one_ring_test")
    if args.quiet:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    # load filter bandpass
    bandpass = chroma.Bandpass(args.datadir+args.filter)

    # load and redshift galaxy SED
    gal_SED = chroma.SED(args.datadir+args.galspec, flux_type='flambda')
    gal_SED = gal_SED.atRedshift(args.redshift)

    # load stellar SED
    star_SED = chroma.SED(args.datadir+args.starspec)

    # Thin bandpass and spectra if requested
    if args.thin is not None:
        gal_SED = gal_SED.thin(args.thin)
        star_SED = star_SED.thin(args.thin)
        bandpass = bandpass.thin(args.thin)

    # Use effective wavelength to set FWHM
    PSF_wave = bandpass.effective_wavelength

    # scale SEDs. This probably isn't strictly required, but in general I think it's good to work
    # with numbers near one.
    gal_SED = gal_SED.withFlux(1.0, bandpass)
    star_SED = star_SED.withFlux(1.0, bandpass)

    # By default, set the PSF size from the args.PSF_FWHM argument.
    # Override this if args.PSF_r2 is explicitly set
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
    if not args.noDCR: # add DCR if not explicitly suppressed
        PSF = galsim.ChromaticAtmosphere(monochromaticPSF, base_wavelength=PSF_wave,
                                         zenith_angle=args.zenith_angle * galsim.degrees,
                                         parallactic_angle=args.parallactic_angle * galsim.degrees,
                                         alpha=args.alpha)
    else: # otherwise just include a powerlaw wavelength dependent FWHM
        PSF = galsim.ChromaticObject(monochromaticPSF)
        PSF = PSF.dilate(lambda w:(w/PSF_wave)**args.alpha)

    # Calculate sqrt(r^2) for the PSF.
    # Ignoring corrections due to non-zero PSF ellipticity.
    if args.moffat:
        r2_PSF = args.PSF_FWHM * np.sqrt(
            2.0 / (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
    elif args.kolmogorov:
        # This line is wrong!!!  What is the relation b/n FWHM and r^2 for a Kolmogorov profile?
        r2_PSF = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))
    else: # default is Gaussian
        r2_PSF = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))

    # Now create some galtools, which are objects that know how to draw images given some
    # parameters, thus bridging GalSim and lmfit.
    offset = (args.image_x0, args.image_y0)
    # The target_tool will be used to generate "true" images of the simulated galaxy, by which I
    # mean that the PSF is actually derived from the galactic PSF.  Note that we also have to
    # supply some bookkeeping arguments defining the image size, scale, and offset.  The galaxy
    # SED and filter bandpass arguments come last, since they are optional.  If not specified, then
    # the assumption is that PSF is a GalSim.GSObject, and not a GalSim.ChromaticObject, and thus
    # that the SED and bandpass are not needed to draw the PSF-convolved image.
    target_tool = chroma.SersicTool(PSF, args.stamp_size, args.pixel_scale, offset,
                                    gal_SED, bandpass)
    # The fit_tool is basically the same as the target_tool, except that we assert a different --
    # incorrect --  SED to investigate the effect of deriving the PSF model from a nearby star with
    # a different SED than the galaxy.
    fit_tool = chroma.SersicTool(PSF, args.stamp_size, args.pixel_scale, offset,
                                 star_SED, bandpass)

    # By default, we compress the wavelength-dependent PSFs into effective PSFs by integrating
    # them against the bandpass throughput over wavelength.  This is much faster than letting
    # GalSim do the wavelength integration every time you draw a target or candidate fit image.
    if not args.slow:
        target_tool.use_effective_PSF()
        fit_tool.use_effective_PSF()

    # Initialize galaxy
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

    # Estimate the shear bias due to effective PSF differences analytically.  The details to these
    # maths are in the paper (Meyers & Burchat 2014).

    # First calculate \Delta V
    if not args.noDCR:
        dmom_DCR1 = star_SED.getDCRMomentShifts(bandpass, args.zenith_angle * np.pi / 180)
        dmom_DCR2 = gal_SED.getDCRMomentShifts(bandpass, args.zenith_angle * np.pi / 180)
        Vstar = dmom_DCR1[1] * (3600 * 180 / np.pi)**2
        Vgal = dmom_DCR2[1] * (3600 * 180 / np.pi)**2
    else:
        Vstar = 0.0
        Vgal = 0.0
    dV = Vgal - Vstar
    # Second calculate \Delta r^2 / r^2
    if args.alpha != 0.0:
        seeing1 = star_SED.getSeeingShift(bandpass, alpha=args.alpha)
        seeing2 = gal_SED.getSeeingShift(bandpass, alpha=args.alpha)
        dr2r2 = (seeing2 - seeing1)/seeing1
        r2r2 = seeing2/seeing1
    else:
        dr2r2 = 0.0
        r2r2 = 1.0

    # The maths here are explained in detail in the paper: (Meyers & Burchat 2014).
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

    # If requested, apply the perturbative type PSF chromatic correction described in
    # Meyers & Burchat (2014) to the fitting tool.  If this works, then the m&c values
    # that come out of this script should be small, or at least much smaller than when
    # no perturbative chromatic correction is applied.
    if args.perturb:
        fit_tool.apply_perturbative_correction(r2r2, Vstar, Vgal,
                                               args.parallactic_angle * galsim.degrees)

    # Print out configuration details for this run
    logger.debug('')
    logger.debug('General settings')
    logger.debug('----------------')
    logger.debug('stamp size: {}'.format(args.stamp_size))
    logger.debug('pixel scale: {} arcsec/pixel'.format(args.pixel_scale))
    logger.debug('ring test angles: {}'.format(args.nring))

    logger.debug('')
    logger.debug('Spectra settings')
    logger.debug('----------------')
    logger.debug('Data directory: {}'.format(args.datadir))
    logger.debug('Filter: {}'.format(args.filter))
    logger.debug('Filter effective wavelength: {}'.format(PSF_wave))
    logger.debug('Thinning with relative error: {}'.format(args.thin))
    logger.debug('Galaxy SED: {}'.format(args.galspec))
    logger.debug('Galaxy redshift: {}'.format(args.redshift))
    logger.debug('Star SED: {}'.format(args.starspec))
    logger.debug('')
    logger.debug('Spectra chromatic biases')
    logger.debug('------------------------')
    logger.debug('Vstar: {:8.6f} arcsec^2'.format(Vstar))
    logger.debug('Vgal: {:8.6f} arcsec^2'.format(Vgal))
    logger.debug('r^2_{{PSF,gal}}/r^2_{{PSF,*}}: {:8.6f}'.format(r2r2))
    logger.debug('')
    if args.moffat:
        logger.debug('Moffat PSF settings')
        logger.debug('-------------------')
        logger.debug('PSF beta: {}'.format(args.PSF_beta))
    elif args.kolmogorov:
        logger.debug('Kolmogorov PSF settings')
        logger.debug('-----------------------')
    else:
        logger.debug('Gaussian PSF settings')
        logger.debug('---------------------')
    logger.debug('PSF phi: {} degrees'.format(args.PSF_phi))
    logger.debug('PSF ellip: {}'.format(args.PSF_ellip))
    logger.debug('PSF FWHM: {} arcsec'.format(args.PSF_FWHM))
    logger.debug('PSF sqrt(r^2): {}'.format(r2_PSF))
    logger.debug('PSF alpha: {}'.format(args.alpha))

    if not args.noDCR:
        logger.debug('')
        logger.debug('Observation settings')
        logger.debug('--------------------')
        logger.debug('zenith angle: {} degrees'.format(args.zenith_angle))
        logger.debug('parallactic angle: {} degrees'.format(args.parallactic_angle))

    logger.debug('')
    logger.debug('Galaxy settings')
    logger.debug('---------------')
    logger.debug('Galaxy Sersic index: {}'.format(args.sersic_n))
    logger.debug('Galaxy ellipticity: {}'.format(args.gal_ellip))
    logger.debug('Galaxy sqrt(r^2): {} arcsec'.format(args.gal_r2))
    logger.debug('Galaxy HLR: {:6.3f} arcsec'.format(gparam['hlr'].value))
    logger.debug('Galaxy PSF-convolved FWHM: {:6.3f} +/- {:6.3f} arcsec'.format(
        gal_fwhm, gal_fwhm_err))

    # If a diagnostic FITS file is requested, then set this up here, and write the images of the
    # PSFs to the output FITS file.
    hdulist = None
    if args.diagnostic is not None:
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(target_tool.get_PSF_image(oversample=4).array, name='GALPSF'))
        hdulist.append(fits.ImageHDU(fit_tool.get_PSF_image(oversample=4).array, name='STARPSF'))

    # The chroma.measure_shear_calib() function requires several input function arguments that we
    # haven't defined yet.  The first of these is target_image_generator, which takes input
    # arguments `gamma` (shear) and `beta` (angle around ring in radians) and produces the
    # appropriate galsim.Image
    gen_target_image = chroma.TargetImageGenerator(gparam, target_tool, hdulist=hdulist)

    # The second function is get_ring_params, which has the same arguments as gen_target_image,
    # but just returns initial guesses for the lmfit parameters needed by measurer.
    def get_ring_params(gamma, beta):
        return fit_tool.get_ring_params(gparam, beta, galsim.Shear(g1=gamma.real, g2=gamma.imag))

    # The third function is measurer, which has arguments of the target GalSim.Image from
    # gen_target_image, and initial lmfit parameters from get_ring_params, and measures the
    # ellipticity of the target image (usually assuming the "wrong" PSF, or a perturbation to
    # the wrong PSF that attempts to correct the wrongness.).
    if args.hsm:
        measurer = chroma.HSMEllipMeasurer(fit_tool)
    else:
        measurer = chroma.LSTSQEllipMeasurer(fit_tool, hdulist=hdulist)

    # Finally, we pass these functions to measure_shear_calib, along with the number of angles
    # around the ring we want to test at, and a gparam describing the galaxy we're testing.  This
    # function spits out two tuples: (m1, m2) and (c1, c2).
    m, c = chroma.measure_shear_calib(gparam, gen_target_image, get_ring_params, measurer,
                                      nring=args.nring)

    # Close the diagnostic FITS file if it was opened above.
    if args.diagnostic is not None:
        path, base = os.path.split(args.diagnostic)
        if path is not '':
            if not os.path.isdir(path):
                os.mkdir(path)
        hdulist.writeto(args.diagnostic, clobber=True)

    # And print the results!

    logger.info('')
    logger.info('Shear Calibration Results')
    logger.info('-------------------------')
    logger.info(('        ' + ' {:>9s}'*4).format('m1','m2','c1','c2'))
    logger.info(('analytic' + ' {:9.4f}'*2 + ' {:9.4f}'*2).format(m1, m2, c1, c2))
    logger.info(('ring    ' + ' {:9.4f}'*2 + ' {:9.4f}'*2).format(m[0], m[1], c[0], c[1]))
    logger.info('')
    logger.info('Survey requirements')
    logger.info(('DES     ' + ' {:9.4f}'*2 + ' {:9.4f}'*2).format(0.008, 0.008, 0.0025, 0.0025))
    logger.info(('LSST    ' + ' {:9.4f}'*2 + ' {:9.4f}'*2).format(0.003, 0.003, 0.0015, 0.0015))

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
    parser.add_argument('-z', '--redshift', type=float, default=0.0,
                        help="galaxy redshift (Default 0.0)")
    parser.add_argument('--thin', type=float, default=1.e-8,
                        help="Thin but retain bandpass integral accuracy to this relative amount."
                        +" (Default: 1.e-8).")

    # Observation input arguments
    parser.add_argument('-za', '--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('-q', '--parallactic_angle', default=0.0, type=float,
                        help="parallactic angle in degrees for differential chromatic refraction " +
                             "computation (Default 0.0)")

    # PSF structural arguments
    parser.add_argument('--kolmogorov', action='store_true',
                        help="Use Kolmogorov PSF (Default Gaussian)")
    parser.add_argument('--moffat', action='store_true',
                        help="Use Moffat PSF (Default Gaussian)")
    parser.add_argument('--PSF_beta', type=float, default=2.5,
                        help="Set beta parameter of Moffat profile PSF. (Default 2.5)")
    parser.add_argument('--PSF_FWHM', type=float, default=0.7,
                        help="Set FWHM of PSF in arcsec (Default 0.7).")
    parser.add_argument('--PSF_r2', type=float,
                        help="Override PSF_FWHM with second moment radius sqrt(r^2).")
    parser.add_argument('--PSF_phi', type=float, default=0.0,
                        help="Set position angle of PSF in degrees (Default 0.0).")
    parser.add_argument('--PSF_ellip', type=float, default=0.0,
                        help="Set ellipticity of PSF (Default 0.0)")

    # Galaxy structural arguments
    parser.add_argument('-n', '--sersic_n', type=float, default=0.5,
                        help='Sersic index (Default 0.5)')
    parser.add_argument('--gal_ellip', type=float, default=0.3,
                        help="Set ellipticity of galaxy (Default 0.3)")
    parser.add_argument('--gal_r2', type=float, default=0.27,
                        help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    parser.add_argument('--gal_convFWHM', type=float,
                        help="Override gal_r2 by setting galaxy PSF-convolved FWHM.")
    parser.add_argument('--gal_HLR', type=float,
                        help="Override gal_r2 by setting galaxy half-light-radius.")

    # Simulation input arguments
    parser.add_argument('--nring', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--image_x0', type=float, default=0.0,
                        help="Image origin x-offset in pixels (Default 0.0)")
    parser.add_argument('--image_y0', type=float, default=0.0,
                        help="Image origin y-offset in pixels (Default 0.0)")
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
    parser.add_argument('--diagnostic',
                        help="Filename to which to write diagnostic images (Default: '')")
    parser.add_argument('--hsm', action='store_true',
                        help="Use HSM regaussianization to estimate ellipticity")
    parser.add_argument('--perturb', action='store_true',
                        help="Use PerturbFastChromaticSersicTool to estimate ellipticity")
    parser.add_argument('--quiet', action='store_true',
                        help="Don't print settings")

    # and run the program...
    args = parser.parse_args()
    one_ring_test(args)
