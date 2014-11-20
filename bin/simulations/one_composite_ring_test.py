"""Perform a ring test for a specific combination of star, galaxy, and PSF structural and spectral
parameters.  Differs from `one_ring_test.py` in that the galaxy is modeled as sum of coelliptical
Sersic profiles.  Use `python one_composite_ring_test.py --help` to see a list of available command
line options.
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

def one_composite_ring_test(args):
    """ Run a single ring test for a composite galaxy.  There are many configurable options here.
    From the command-line, run `python one_composite_ring_test.py --help` to see them.
    """
    # setup logging
    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger("one_composite_ring_test")
    if args.quiet:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    #----------------------------------------------------------------------------------------------
    # Defaults                                                                                    #
    #----------------------------------------------------------------------------------------------

    # We set default values depending whether or not the --voigt command-line option is set.  The
    # --voigt option is intended to reproduce the fiducial model in the Voigt++12 paper discussing
    # shear biases for galaxies with color-gradients, observed with Euclid.
    if args.voigt:
        # Can't override following when --voigt is set.
        args.noDCR = True
        # These you can override, but defaults are set for Voigt++12 comparison.
        if args.filter is None:
            args.filter = 'filters/Euclid_350.dat'
        if args.redshift is None:
            args.redshift = 0.9
        if args.alpha is None:
            args.alpha = 0.6
        if args.gal_ellip is None:
            args.gal_ellip = 0.2
        if args.bulge_HLR is None:
            args.bulge_HLR = 1.1
        if args.disk_HLR is None:
            args.disk_HLR = 1.0
        if args.pixel_scale is None:
            args.pixel_scale = 1.0
        if args.PSF_FWHM is None:
            args.PSF_FWHM = 1.4
        if args.PSF_wave is None:
            args.PSF_wave = 520.0
        if args.PSF_ellip is None:
            args.PSF_ellip = 0.05
        if args.stamp_size is None:
            args.stamp_size = 15
        if args.FWHM_ratio is None:
            args.FWHM_ratio = 1.4
    else:
        # Different set of overridable defaults if not doing Voigt++12 comparison.
        if args.filter is None:
            args.filter = 'filters/LSST_r.dat'
        if args.redshift is None:
            args.redshift = 0.0
        if args.alpha is None:
            args.alpha = -0.2
        if args.gal_ellip is None:
            args.gal_ellip = 0.3
        if args.pixel_scale is None:
            args.pixel_scale = 0.2
        if args.PSF_FWHM is None:
            args.PSF_FWHM = 0.7
        if args.PSF_ellip is None:
            args.PSF_ellip = 0.0
        if args.stamp_size is None:
            args.stamp_size = 31

    #----------------------------------------------------------------------------------------------
    # Spectra setup                                                                               #
    #----------------------------------------------------------------------------------------------

    # load filter bandpass
    bandpass = chroma.Bandpass(args.datadir+args.filter)

    # load and redshift bulge and disk SEDs
    bulge_SED = galsim.SED(args.datadir+args.bulgespec, flux_type="flambda")
    bulge_SED = bulge_SED.atRedshift(args.redshift)
    disk_SED = galsim.SED(args.datadir+args.diskspec, flux_type="flambda")
    disk_SED = disk_SED.atRedshift(args.redshift)

    # normalize SEDs to 1.0, let gparams below handle the actual relative fluxes of
    # the bulge and disk components
    bulge_SED = bulge_SED.withFlux(1.0, bandpass)
    disk_SED = disk_SED.withFlux(1.0, bandpass)

    # load stellar SED
    # By default, create stellar spectrum as flux-weighted average SED of galaxy bulge and disk.
    # Though, if specified, override with input filename.
    if args.starspec is None:
        star_SED = args.bulge_frac*bulge_SED + (1.0-args.bulge_frac)*disk_SED
    else:
        star_SED = galsim.SED(args.datadir+args.starspec, flux_type="flambda")
    star_SED = star_SED.withFlux(1.0, bandpass)

    # Thin bandpass and spectra if requested
    if args.thin is not None:
        bulge_SED = bulge_SED.thin(args.thin)
        disk_SED = disk_SED.thin(args.thin)
        star_SED = star_SED.thin(args.thin)
        bandpass = bandpass.thin(args.thin)

    # Use effective wavelength to set FWHM
    if args.PSF_wave is None:
        PSF_wave = bandpass.effective_wavelength
    else:
        PSF_wave = float(args.PSF_wave)

    #----------------------------------------------------------------------------------------------
    # PSF setup                                                                                   #
    #----------------------------------------------------------------------------------------------

    # There are several command-line options to set the size of the PSF.  By default, the size is
    # set such that the FWHM is 0.7 arcseconds at the filter effective wavelength, or if the --voigt
    # option is used, such that FWHM = 1.4 pixels at wavelength of 520.0 nm. To change this, use the
    # --PSF_FWHM option.
    # To set the PSF size using the second-moment squared radius instead, use --PSF_r2
    # Note that it is an error to specify both --PSF_FWHM and --PSF_r2.

    # If it exists, convert PSF_r2 to PSF_FWHM using the details of the PSF profile.
    # If it doesn't exist, then convert PSF_FWHM into PSF_r2.
    if args.PSF_r2 is not None:
        if args.moffat:
            args.PSF_FWHM = args.PSF_r2 / np.sqrt(
                2.0 / (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
        elif args.kolmogorov:
            raise ValueError("Kolmogorov profile not implemented yet!")
        else: # default is Gaussian
            args.PSF_FWHM = args.PSF_r2 / np.sqrt(2.0/np.log(256.0))
    else:
        if args.moffat:
            args.PSF_r2 = args.PSF_FWHM * np.sqrt(
                2.0 / (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
        elif args.kolmogorov:
            raise ValueError("Kolmogorov profile not implemented yet!")
        else: # default is Gaussian
            args.PSF_r2 = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))

    # Create circularly-symmetric, monochromatic, PSF with requested profile.
    if args.moffat:
        circularPSF = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    elif args.kolmogorov:
        circularPSF = galsim.Kolmogorov(lam_over_r0=args.PSF_FWHM/0.976)
    else:
        circularPSF = galsim.Gaussian(fwhm=args.PSF_FWHM)
    # If requested, add PSF ellipticity, but keep circularized PSF around for later.
    monochromaticPSF = circularPSF.shear(
            g=args.PSF_ellip, beta=args.PSF_phi * galsim.degrees)
    # Turn monochromatic PSF into a chromatic PSF by specifying the wavelength-dependence, either
    # with the ChromaticAtmosphere function (which implements chromatic seeing and differential
    # chromatic refraction (DCR), or with wavelength-dependent power-law dilation if no DCR.
    if not args.noDCR:
        PSF = galsim.ChromaticAtmosphere(monochromaticPSF, base_wavelength=PSF_wave,
                                         zenith_angle=args.zenith_angle*galsim.degrees,
                                         parallactic_angle=args.parallactic_angle*galsim.degrees,
                                         alpha=args.alpha)
        circularPSF = galsim.ChromaticAtmosphere(circularPSF, base_wavelength=PSF_wave,
                                         zenith_angle=args.zenith_angle*galsim.degrees,
                                         parallactic_angle=args.parallactic_angle*galsim.degrees,
                                         alpha=args.alpha)
    else:
        PSF = galsim.ChromaticObject(monochromaticPSF)
        PSF = PSF.dilate(lambda w:(w/PSF_wave)**args.alpha)
        circularPSF = galsim.ChromaticObject(circularPSF)
        circularPSF = circularPSF.dilate(lambda w:(w/PSF_wave)**args.alpha)

    #----------------------------------------------------------------------------------------------
    # Galaxy setup                                                                                #
    #----------------------------------------------------------------------------------------------

    # Now create some galtools, which are objects that know how to draw images given some
    # parameters, thus bridging GalSim and lmfit.

    # We use image offsets to investigate subpixel dithers, since using GalSim shifts would have
    # the effect of also changing the intended meaning of shears and rotations, since these are
    # defined with respect to the galaxy origin.
    offset = (args.image_x0, args.image_y0)
    # The target_tool will be used to generate "true" images of the simulated galaxy, by which I
    # mean that the PSF is actually derived from the galactic SEDs (a different effective PSF for
    # the bulge and for the disk).  Note that we also have to supply some bookkeeping arguments
    # defining the image size, scale, and offset.  The galaxy SED and filter bandpass arguments
    # come last, since they are optional.  If not specified, then the assumption is that PSFs are
    # GalSim.GSObjects, and not GalSim.ChromaticObjects, and thus that the SED and bandpass are not
    # needed to draw the PSF-convolved image.
    target_tool = chroma.DoubleSersicTool(PSF, args.stamp_size, args.pixel_scale, offset,
                                          bulge_SED, disk_SED, bandpass)
    circ_tool = chroma.DoubleSersicTool(circularPSF, args.stamp_size, args.pixel_scale, offset,
                                          star_SED, star_SED, bandpass)
    # The fit_tool is basically the same as the target_tool, except that we assert a different --
    # incorrect -- SED to investigate the effect of deriving the PSF model from a nearby star with
    # a different SED than the galaxy.
    fit_tool = chroma.DoubleSersicTool(PSF, args.stamp_size, args.pixel_scale, offset,
                                       star_SED, star_SED, bandpass)
    # By default, we compress the wavelength-dependent PSFs into effective PSFs by integrating
    # them against the bandpass throughput over wavelength.  This is much faster than letting
    # GalSim do the wavelength integration every time you draw a target image or candidate fit
    # image.
    if not args.slow:
        target_tool.use_effective_PSF()
        fit_tool.use_effective_PSF()

    # Now build up requested attributes of the galaxy.
    gparam = target_tool.default_galaxy()
    gparam['n_1'].value = args.bulge_n
    gparam['n_2'].value = args.disk_n
    gparam['flux_1'].value = args.bulge_frac
    gparam['flux_2'].value = 1.0 - args.bulge_frac
    # There are a few ways to set the size of the galaxy:
    # --bulge_HLR, --disk_HLR - set the half-light-radii of the bulge and disk components.  These
    #                           are the pre-PSF-convolution and circularized radii.
    # --bulge_r2, --disk_r2 - set the (square root of the) second-moment squared radii of the bulge
    #                         and disk.
    # --FWHM_ratio - set the ratio of the PSF-convolved galaxy full-width-half-maximum to
    #                PSF FWHM.  This is done empirically by iteratively drawing images.  The
    #                ratio of bulge and disk half-light-radii are held fixed during this process.
    if args.bulge_HLR is not None:
        gparam['hlr_1'].value = args.bulge_HLR
    else:
        gparam = target_tool.set_uncvl_r2(gparam, args.bulge_r2)
    if args.disk_HLR is not None:
        gparam['hlr_2'].value = args.disk_HLR
    else:
        gparam = target_tool.set_uncvl_r2(gparam, args.disk_r2)
    if args.FWHM_ratio is not None:
        integrated_PSF_FWHM = circ_tool.compute_PSF_FWHM(oversample=16)[0]
        gparam = circ_tool.set_FWHM(gparam, args.FWHM_ratio * integrated_PSF_FWHM)
    # Once the size is set, can set the ellipticity.
    gparam['g_1'].value = args.gal_ellip
    gparam['g_2'].value = args.gal_ellip

    #----------------------------------------------------------------------------------------------
    # Echo Configuration Options                                                                  #
    #----------------------------------------------------------------------------------------------

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
    logger.debug('Thinning with relative error: {}'.format(args.thin))
    logger.debug('Bulge fraction: {}'.format(args.bulge_frac))
    logger.debug('Bulge SED: {}'.format(args.bulgespec))
    logger.debug('Disk SED: {}'.format(args.diskspec))
    logger.debug('Galaxy redshift: {}'.format(args.redshift))
    if args.starspec is None:
        logger.debug('Star SED: composite bulge + disk spectrum')
    else:
        logger.debug('Star SED: {}'.format(args.starspec))
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
    logger.debug('PSF size wavelength: {} nm'.format(PSF_wave))
    logger.debug('PSF FWHM: {} arcsec'.format(args.PSF_FWHM))
    if args.FWHM_ratio is not None:
        logger.debug('PSF integrated FWHM: {}'.format(integrated_PSF_FWHM))
    logger.debug('PSF sqrt(r^2): {}'.format(args.PSF_r2))
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
    logger.debug('Bulge Sersic index: {}'.format(args.bulge_n))
    logger.debug('Disk Sersic index: {}'.format(args.disk_n))
    logger.debug('Galaxy ellipticity: {}'.format(args.gal_ellip))

    #----------------------------------------------------------------------------------------------
    # Ring Test                                                                                   #
    #----------------------------------------------------------------------------------------------

    # The chroma.measure_shear_calib() function requires several input function arguments that we
    # haven't defined yet.  The first of these is target_image_generator, which takes input
    # arguments `gamma` (shear) and `beta` (angle around ring in radians) and produces the
    # appropriate galsim.Image
    gen_target_image = chroma.TargetImageGenerator(gparam, target_tool)

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
        measurer = chroma.LSTSQEllipMeasurer(fit_tool)

    # Finally, we pass these functions to measure_shear_calib, along with the number of angles
    # around the ring we want to test at, and a gparam describing the galaxy we're testing.  This
    # function spits out two tuples: (m1, m2) and (c1, c2).
    m, c = chroma.measure_shear_calib(gparam, gen_target_image, get_ring_params, measurer,
                                      nring=args.nring)

    #----------------------------------------------------------------------------------------------
    # Results!                                                                                    #
    #----------------------------------------------------------------------------------------------

    logger.info('')
    logger.info('Shear Calibration Results')
    logger.info('-------------------------')
    logger.info(('        ' + ' {:>10s}'*4).format('m1','m2','c1','c2'))
    logger.info(('ring    ' + ' {:10.5f}'*4).format(m[0], m[1], c[0], c[1]))
    logger.info('')
    logger.info('Survey requirements')
    logger.info(('DES     ' + ' {:10.5f}'*4).format(0.008, 0.008, 0.0025, 0.0025))
    logger.info(('LSST    ' + ' {:10.5f}'*4).format(0.003, 0.003, 0.0015, 0.0015))
    logger.info(('Euclid  ' + ' {:10.5f}'*4).format(0.002, 0.002, 0.0002, 0.0002))

if __name__ == '__main__':
    parser = ArgumentParser()

    # Input data file arguments
    parser.add_argument("--datadir", default="../../data/",
                        help="directory to find SED and filter files.")
    fitspec = parser.add_mutually_exclusive_group()
    fitspec.add_argument("--starspec",
                         help="stellar spectrum to use when fitting (Default use composite"
                              +" bulge+disk spectrum)")
    parser.add_argument("--bulgespec", default="SEDs/CWW_E_ext.ascii",
                        help="bulge spectrum of true galaxy (Default 'SEDs/CWW_E_ext.ascii')")
    parser.add_argument("--diskspec", default="SEDs/CWW_Sbc_ext.ascii",
                        help="bulge spectrum of true galaxy (Default 'SEDs/CWW_Sbc_ext.ascii')")
    parser.add_argument("-f", "--filter", type=str,
                        help="filter for simulation (Default 'filters/LSST_r.dat')")

    # Spectrum treatment arguments
    parser.add_argument("-z", "--redshift", type=float,
                        help="galaxy redshift (Default 0.0)")
    parser.add_argument("--thin", type=float, default=1.e-8,
                        help="Thin but retain bandpass integral accuracy to this relative amount."
                        +" (Default: 1.e-8).")

    # Observation input arguments
    parser.add_argument("-za", "--zenith_angle", default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default: 45.0)")
    parser.add_argument("-q", "--parallactic_angle", default=0.0, type=float,
                        help="parallactic angle in degrees for differential chromatic refraction " +
                             "computation (Default: 0.0)")

    # PSF structural arguments
    PSF_profile = parser.add_mutually_exclusive_group()
    PSF_profile.add_argument("--kolmogorov", action="store_true",
                             help="Use Kolmogorov PSF (Default Gaussian)")
    PSF_profile.add_argument("--moffat", action="store_true",
                             help="Use Moffat PSF (Default Gaussian)")
    parser.add_argument("--PSF_beta", type=float, default=3.0,
                        help="Set beta parameter of Moffat profile PSF. (Default 2.5)")
    PSF_size = parser.add_mutually_exclusive_group()
    PSF_size.add_argument("--PSF_FWHM", type=float,
                          help="Set FWHM of PSF in arcsec (Default 0.7).")
    PSF_size.add_argument("--PSF_r2", type=float,
                          help="Override PSF_FWHM with second moment radius sqrt(r^2).")
    parser.add_argument("--PSF_wave", type=float,
                        help="Wavelength (in nm) at which to set PSF size. "
                        +"(Default: filter effective wavelength.")
    parser.add_argument("--PSF_phi", type=float, default=0.0,
                        help="Set position angle of PSF in degrees (Default 0.0).")
    parser.add_argument("--PSF_ellip", type=float,
                        help="Set ellipticity of PSF (Default 0.0)")

    # Galaxy structural arguments
    parser.add_argument("-bn", "--bulge_n", type=float, default=4.0,
                        help="Bulge component Sersic index (Default 4.0)")
    parser.add_argument("-dn", "--disk_n", type=float, default=1.0,
                        help="Disk component Sersic index (Default 1.0)")
    parser.add_argument("--bulge_frac", type=float, default=0.25,
                        help="Fraction of total flux in bulge component (Default 0.25)")
    bulge_size = parser.add_mutually_exclusive_group()
    bulge_size.add_argument("--bulge_r2", type=float, default=0.27,
                            help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    bulge_size.add_argument("--bulge_HLR", type=float,
                            help="Override bulge_r2 by setting half-light-radius.")
    disk_size = parser.add_mutually_exclusive_group()
    disk_size.add_argument("--disk_r2", type=float, default=0.27,
                            help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    disk_size.add_argument("--disk_HLR", type=float,
                            help="Override disk_r2 by setting half-light-radius.")
    parser.add_argument("--FWHM_ratio", type=float,
                        help="ratio of (circular) PSF convolved galaxy FWHM to PSF FWHM.")
    parser.add_argument("--gal_ellip", type=float,
                        help="Set ellipticity of galaxy (Default 0.3)")

    # Simulation input arguments
    parser.add_argument("--nring", type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument("--pixel_scale", type=float,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument("--stamp_size", type=int,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument("--image_x0", type=float, default=0.0,
                        help="Image origin x-offset in pixels (Default 0.0)")
    parser.add_argument("--image_y0", type=float, default=0.0,
                        help="Image origin y-offset in pixels (Default 0.0)")
    parser.add_argument("--slow", action="store_true",
                        help="Build entire image by integrating over wavelength instead of simply"
                        +" building an effective PSF by integrating over wavelength.")

    # Physics arguments
    parser.add_argument("--alpha", type=float,
                        help="Power law index for chromatic seeing (Default: -0.2)")
    parser.add_argument("--noDCR", action="store_true",
                        help="Exclude differential chromatic refraction (DCR) in PSF."
                        +" (Default: include DCR)")

    # Miscellaneous arguments
    parser.add_argument("--hsm", action="store_true",
                        help="Use HSM regaussianization to estimate ellipticity")
    parser.add_argument("--quiet", action="store_true",
                        help="Don't print settings")
    parser.add_argument("--voigt", action="store_true",
                        help="Setup PSF, galaxy, and filter to Voigt++12 default values.")

    # and run the program...
    args = parser.parse_args()
    one_composite_ring_test(args)
