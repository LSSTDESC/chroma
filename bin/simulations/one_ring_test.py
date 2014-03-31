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

def fiducial_galaxy():
    """Setup lmfit.Parameters to represent a Single Sersic galaxy.
    """
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('hlr', value=0.27)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)
    return gparam

class TargetImageGenerator(object):
    """ Use a galtool to generate a target image and optionally append an ImageHDU with galaxy
    parameters used to create it to a FITS HDUList.
    """
    def __init__(self, gparam, galtool, hdulist=None, oversample=4):
        self.gparam = gparam.copy()
        self.galtool = galtool
        self.hdulist = hdulist
        self.oversample = oversample
    def __call__(self, gamma, beta):
        shear = galsim.Shear(g1=gamma.real, g2=gamma.imag)
        target_image = self.galtool.get_image(self.gparam, ring_beta=beta, ring_shear=shear)
        if self.hdulist is not None:
            hdu = fits.ImageHDU(target_image.array, name='TARGET')
            hdu.header.append(('GAMMA1', gamma.real))
            hdu.header.append(('GAMMA2', gamma.imag))
            hdu.header.append(('BETA', beta))
            for k,v in self.gparam.iteritems():
                hdu.header.append((k, v.value))
            self.hdulist.append(hdu)
            target_high_res = self.galtool.get_image(self.gparam, ring_beta=beta, ring_shear=shear,
                                                     oversample=self.oversample)
            self.hdulist.append(fits.ImageHDU(target_high_res.array, name='TARGETHR'))
            target_uncvl = self.galtool.get_uncvl_image(self.gparam, ring_beta=beta,
                                                        ring_shear=shear,
                                                        oversample=self.oversample,
                                                        center=True)
            self.hdulist.append(fits.ImageHDU(target_uncvl.array, name='TARGETUC'))
        return target_image

class EllipMeasurer(object):
    """ Measure ellipticity and optionally append an ImageHDU with best fit galaxy parameters
    to a FITS HDUList.
    """
    def __init__(self, galtool, hdulist=None, oversample=4):
        self.galtool = galtool
        self.hdulist = hdulist
        self.oversample = oversample
    def __call__(self):
        raise NotImplementedError

class LSTSQEllipMeasurer(EllipMeasurer):
    """ Measure ellipticity by performing a least-squares fit over galaxy parameters.
    """
    def resid(self, param, target_image):
        image = self.galtool.get_image(param)
        return (image.array - target_image.array).flatten()
    def __call__(self, target_image, init_param):
        result = lmfit.minimize(self.resid, init_param, args=(target_image,))
        if self.hdulist is not None:
            fit_image = self.galtool.get_image(result.params)
            hdu = fits.ImageHDU(fit_image.array, name='FIT')
            for k,v in result.params.iteritems():
                hdu.header.append((k, v.value))
            self.hdulist.append(hdu)
            fit_image_high_res = self.galtool.get_image(result.params, oversample=self.oversample)
            self.hdulist.append(fits.ImageHDU(fit_image_high_res.array, name='FITHR'))
            fit_image_uncvl = self.galtool.get_uncvl_image(result.params,
                                                           oversample=self.oversample,
                                                           center=True)
            self.hdulist.append(fits.ImageHDU(fit_image_uncvl.array, name='FITUC'))
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        return gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))

class HSMEllipMeasurer(EllipMeasurer):
    """ Use the Hirata-Seljak-Mandelbaum regaussianization PSF correction algorithm to estimate
    ellipticity.
    """
    def psf_image(self):
        if not hasattr(self, '_psf_image'):
            self._psf_image = self.galtool.get_PSF_image()
        return self._psf_image

    def __call__(self, target_image, init_param):
        psf_image = self.psf_image()
        results = galsim.hsm.EstimateShear(target_image, psf_image)
        ellip = galsim.Shear(e1=results.corrected_e1, e2=results.corrected_e2)
        return complex(ellip.g1, ellip.g2)

def measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF, pixel_scale, stamp_size,
                        ring_n, galtool, diagfile=None, hsm=False, maximum_fft_size=32768,
                        deltaRbar=None, deltaV=None, r2byr2=None, offset=(0,0)):
    """Perform two ring tests to solve for shear calibration parameters `m` and `c`."""

    gsparams = galsim.GSParams()
    gsparams.maximum_fft_size = maximum_fft_size
    target_tool = galtool(gal_SED, bandpass, PSF, stamp_size, pixel_scale,
                          offset=offset, gsparams=gsparams)
    if galtool == chroma.PerturbFastChromaticSersicTool:
        fit_tool = galtool(star_SED, bandpass, PSF, stamp_size, pixel_scale,
                           deltaRbar, deltaV, r2byr2,
                           offset=offset, gsparams=gsparams)
    else:
        fit_tool = galtool(star_SED, bandpass, PSF, stamp_size, pixel_scale,
                           offset=offset, gsparams=gsparams)

    hdulist=None
    if diagfile is not None:
        hdulist=fits.HDUList()
        hdulist.append(fits.ImageHDU(target_tool.get_PSF_image(oversample=4).array, name='GALPSF'))
        hdulist.append(fits.ImageHDU(fit_tool.get_PSF_image(oversample=4).array, name='STARPSF'))

    gen_target_image = TargetImageGenerator(gparam, target_tool, hdulist=hdulist)
    if hsm:
        measure_ellip = HSMEllipMeasurer(fit_tool)
    else:
        measure_ellip = LSTSQEllipMeasurer(fit_tool, hdulist=hdulist)

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

    if diagfile is not None:
        hdulist.writeto(diagfile, clobber=True)

    return m, c

def one_ring_test(args):
    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger("one_ring_test")
    if args.quiet:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    # build filter bandpass
    bandpass = chroma.Bandpass(args.datadir+args.filter)

    # build galaxy SED
    gal_SED = chroma.SED(args.datadir+args.galspec, flux_type='flambda')
    gal_SED = gal_SED.atRedshift(args.redshift)

    # build G5v star SED
    star_SED = chroma.SED(args.datadir+args.starspec)

    # Thin if requested
    if args.thin is not None:
        gal_SED = gal_SED.thin(args.thin)
        star_SED = star_SED.thin(args.thin)
        bandpass = bandpass.thin(args.thin)

    # Use effective wavelength to set FWHM
    PSF_wave = bandpass.effective_wavelength

    # scale SEDs
    gal_SED = gal_SED.withFlux(1.0, bandpass)
    star_SED = star_SED.withFlux(1.0, bandpass)

    logger.debug('')
    logger.debug('General settings')
    logger.debug('----------------')
    logger.debug('stamp size: {}'.format(args.stamp_size))
    logger.debug('pixel scale: {} arcsec/pixel'.format(args.pixel_scale))
    logger.debug('ring test angles: {}'.format(args.ring_n))

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

    # Define the PSF
    if args.moffat:
        monoPSF = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    else:
        monoPSF = galsim.Gaussian(fwhm=args.PSF_FWHM)
    monoPSF.applyShear(g=args.PSF_ellip, beta=args.PSF_phi * galsim.degrees)
    if not args.noDCR: #include DCR
        PSF = galsim.ChromaticAtmosphere(monoPSF, base_wavelength=PSF_wave,
                                         zenith_angle=args.zenith_angle * galsim.degrees,
                                         parallactic_angle=args.parallactic_angle * galsim.degrees,
                                         alpha=args.alpha)
    else: #otherwise just include a powerlaw wavelength dependent FWHM
        PSF = galsim.ChromaticObject(monoPSF)
        PSF.applyDilation(lambda w:(w/PSF_wave)**args.alpha)

    logger.debug('')
    if args.moffat:
        logger.debug('Moffat PSF settings')
        logger.debug('-------------------')
        logger.debug('PSF beta: {}'.format(args.PSF_beta))
    else:
        logger.debug('Gaussian PSF settings')
        logger.debug('---------------------')
    logger.debug('PSF phi: {}'.format(args.PSF_phi))
    logger.debug('PSF ellip: {}'.format(args.PSF_ellip))
    logger.debug('PSF FWHM: {} arcsec'.format(args.PSF_FWHM))
    logger.debug('PSF alpha: {}'.format(args.alpha))

    # Go ahead and calculate sqrt(r^2) for PSF here...
    # Ignoring corrections due to ellipticity for now.
    if args.moffat:
        r2_psf = args.PSF_FWHM * np.sqrt(2.0 /
                                         (8.0*(2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0)))
    else: #gaussian
        r2_psf = args.PSF_FWHM * np.sqrt(2.0/np.log(256.0))

    logger.debug('PSF sqrt(r^2): {}'.format(r2_psf))

    if not args.noDCR:
        logger.debug('')
        logger.debug('Observation settings')
        logger.debug('--------------------')
        logger.debug('zenith angle: {} degrees'.format(args.zenith_angle))
        logger.debug('parallactic angle: {} degrees'.format(args.parallactic_angle))

    if args.slow:
        galtool = chroma.ChromaticSersicTool
    else:
        galtool = chroma.FastChromaticSersicTool

    gparam = fiducial_galaxy()
    gparam['n'].value = args.sersic_n
    gparam['x0'].value = args.gal_x0 * args.pixel_scale
    gparam['y0'].value = args.gal_y0 * args.pixel_scale
    gparam['gmag'].value = args.gal_ellip
    offset = (args.image_x0, args.image_y0)
    gtool = galtool(gal_SED, bandpass, PSF, args.stamp_size, args.pixel_scale, offset=offset)
    gparam = gtool.set_uncvl_r2(gparam, args.gal_r2)

    logger.debug('')
    logger.debug('Galaxy settings')
    logger.debug('---------------')
    logger.debug('Galaxy Sersic index: {}'.format(args.sersic_n))
    logger.debug('Galaxy ellipticity: {}'.format(args.gal_ellip))
    logger.debug('Galaxy x-offset: {} arcsec'.format(args.gal_x0))
    logger.debug('Galaxy y-offset: {} arcsec'.format(args.gal_y0))
    logger.debug('Galaxy sqrt(r^2): {} arcsec'.format(args.gal_r2))
    gal_fwhm, gal_fwhm_err = gtool.compute_FWHM(gparam)
    logger.debug('Galaxy FWHM: {:6.3f} +/- {:6.3f} arcsec'.format(gal_fwhm, gal_fwhm_err))


    # Analytic estimate of shear bias

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
        logger.debug("star seeing correction: {}".format(seeing1))
        logger.debug("galaxy seeing correction: {}".format(seeing2))
        r2byr2 = seeing2/seeing1
    else:
        dr2r2 = 0.0
        r2byr2 = 1.0

    dIxx = (r2_psf**2/2.0) * dr2r2
    dIxy = 0.0
    dIyy = (r2_psf**2/2.0) * dr2r2
    dIyy += dV

    m1 = m2 = -(dIxx + dIyy) / args.gal_r2**2
    c1 = (dIxx-dIyy) / (2.0 * (args.gal_r2**2))
    c2 = dIxy / args.gal_r2**2

    if args.perturb:
        galtool = chroma.PerturbFastChromaticSersicTool

    # Measure shear bias with ring test
    m, c = measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF,
                               args.pixel_scale, args.stamp_size, args.ring_n,
                               galtool, args.diagnostic, args.hsm, r2byr2=r2byr2,
                               deltaV = dV, offset=offset)

    # And ... drumroll ... results!

    logger.info('')
    logger.info('Shear Calibration Results')
    logger.info('-------------------------')
    logger.info(('        ' + ' {:>10s}'*4).format('m1','m2','c1','c2'))
    logger.info(('analytic' + ' {:10.5f}'*2 + ' {:10.6f}'*2).format(m1, m2, c1, c2))
    logger.info(('ring    ' + ' {:10.5f}'*2 + ' {:10.6f}'*2).format(m[0], m[1], c[0], c[1]))
    logger.info(('DES req ' + ' {:10.5f}'*2 + ' {:10.6f}'*2).format(0.008, 0.008, 0.0025, 0.0025))
    logger.info(('LSST req' + ' {:10.5f}'*2 + ' {:10.6f}'*2).format(0.003, 0.003, 0.0015, 0.0015))

def runme():
    """Useful for profiling one_ring_test() using IPython and prun.
    """
    class junk(object): pass
    args = junk()
    args.datadir = '../../data/'
    args.starspec = 'SEDs/ukg5v.ascii'
    args.galspec = 'SEDs/CWW_E_ext.ascii'
    args.redshift = 0.0
    args.filter = 'filters/LSST_r.dat'
    args.zenith_angle = 45.0
    args.gaussian = False
    args.PSF_beta = 2.5
    args.PSF_FWHM = 0.7
    args.PSF_phi = 0.0
    args.PSF_ellip = 0.0
    args.sersic_n = 0.5
    args.gal_ellip = 0.3
    args.gal_x0 = 0.0
    args.gal_y0 = 0.0
    args.gal_r2 = 0.27
    args.ring_n = 3
    args.pixel_scale = 0.2
    args.stamp_size = 31
    args.thin = 1.e-5
    args.slow = False
    args.alpha = -0.2
    args.noDCR = False
    one_ring_test(args)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', default='../../data/',
                        help="directory to find SED and filter files.")
    parser.add_argument('-s', '--starspec', default='SEDs/ukg5v.ascii',
                        help="stellar spectrum to use when fitting (Default 'SEDs/ukg5v.ascii')")
    parser.add_argument('-g', '--galspec', default='SEDs/CWW_E_ext.ascii',
                        help="galactic spectrum used to create target image " +
                             "(Default 'SEDs/CWW_E_ext.ascii')")
    parser.add_argument('-z', '--redshift', type=float, default=0.0,
                        help="galaxy redshift (Default 0.0)")
    parser.add_argument('-f', '--filter', default='filters/LSST_r.dat',
                        help="filter for simulation (Default 'filters/LSST_r.dat')")
    parser.add_argument('-za', '--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('-q', '--parallactic_angle', default=0.0, type=float,
                        help="parallactic angle in degrees for differential chromatic refraction " +
                             "computation (Default 0.0)")
    parser.add_argument('--moffat', action='store_true',
                        help="Use Moffat PSF (Default Gaussian)")
    parser.add_argument('--PSF_beta', type=float, default=2.5,
                        help="Set beta parameter of Moffat profile PSF. (Default 2.5)")
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
    parser.add_argument('--ring_n', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--image_x0', type=float, default=0.0,
                        help="Image origin x-offset")
    parser.add_argument('--image_y0', type=float, default=0.0,
                        help="Image origin y-offset")
    parser.add_argument('--thin', type=float, default=1.e-4,
                        help="Thin but retain bandpass integral accuracy to this relative amount."
                        +" (Default: 1.e-4).")
    parser.add_argument('--slow', action='store_true',
                        help="Use ChromaticSersicTool (somewhat more careful) instead of "
                            +"FastChromaticSersicTool.")
    parser.add_argument('--alpha', type=float, default=-0.2,
                        help="Power law index for chromatic seeing (Default: -0.2)")
    parser.add_argument('--noDCR', action='store_true',
                        help="Exclude differential chromatic refraction (DCR) in PSF."
                        +" (Default: include DCR)")
    parser.add_argument('--diagnostic',
                        help="Filename to which to write diagnostic images (Default: '')")
    parser.add_argument('--hsm', action='store_true',
                        help="Use HSM regaussianization to estimate ellipticity")
    parser.add_argument('--perturb', action='store_true',
                        help="Use PerturbFastChromaticSersicTool to estimate ellipticity")
    parser.add_argument('--quiet', action='store_true',
                        help="Don't print ring test settings")

    args = parser.parse_args()

    one_ring_test(args)
