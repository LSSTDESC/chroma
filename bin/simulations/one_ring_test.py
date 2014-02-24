import os
from argparse import ArgumentParser
import logging

import lmfit
import galsim
import numpy as np
import astropy.io.fits as fits

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

class TargetImageGenerator(object):
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
            target_uncvl = self.galtool.get_uncvl_image(self.gparam, ring_beta=beta, ring_shear=shear,
                                                        oversample=self.oversample)
            self.hdulist.append(fits.ImageHDU(target_uncvl.array, name='TARGETUC'))
        return target_image

class EllipMeasurer(object):
    def __init__(self, galtool, hdulist=None, oversample=4):
        self.galtool = galtool
        self.hdulist = hdulist
        self.oversample = oversample
    def __call__(self):
        raise NotImplementedError

class LSTSQEllipMeasurer(EllipMeasurer):
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
                                                           oversample=self.oversample)
            self.hdulist.append(fits.ImageHDU(fit_image_uncvl.array, name='FITUC'))
        gmag = result.params['gmag'].value
        phi = result.params['phi'].value
        return gmag * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))

class HSMEllipMeasurer(EllipMeasurer):
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
                        ring_n, galtool, diagfile=None, use_hsm=False, maximum_fft_size=32768):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''

    gsparams = galsim.GSParams()
    gsparams.maximum_fft_size = maximum_fft_size
    target_tool = galtool(gal_SED, bandpass, PSF, stamp_size, pixel_scale, gsparams)
    fit_tool = galtool(star_SED, bandpass, PSF, stamp_size, pixel_scale, gsparams)

    hdulist=None
    if diagfile is not None:
        hdulist=fits.HDUList()
        hdulist.append(fits.ImageHDU(target_tool.get_PSF_image(oversample=4).array, name='GALPSF'))
        hdulist.append(fits.ImageHDU(fit_tool.get_PSF_image(oversample=4).array, name='STARPSF'))

    gen_target_image = TargetImageGenerator(gparam, target_tool, hdulist=hdulist)
    if use_hsm:
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
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger("one_ring_test")

    # build filter bandpass
    bandpass = chroma.Bandpass(args.datadir+args.filter)
    bandpass = bandpass.createThinned(args.thin)

    # build galaxy SED
    gal_SED = chroma.SED(args.datadir+args.galspec, flux_type='flambda')
    gal_SED = gal_SED.createRedshifted(args.redshift)

    # build G5v star SED
    star_SED = chroma.SED(args.datadir+args.starspec)

    # scale SEDs
    gal_SED = gal_SED.createWithFlux(bandpass, 1.0)
    star_SED = star_SED.createWithFlux(bandpass, 1.0)

    logger.info('')
    logger.info('General settings')
    logger.info('----------------')
    logger.info('stamp size: {}'.format(args.stamp_size))
    logger.info('pixel scale: {} arcsec/pixel'.format(args.pixel_scale))
    logger.info('ring test angles: {}'.format(args.ring_n))

    logger.info('')
    logger.info('Spectra settings')
    logger.info('----------------')
    logger.info('Data directory: {}'.format(args.datadir))
    logger.info('Filter: {}'.format(args.filter))
    logger.info('Thinning filter by factor: {}'.format(args.thin))
    logger.info('Galaxy SED: {}'.format(args.galspec))
    logger.info('Galaxy redshift: {}'.format(args.redshift))
    logger.info('Star SED: {}'.format(args.starspec))

    # Define the PSF
    if args.gaussian:
        PSF685 = galsim.Gaussian(fwhm=args.PSF_FWHM)
    else:
        PSF685 = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    PSF685.applyShear(g=args.PSF_ellip, beta=args.PSF_phi * galsim.radians)
    if not args.noDCR:
        PSF = galsim.ChromaticAtmosphere(PSF685, base_wavelength=685.0,
                                         zenith_angle=args.zenith_angle * galsim.degrees,
                                         alpha=args.alpha)
    else:
        PSF = galsim.ChromaticObject(PSF685)
        PSF.applyDilation(lambda w:(w/685)**args.alpha)

    logger.info('')
    if not args.gaussian:
        logger.info('Moffat PSF settings')
        logger.info('-------------------')
        logger.info('PSF beta: {}'.format(args.PSF_beta))
    else:
        logger.info('Gaussian PSF settings')
        logger.info('---------------------')
    logger.info('PSF phi: {}'.format(args.PSF_phi))
    logger.info('PSF ellip: {}'.format(args.PSF_ellip))
    logger.info('PSF FWHM: {} arcsec'.format(args.PSF_FWHM))
    logger.info('PSF alpha: {}'.format(args.alpha))

    if not args.noDCR:
        logger.info('')
        logger.info('Observation settings')
        logger.info('--------------------')
        logger.info('zenith angle: {} degrees'.format(args.zenith_angle))

    if args.slow:
        galtool = chroma.SersicTool
    else:
        galtool = chroma.SersicFastTool

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
    logger.info('Galaxy x-offset: {} arcsec'.format(args.gal_x0))
    logger.info('Galaxy y-offset: {} arcsec'.format(args.gal_y0))
    logger.info('Galaxy sqrt(r2): {} arcsec'.format(args.gal_r2))

    gtool = galtool(gal_SED, bandpass, PSF, args.stamp_size, args.pixel_scale)
    gparam = gtool.set_uncvl_r2(gparam, (args.gal_r2)**2)

    # Measure shear bias
    m, c = measure_shear_calib(gparam, bandpass, gal_SED, star_SED, PSF,
                               args.pixel_scale, args.stamp_size, args.ring_n,
                               galtool, args.diagnostic, args.use_hsm)

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
        seeing1 = star_SED.getSeeingShift(bandpass, alpha=args.alpha, base_wavelength=685.0)
        seeing2 = gal_SED.getSeeingShift(bandpass, alpha=args.alpha, base_wavelength=685.0)
        dr2r2 = (seeing2 - seeing1)/seeing1
        logger.info("star seeing correction: {}".format(seeing1))
        logger.info("galaxy seeing correction: {}".format(seeing2))
    else:
        dr2r2 = 0.0

    # Third, need the second moment square radius of the PSF:
    # Ignoring corrections due to ellipticity for now.
    if args.gaussian:
        r2_psf = 2.0 * (args.PSF_FWHM/(2.0*np.sqrt(2.0*np.log(2.0))))**2
    else:
        r2_psf = args.PSF_FWHM**2 / (8.0 * (2.0**(1.0/args.PSF_beta)-1.0)*(args.PSF_beta-2.0))

    dIxx = (r2_psf/2.0) * dr2r2
    dIxy = 0.0
    dIyy = (r2_psf/2.0) * dr2r2
    dIyy += dV

    m1 = m2 = -(dIxx + dIyy) / args.gal_r2**2
    c1 = (dIxx-dIyy) / (2.0 * (args.gal_r2**2))
    c2 = dIxy / args.gal_r2**2

    logger.info('')
    logger.info('Shear Calibration Results')
    logger.info('-------------------------')
    logger.info(('        ' + ' {:>12s}'*4).format('m1','m2','c1','c2'))
    logger.info(('analytic' + ' {:12.8f}'*4).format(m1, m2, c1, c2))
    logger.info(('ring    ' + ' {:12.8f}'*4).format(m[0], m[1], c[0], c[1]))

def runme():
    """Useful for profiling one_ring_test() using IPython and prun.
    """
    class junk(object):
        pass
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
    args.thin = 10
    args.slow = False
    args.alpha = 0.0
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
    parser.add_argument('--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('--gaussian', action='store_true',
                        help="Use Gaussian PSF (Default Moffat)")
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
    parser.add_argument('--ring_n', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--thin', type=int, default=10,
                        help="Reduce the wavelengths at which Bandpass is evaluted by factor"
                        +" (Default 10).")
    parser.add_argument('--slow', action='store_true',
                        help="Use GalTool (somewhat more careful) instead of GalFastTool")
    parser.add_argument('--alpha', type=float, default=0.0,
                        help="Index to use for chromatic seeing (Default: 0.0)")
    parser.add_argument('--noDCR', action='store_true',
                        help="Implement differential chromatic refraction (DCR) in PSF? "
                        +" (Default: True)")
    parser.add_argument('--diagnostic',
                        help="Filename to which to write diagnostic images (Default: '')")
    parser.add_argument('--use_hsm', action='store_true',
                        help="Use HSM regaussianization to estimate ellipticity")

    args = parser.parse_args()

    one_ring_test(args)
