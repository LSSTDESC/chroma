""" Make a toy model of a chromatic seeing PSF correction, in which the effective stellar PSF
is dilated to match the second moment square radius of the effective galactic PSF.

In the toy, the galaxy and stellar spectra are bichromatic, and thus the PSFs are both mixtures
of two Gaussians with different widths.

Command line options are available to change the parameters of the Gaussian mixtures.
"""

from argparse import ArgumentParser

import lmfit
import galsim
import numpy as np
try:
    import astropy.io.fits as fits
except:
    import pyfits as fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _mypath
import chroma

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
                                                           oversample=self.oversample)
            self.hdulist.append(fits.ImageHDU(fit_image_uncvl.array, name='FITUC'))
        g = result.params['g'].value
        phi = result.params['phi'].value
        return g * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))


def measure_shear_calib(gparam, star_PSF, gal_PSF, pixel_scale, stamp_size, ring_n, galtool):
    """Perform two ring tests to solve for shear calibration parameters `m` and `c`."""

    target_tool = galtool(gal_PSF, stamp_size, pixel_scale)
    fit_tool = galtool(star_PSF, stamp_size, pixel_scale)

    gen_target_image = TargetImageGenerator(gparam, target_tool)
    measure_ellip = LSTSQEllipMeasurer(fit_tool)

    def get_ring_params(gamma, beta):
        return fit_tool.get_ring_params(gparam, beta, galsim.Shear(g1=gamma.real, g2=gamma.imag))

    # Ring test for two values of gamma, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.ringtest(gamma0, ring_n, gen_target_image, get_ring_params, measure_ellip,
                                 silent=True)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.02 + 0.01j
    gamma1_hat = chroma.ringtest(gamma1, ring_n, gen_target_image, get_ring_params, measure_ellip,
                                 silent=True)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1

    return m, c

def toy_PSF_correction(args):
    # build effective PSFs for star and galaxy:
    PSF1 = galsim.Gaussian(sigma=args.s1)
    PSF2 = galsim.Gaussian(sigma=args.s2)
    star_PSF = args.starfrac1*PSF1 + (1.0 - args.starfrac1)*PSF2
    gal_PSF = args.galfrac1*PSF1 + (1.0 - args.galfrac1)*PSF2

    # next compute factor by which to dilate star PSF to have same second moments as gal PSF
    # r2 is the second moment square radius
    star_r2 = args.s1**2 * args.starfrac1 + args.s2**2 * (1.0 - args.starfrac1)
    gal_r2 = args.s1**2 * args.galfrac1 + args.s2**2 * (1.0 - args.galfrac1)
    scale_factor = (gal_r2 / star_r2)**0.5
    fit_PSF = star_PSF.createDilated(scale_factor)

    gparam = chroma.SersicTool.default_galaxy()

    # need a plot with 1d PSF profiles
    fig = plt.figure(figsize=(6,4))
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel("PSF")
    ax1.set_xlabel("r")
    x = np.arange(-0.5,0.5,0.01)
    star_PSF_y = np.array([star_PSF.xValue((0.0, xi)) for xi in x])
    gal_PSF_y = np.array([gal_PSF.xValue((0.0, xi)) for xi in x])
    fit_PSF_y = np.array([fit_PSF.xValue((0.0, xi)) for xi in x])

    ax1.plot(x, star_PSF_y, label='star')
    ax1.plot(x, gal_PSF_y, label='gal')
    ax1.plot(x, fit_PSF_y, label='galmatch')
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.set_ylabel(r"r $\times$ PSF(r)")
    ax2.set_xlabel("r")
    x = np.arange(0.0,0.5,0.01)
    star_PSF_y = np.array([star_PSF.xValue((0.0, xi))*xi for xi in x])
    gal_PSF_y = np.array([gal_PSF.xValue((0.0, xi))*xi for xi in x])
    fit_PSF_y = np.array([fit_PSF.xValue((0.0, xi))*xi for xi in x])

    ax2.plot(x, star_PSF_y, label='star')
    ax2.plot(x, gal_PSF_y, label='gal')
    ax2.plot(x, fit_PSF_y, label='galmatch')
    ax2.legend()

    fig.tight_layout()
    fig.savefig('output/toy.png', dpi=220)

    m, c = measure_shear_calib(gparam, star_PSF, gal_PSF, 0.2, 64, 18, chroma.MonoSersicTool)
    print (' '*16+' {:9}'*4).format('m1', 'm2', 'c1', 'c2')
    print (' uncorrected: '+' {:9.5f}'*4).format(m[0], m[1], c[0], c[1])
    m, c = measure_shear_calib(gparam, fit_PSF, gal_PSF, 0.2, 64, 18, chroma.MonoSersicTool)
    print ('   corrected: '+' {:9.5f}'*4).format(m[0], m[1], c[0], c[1])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s1', default=0.2, type=float, help="PSF1 sigma (Default 0.2)")
    parser.add_argument('-s2', default=0.3, type=float, help="PSF2 sigma (Default 0.3)")
    parser.add_argument('-starfrac1', default=0.5, type=float,
                        help="star PSF1 fraction (Default 0.5)")
    parser.add_argument('-galfrac1', default=0.7, type=float,
                        help="gal PSF1 fraction (Default 0.7)")
    args = parser.parse_args()
    toy_PSF_correction(args)
