import lmfit
import numpy as np
try:
    import astropy.io.fits as fits
except:
    import pyfits as fits
import galsim

import chroma

class EllipMeasurer(object):
    """ Abstract base class for ellipticity measurer, which measures the ellipticities of ring
    test target images, and optionally writes best-fit images to a FITS hdulist.

    @param galtool     Used to draw PSF, and possibly candidate galaxy fit images.
    @param hdulist     Optionally write PSF images and best-fit images here.
    @param oversample  If writing to an hdulist, use this oversampling factor.
    """
    def __init__(self, galtool, hdulist=None, oversample=4):
        self.galtool = galtool
        self.hdulist = hdulist
        self.oversample = oversample

    def __call__(self):
        raise NotImplementedError("EllipMeasurer needs to be subclassed.")

class LSTSQEllipMeasurer(EllipMeasurer):
    """ Measure ellipticity by performing a least-squares fit over galaxy parameters.
    """
    def resid(self, param, target_image):
        """ Return least-squares residuals of image generated from `param` and `target_image`.

        @param  param         lmfit.Parameters object describing candidate galaxy fit.
        @param  target_image  Image to match via least-squares
        @returns    Flattened residuals for lmfit.minimize()
        """
        image = self.galtool.get_image(param)

        # # sanity check
        # lmfit.report_errors(param)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax1 = fig.add_subplot(311)
        # ax1.imshow(target_image.array)
        # ax2 = fig.add_subplot(312)
        # ax2.imshow(image.array)
        # ax3 = fig.add_subplot(313)
        # ax3.imshow((target_image-image).array)
        # print(np.min(target_image.array), np.max(target_image.array))
        # print(np.min(image.array), np.max(image.array))
        # plt.show()

        return (image.array - target_image.array).flatten()

    def __call__(self, target_image, init_param):
        """ Return estimated ellipticity of `target_image`

        @param target_image  Image to match via least-squares
        @param init_param    Initial guess for best-fit galaxy parameters.
        @returns   Ellipticity estimate.
        """
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
        if 'g' in result.params:
            g = result.params['g'].value
            phi = result.params['phi'].value
            return g * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))
        elif 'g_1' in result.params:
            g = result.params['g_1'].value
            phi = result.params['phi_1'].value
            return g * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))


class HSMEllipMeasurer(EllipMeasurer):
    """ Use the Hirata-Seljak-Mandelbaum regaussianization PSF correction algorithm to estimate
    ellipticity.
    """
    def psf_image(self):
        """ Use self.galtool to lazily return an image of the PSF.
        @returns PSF image
        """
        if not hasattr(self, '_psf_image'):
            self._psf_image = self.galtool.get_PSF_image()
        return self._psf_image

    def __call__(self, target_image, init_param=None):
        """ Return estimated ellipticity of `target_image`

        @param target_image  Image to match via least-squares
        @param init_param    Initial guess for best-fit galaxy parameters.
        @returns   Ellipticity estimate.
        """
        psf_image = self.psf_image()
        results = galsim.hsm.EstimateShear(target_image, psf_image)
        ellip = galsim.Shear(e1=results.corrected_e1, e2=results.corrected_e2)
        return complex(ellip.g1, ellip.g2)
