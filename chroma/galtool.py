"""
Some tools to conveniently transform lmfit.Parameters into GalSim.Image's.
"""

import copy

import numpy as np
from scipy.optimize import newton
import galsim
import lmfit

import chroma

class GalTool(object):
    """ Some generic utilities for drawing ringtest images using GalSim and measuring second moment
    radii.
    """
    def __init__(self):
        # Subclasses of GalTool must initialize the following:
        #
        #   attributes
        #   ----------
        #   stamp_size - Integer number of pixels in which to draw images
        #   pixel_scale - arcsec / pixel
        #   PSF - either a ChromaticObject or an effective PSF as a GSObject.
        #   offset - tuple defining subpixel offset of image origin from center
        #   gsparams - galsim.GSParams instance defining parameters for GalSim.
        #
        #   methods
        #   -------
        #   _gparam_to_galsim - turn lmfit.Parameters into a galsim.GSObject or
        #                       galsim.ChromaticObject
        raise NotImplementedError("ABC GalTool must be instatiated through a subclass.")

    def get_image(self, gparam, ring_beta=None, ring_shear=None, oversample=1):
        """ Draw a galaxy image using GalSim.  Potentially rotate and shear the galaxy as part of a
        ring test.  Optionally draw a high-resolution image.

        @param gparam      An lmfit.Paramters object that will be used to initialize a GalSim object.
        @param ring_beta   Angle around ellipticity ring in ring test.
        @param ring_shear  Shear to apply after rotation as part of ring test. (type=?)
        @param oversample  Integer factor by which to scale output image resolution and size.
        @returns  galsim.Image
        """
        stamp_size = self.stamp_size * oversample
        pixel_scale = self.pixel_scale / float(oversample)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        gal = self._gparam_to_galsim(gparam)
        pix = galsim.Pixel(pixel_scale)
        if ring_beta is not None:
            gal = gal.rotate(ring_beta / 2.0 * galsim.radians)
        if ring_shear is not None:
            gal = gal.shear(ring_shear)
        final = galsim.Convolve(gal, self.PSF, pix)
        if isinstance(final, galsim.ChromaticObject):
            final.draw(self.bandpass, image=im, offset=self.offset)
        elif isinstance(final, galsim.GSObject):
            final.draw(image=im, offset=self.offset)
        else:
            raise ValueError("Don't recognize galaxy object type in GalTool.")
        return im

    def get_PSF_image(self, oversample=1):
        """ Draw an image of the effective PSF.  Note that we choose to include convolution by the
        pixel response function here.

        @param oversample  Integer factor by which to scale output image resolution and size.
        @returns  galsim.Image
        """
        stamp_size = self.stamp_size * oversample
        pixel_scale = self.pixel_scale / float(oversample)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        pix = galsim.Pixel(pixel_scale)
        if isinstance(self.PSF, galsim.ChromaticObject):
            star = galsim.Gaussian(fwhm=1.e-8) * self.SED
            final = galsim.Convolve(star, self.PSF, pix)
            final.draw(self.bandpass, image=im)
        elif isinstance(self.PSF, galsim.GSObject):
            final = galsim.Convolve(self.PSF, pix)
            final.draw(image=im)
        else:
            raise ValueError("Don't recognize galaxy object type.")
        return im

    def get_r2(self, gparam, oversample=1):
        """ Compute object second moment radius sqrt(r^2) directly from image.  This may be biased
        if the object wings are significant or the postage stamp size is too small.

        @param gparam   An lmfit.Parameters object that will be used to initialize a GalSim object.
        @returns        Second moment radius (in arcsec)
        """
        im = self.get_image(gparam, oversample=oversample)
        mx, my, mxx, myy, mxy = chroma.moments(im)
        return np.sqrt(mxx + myy)

    def get_uncvl_image(self, gparam, ring_beta=None, ring_shear=None, oversample=1, center=False):
        """ Draw a galaxy image, not convolved with a PSF, using GalSim.  Potentially rotate and
        shear the galaxy as part of a ring test.  Optionally draw a high-resolution image.

        @param gparam      An lmfit.Paramters object that will be used to initialize a GalSim object.
        @param ring_beta   Angle around ellipticity ring in ring test.
        @param ring_shear  Shear to apply after rotation as part of ring test. (type=?)
        @param oversample  Integer factor by which to scale output image resolution and size.
        @param center      Force center of profile to (0,0).
        @returns  galsim.Image
        """
        stamp_size = self.stamp_size * oversample
        pixel_scale = self.pixel_scale / float(oversample)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        gal = self._gparam_to_galsim(gparam)
        if center:
            centroid = gal.centroid()
            gal = gal.shift(-centroid)
        pix = galsim.Pixel(pixel_scale)
        if ring_beta is not None:
            gal = gal.rotate(ring_beta / 2.0 * galsim.radians)
        if ring_shear is not None:
            gal = gal.shear(ring_shear)
        final = galsim.Convolve(gal, pix)
        if isinstance(final, galsim.ChromaticObject):
            final.draw(self.bandpass, image=im, offset=self.offset)
        elif isinstance(final, galsim.GSObject):
            final.draw(image=im, offset=self.offset)
        else:
            raise ValueError("Don't recognize galaxy object type in GalTool.")
        return im

    def get_uncvl_r2(self, gparam, oversample=1):
        """ Compute object second moment radius directly from image.  This may be biased if the
        object wings are significant or the postage stamp size is too small.

        @param gparam   An lmfit.Parameters object that will be used to initialize a GalSim object.
        @returns        Second moment radius (in arcsec)
        """
        im = self.get_uncvl_image(gparam, oversample=oversample)
        mx, my, mxx, myy, mxy = chroma.moments(im)
        return np.sqrt(mxx + myy)

    def compute_AHM(self, gparam, oversample=4):
        """ Compute the area above half maximum of the convolved image.
        """
        original_offset = self.offset
        original_scale = self.pixel_scale
        ahms = []
        for i in range(10):
            itry = 0
            while itry < 10:
                xdither = np.random.uniform(-0.5, 0.5, 1)[0]
                ydither = np.random.uniform(-0.5, 0.5, 1)[0]
                rescale = np.random.uniform(0.9, 1.1, 1)[0]
                self.offset = (xdither, ydither)
                self.pixel_scale = original_scale * rescale
                try:
                    im = self.get_image(gparam, oversample=oversample)
                except RuntimeError:
                    itry += 1
                else:
                    break
            if itry >= 10:
                raise RuntimeError("Unable to create image to estimate AHM")
            mx = im.array.max()
            ahms.append(self.pixel_scale**2 * (im.array > mx/2.0).sum() / oversample**2)
        self.offset = original_offset
        self.pixel_scale = original_scale
        return np.mean(ahms), np.std(ahms)/np.sqrt(len(ahms))

    def compute_FWHM(self, gparam, oversample=4):
        """ Compute FWHM of the convolved galaxy image.
        """
        ahm, err = self.compute_AHM(gparam, oversample=oversample)
        fwhm = np.sqrt(4.0/np.pi * ahm)
        return fwhm, fwhm * err/ahm * 0.5

    def compute_HLA(self, gparam, oversample=4, flux=None):
        """ Compute the half-light-area of the PSF-convolved galaxy image.
        I.e., the area of the contour containing half the image light.
        """
        im = self.get_image(gparam, oversample=oversample)
        if flux is None:
            flux = im.array.sum()
        pixel_values = im.array.ravel()
        pixel_values.sort()
        cumulative_sum = np.cumsum(pixel_values[::-1])
        npix = np.interp(0.5, cumulative_sum, np.arange(len(cumulative_sum)))
        return npix * self.pixel_scale**2 / oversample**2

    def compute_HLR(self, gparam, oversample=4, flux=None):
        """ Compute the half-light-radius of the PSF-convolved galaxy image.
        """
        return np.sqrt(1.0/np.pi * self.compute_HLA(gparam, oversample, flux))

    def compute_uncvl_HLA(self, gparam, oversample=4, flux=None):
        """ Compute the half-light-area of the unconvolved galaxy image.
        I.e., the area of the contour containing half the image light.
        """
        im = self.get_uncvl_image(gparam, oversample=oversample)
        if flux is None:
            flux = im.array.sum()
        pixel_values = im.array.ravel()
        pixel_values.sort()
        cumulative_sum = np.cumsum(pixel_values[::-1])
        npix = np.interp(0.5, cumulative_sum, np.arange(len(cumulative_sum)))
        return npix * self.pixel_scale**2 / oversample**2

    def compute_uncvl_HLR(self, gparam, oversample=4, flux=None):
        """ Compute the half-light-radius of the unconvolved galaxy image.
        """
        return np.sqrt(1.0/np.pi * self.compute_uncvl_HLA(gparam, oversample, flux))


class SersicTool(GalTool):
    def __init__(self, PSF, stamp_size, pixel_scale, offset=(0,0),
                 SED=None, bandpass=None, gsparams=None):
        self.PSF = PSF
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.offset = offset
        self.gsparams = gsparams
        self.SED = SED
        self.bandpass = bandpass

    def _gparam_to_galsim(self, gparam):
        # Turn lmfit.Parameters into a galsim.ChromaticObject
        gal = galsim.Sersic(n=gparam['n'].value,
                            half_light_radius=gparam['hlr'].value,
                            gsparams=self.gsparams)
        gal = gal.shear(g=gparam['g'].value, beta=gparam['phi'].value * galsim.radians)
        gal = gal.shift(gparam['x0'].value, gparam['y0'].value)
        gal = gal.withFlux(gparam['flux'].value)
        return gal

    def set_FWHM(self, gparam, FWHM, oversample=4):
        """ Set the galaxy PSF-convolved FWHM.
        """
        def FWHM_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr'].value *= scale
            current_FWHM = self.compute_FWHM(g1, oversample=oversample)
            return current_FWHM[0] - FWHM
        scale = newton(FWHM_resid, 1.0, tol=0.001)
        gparam['hlr'].value *= scale
        return gparam

    def set_r2(self, gparam, r2, oversample=4):
        """ Set the second moment radius sqrt(r^2).

        @param gparam      lmfit.Parameters object describing galaxy.
        @param r2          Target second moment radius sqrt(r^2)
        @param oversample  Factor by which to oversample drawn image for computation.
        @returns           New lmfit.Parameters object.
        """
        def r2_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr'].value *= scale
            current_r2 = self.get_r2(g1, oversample=oversample)
            return current_r2 - r2
        scale = newton(r2_resid, 1.0)
        gparam['hlr'].value *= scale
        return gparam

    def get_uncvl_r2(self, gparam):
        """ Get second moment radius sqrt(r^2) of pre-PSF-convolved profile using polynomial
        approximation.
        @gparam   lmfit.Parameters
        """
        return gparam['hlr'].value * chroma.Sersic_r2_over_hlr(gparam['n'].value)

    def set_uncvl_r2(self, gparam, r2):
        """ Set the second moment radius sqrt(r^2) of the pre-PSF-convolved profile using
        polynomial approximation.

        @param gparam      lmfit.Parameters object describing galaxy.
        @param r2          Target second moment square radius
        @param oversample  Factor by which to oversample drawn image for r2 computation.
        @returns           New lmfit.Parameters object.
        """
        gparam1 = copy.deepcopy(gparam)
        r2_now = self.get_uncvl_r2(gparam)
        scale = r2 / r2_now
        gparam1['hlr'].value = gparam['hlr'].value * scale
        return gparam1

    def get_ring_params(self, gparam, ring_beta, ring_shear):
        """ Compute initial guess parameters for given angle around ellipticity ring during a ring
        test.

        @param gparam      lmfit.Parameters object describing galaxy.
        @param ring_beta   Angle around ellipticity ring in ring test.
        @param ring_shear  Shear to apply after rotation as part of ring test. (type=?)
        @returns           New lmfit.Parameters object.
        """
        gparam1 = copy.deepcopy(gparam)
        rot_phi = gparam['phi'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip = gparam['g'].value * complex(np.cos(2.0 * rot_phi), np.sin(2.0 * rot_phi))
        c_gamma = ring_shear.g1 + 1j * ring_shear.g2
        # sheared complex ellipticity
        s_c_ellip = chroma.apply_shear(c_ellip, c_gamma)
        s_g = abs(s_c_ellip)
        s_phi = np.angle(s_c_ellip) / 2.0

        gparam1['x0'].value \
          = gparam['x0'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0'].value * np.sin(ring_beta / 2.0)
        gparam1['y0'].value \
          = gparam['x0'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0'].value * np.cos(ring_beta / 2.0)
        gparam1['g'].value = s_g
        gparam1['phi'].value = s_phi
        return gparam1

    @staticmethod
    def default_galaxy():
        """Setup lmfit.Parameters to represent a single Sersic galaxy.  Pick some default
        parameter values.  Parameters defining the single Sersic galaxy are:
        x0   - the x-coordinate of the galaxy center
        y0   - the y-coordinate of the galaxy center
        n    - the Sersic index.  0.5 gives a Gaussian profile, 1.0 gives an exponential profile,
               4.0 gives a de Vaucouleurs profile.
        hlr  - the galaxy half-light-radius.  This is strictly speaking the half light radius of
               a circularly symmetric profile of the given Sersic index `n`.
        g    - the magnitude of the galaxy ellipticity given in `g` units as used by GalSim.  In
               this convention, the major/minor axis ratio is given by: b/a = (1 - g) / (1 + g)
        phi  - the position angle of the galaxy major axis in radians.  0 indicates that the major
               axis is along the x-axis.
        """
        gparam = lmfit.Parameters()
        gparam.add('x0', value=0.0)
        gparam.add('y0', value=0.0)
        gparam.add('n', value=4.0, vary=False)
        gparam.add('hlr', value=0.27)
        gparam.add('flux', value=1.0, vary=False)
        gparam.add('g', value=0.2, min=0.0, max=1.0)
        gparam.add('phi', value=0.0)
        return gparam

    def use_effective_PSF(self):
        star = galsim.Gaussian(fwhm=1.e-8) * self.SED
        prof = galsim.Convolve(star, self.PSF)
        prof0 = prof.evaluateAtWavelength(self.bandpass.effective_wavelength)
        scale = prof0.nyquistDx()
        N = prof0.SBProfile.getGoodImageSize(scale,1.0)
        im = galsim.ImageD(N, N, scale=scale)
        prof.draw(self.bandpass, image=im)
        self.PSF = galsim.InterpolatedImage(im) # remember the effective PSF

    def apply_perturbative_correction(self, r2byr2=1.0, Vstar=1.e-8, Vgal=1.e-8,
                                      parang=0.0*galsim.degrees):
        if isinstance(self.PSF, galsim.ChromaticObject):
            star = galsim.Gaussian(fwhm=1.e-8) * self.SED
            prof = galsim.Convolve(star, self.PSF)
        elif isinstance(self.PSF, galsim.GSObject):
            prof = self.PSF
        else:
            raise ValueError("Don't recognize galaxy object type.")

        #-----------------------
        # Stellar DCR correction

        # `q` is the axis ratio of a 2D Gaussian representing the 1D DCR kernel. In principle, this
        # should be 0.0, but we need to set it to some small value for computability.
        q = 1.e-4
        sigma = (q * Vstar)**0.5
        kernel = galsim.Gaussian(sigma=sigma)
        kernel = kernel.shear(g1=-(1-q)/(1+q))
        kernel = kernel.rotate(parang)
        prof = galsim.Convolve(galsim.Deconvolve(kernel), prof)

        #----------------------------
        # Chromatic Seeing correction
        prof = prof.dilate(np.sqrt(r2byr2))

        #------------------------
        # Galactic DCR correction
        sigma = (q * Vgal)**0.5
        kernel = galsim.Gaussian(sigma=sigma)
        kernel = kernel.shear(g1=-(1-q)/(1+q))
        kernel = kernel.rotate(parang)
        prof = galsim.Convolve(kernel, prof)

        # and draw into an InterpolatedImage
        prof0 = prof.evaluateAtWavelength(self.bandpass.effective_wavelength)
        scale = prof0.nyquistDx()
        N = prof0.SBProfile.getGoodImageSize(scale, 1.0)
        im = galsim.ImageD(N*9, N*9, scale=scale*0.3)
        if isinstance(prof, galsim.ChromaticObject):
            prof.draw(self.bandpass, image=im)
        else:
            prof.draw(image=im)
        self.PSF = galsim.InterpolatedImage(im)


# Note that DoubleSersicTool and FastDoubleSersicTool are both currently untested.
class DoubleSersicTool(GalTool):
    """ A GalTool to represent a sum of two chroma Sersic profiles.
    """
    def __init__(self, SED1, SED2, bandpass, PSF, stamp_size, pixel_scale, offset=(0,0),
                 gsparams=None):
        """ Initialize a single Sersic profile chromatic galaxy.

        @param SED1         galsim.SED galaxy spectrum for first component
        @param SED2         galsim.SED galaxy spectrum for second component
        @param bandpass     galsim.Bandpass to represent filter being imaged through.
        @param PSF          galsim.ChromaticObject representing chromatic PSF
        @param stamp_size   Draw images this many pixels square
        @param pixel_scale  Pixels are this wide in arcsec.
        """
        self.SED1 = SED1
        self.SED2 = SED2
        self.bandpass = bandpass
        self.PSF = PSF
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.offset = offset
        self.gsparams = gsparams

    def _gparam_to_galsim(self, gparam):
        # Turn lmfit.gparam into a galsim.ChromaticObject
        mono_gal1 = galsim.Sersic(n=gparam['n_1'].value,
                                  half_light_radius=gparam['hlr_1'].value,
                                  gsparams=self.gsparams)
        mono_gal1 = mono_gal1.shear(
            g=gparam['g_1'].value, beta=gparam['phi_1'].value * galsim.radians)
        mono_gal1 = mono_gal1.shift(gparam['x0_1'].value, gparam['y0_1'].value)
        mono_gal1 = mono_gal1.withFlux(gparam['flux_1'].value)

        mono_gal2 = galsim.Sersic(n=gparam['n_2'].value,
                                  half_light_radius=gparam['hlr_2'].value,
                                  gsparams=self.gsparams)
        mono_gal2 = mono_gal2.shear(
            g=gparam['g_2'].value, beta=gparam['phi_2'].value * galsim.radians)
        mono_gal2 = mono_gal2.shift(gparam['x0_2'].value, gparam['y0_2'].value)
        mono_gal2 = mono_gal2.withFlux(gparam['flux_2'].value)

        gal1 = galsim.Chromatic(mono_gal1, self.SED1)
        gal2 = galsim.Chromatic(mono_gal2, self.SED2)
        return gal1 + gal2

    def get_PSF_image(self, oversample=1):
        """ Draw an image of both effective PSFs.  Note that we choose to convolve by the pixel
        response function too here.

        @param oversample  Integer factor by which to scale output image resolution and size.
        @returns (im1, im2)  Both effective PSFs corresponding to both component SEDs.
        """
        stamp_size = self.stamp_size * oversample
        pixel_scale = self.pixel_scale / float(oversample)
        im1 = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        im2 = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        pix = galsim.Pixel(pixel_scale)
        star1 = galsim.Gaussian(fwhm=1.e-8) * self.SED1
        final1 = galsim.Convolve(star1, self.PSF, pix)
        final1.draw(self.bandpass, image=im1)
        star2 = galsim.Gaussian(fwhm=1.e-8) * self.SED2
        final2 = galsim.Convolve(star2, self.PSF, pix)
        final2.draw(self.bandpass, image=im2)
        return im1, im2

    def set_FWHM(self, gparam, FWHM, oversample=4):
        """ Set the galaxy PSF-convolved FWHM.
        """
        def FWHM_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr_1'].value *= scale
            g1['hlr_2'].value *= scale
            current_FWHM = self.compute_FWHM(g1, oversample=oversample)
            return current_FWHM[0] - FWHM
        scale = newton(FWHM_resid, 1.0, tol=0.01)
        gparam['hlr_1'].value *= scale
        gparam['hlr_2'].value *= scale
        return gparam

    def set_r2(self, gparam, r2, oversample=4):
        """ Set the second moment radius sqrt(r^2).

        @param gparam      lmfit.Parameters object describing galaxy.
        @param r2          Target second moment radius sqrt(r^2)
        @param oversample  Factor by which to oversample drawn image for r2 computation.
        @returns           New lmfit.Parameters object.
        """
        def r2_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr_1'].value *= scale
            g1['hlr_2'].value *= scale
            current_r2 = self.get_r2(g1, oversample=oversample)
            return current_r2 - r2
        scale = newton(r2_resid, 1.0)
        gparam['hlr_1'].value *= scale
        gparam['hlr_2'].value *= scale
        return gparam

    def get_uncvl_r2(self, gparam):
        """ Get second moment radius of pre-PSF-convolved profile using polynomial approximation.
        @gparam   lmfit.Parameters
        """
        return chroma.component_Sersic_r2([gparam['n_1'].value, gparam['n_2'].value],
                                          [gparam['flux_1'].value, gparam['flux_2'].value],
                                          [gparam['hlr_1'].value, gparam['hlr_2'].value])

    def set_uncvl_r2(self, gparam, r2):
        """ Set the second moment square radius of the pre-PSF-convolved profile using polynomial
        approximation.

        @param gparam      lmfit.Parameters object describing galaxy.
        @param r2          Target second moment radius sqrt(r^2)
        @param oversample  Factor by which to oversample drawn image for computation.
        @returns           New lmfit.Parameters object.
        """
        gparam1 = copy.deepcopy(gparam)
        r2_now = self.get_uncvl_r2(gparam)
        scale = np.sqrt(r2 / r2_now)
        gparam1['hlr_1'].value *= scale
        gparam1['hlr_2'].value *= scale
        return gparam1

    def get_ring_params(self, gparam, ring_beta, ring_shear):
        """ Compute initial guess parameters for given angle around ellipticity ring during a ring
        test.

        @param gparam      lmfit.Parameters object describing galaxy.
        @param ring_beta   Angle around ellipticity ring in ring test.
        @param ring_shear  Shear to apply after rotation as part of ring test. (type=?)
        @returns           New lmfit.Parameters object.
        """
        gparam1 = copy.deepcopy(gparam)

        rot_phi1 = gparam['phi_1'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip1 = gparam['g_1'].value * complex(np.cos(2.0 * rot_phi1), np.sin(2.0 * rot_phi1))
        c_gamma1 = ring_shear.g1 + 1j * ring_shear.g2
        # sheared complex ellipticity
        s_c_ellip1 = chroma.apply_shear(c_ellip1, c_gamma)
        s_g1 = abs(s_c_ellip1)
        s_phi1 = np.angle(s_c_ellip1) / 2.0

        gparam1['x0_1'].value \
          = gparam['x0_1'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0_1'].value * np.sin(ring_beta / 2.0)
        gparam1['y0_1'].value \
          = gparam['x0_1'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0_1'].value * np.cos(ring_beta / 2.0)
        gparam1['g_1'].value = s_g1
        gparam1['phi_1'].value = s_phi1

        rot_phi2 = gparam['phi_2'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip2 = gparam['g_2'].value * \
          complex(np.cos(2.0 * rot_phi2), np.sin(2.0 * rot_phi2))
        c_gamma2 = ring_shear.g2 + 1j * ring_shear.g2
        # sheared complex ellipticity
        s_c_ellip2 = chroma.apply_shear(c_ellip2, c_gamma)
        s_g2 = abs(s_c_ellip2)
        s_phi2 = np.angle(s_c_ellip2) / 2.0

        gparam1['x0_2'].value \
          = gparam['x0_2'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0_2'].value * np.sin(ring_beta / 2.0)
        gparam1['y0_2'].value \
          = gparam['x0_2'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0_2'].value * np.cos(ring_beta / 2.0)
        gparam1['g_2'].value = s_g2
        gparam1['phi_2'].value = s_phi2

        return gparam1


class FastDoubleSersicTool(DoubleSersicTool):
    def __init__(self, SED1, SED2, bandpass, PSF, stamp_size, pixel_scale, offset=(0,0),
                 gsparams=None):
        """ Initialize a 2 component chromatic Sersic profile.  Internally use some trickery to
        speed up image drawing by cacheing two effective PSFs.

        @param SED1         galsim.SED galaxy spectrum for first component
        @param SED2         galsim.SED galaxy spectrum for second component
        @param bandpass     galsim.Bandpass to represent filter being imaged through.
        @param PSF          galsim.ChromaticObject representing chromatic PSF
        @param stamp_size   Draw images this many pixels square
        @param pixel_scale  Pixels are this wide in arcsec.
        """
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        # now create the effective PSFs
        star1 = galsim.Gaussian(fwhm=1.e-8) * SED1
        prof1 = galsim.Convolve(star1, PSF)
        prof10 = prof1.evaluateAtWavelength(bandpass.effective_wavelength)
        scale = prof10.nyquistDx()
        N = prof10.SBProfile.getGoodImageSize(scale,1.0)
        im = galsim.ImageD(N, N, scale=scale)
        prof1.draw(bandpass, image=im)
        self.PSF1 = galsim.InterpolatedImage(im) # remember the effective PSF

        star2 = galsim.Gaussian(fwhm=1.e-8) * SED2
        prof2 = galsim.Convolve(star2, PSF)
        prof20 = prof2.evaluateAtWavelength(bandpass.effective_wavelength)
        scale = prof20.nyquistDx()
        N = prof10.SBProfile.getGoodImageSize(scale,1.0)
        im = galsim.ImageD(N, N, scale=scale)
        prof2.draw(bandpass, image=im)
        self.PSF2 = galsim.InterpolatedImage(im2) # remember the effective PSF

        self.offset = offset
        self.gsparams = gsparams

    def _gparam_to_galsim(self, gparam):
        # Turn lmfit.gparam into a galsim.ChromaticObject
        mono_gal1 = galsim.Sersic(n=gparam['n_1'].value,
                                  half_light_radius=gparam['hlr_1'].value,
                                  gsparams=self.gsparams)
        mono_gal1 = mono_gal1.shear(
            g=gparam['g_1'].value, beta=gparam['phi_1'].value * galsim.radians)
        mono_gal1 = mono_gal1.shift(gparam['x0_1'].value, gparam['y0_1'].value)
        mono_gal1 = mono_gal1.withFlux(gparam['flux_1'].value)

        mono_gal2 = galsim.Sersic(n=gparam['n_2'].value,
                                  half_light_radius=gparam['hlr_2'].value,
                                  gsparams=self.gsparams)
        mono_gal2 = mono_gal2.shear(
            g=gparam['g_2'].value, beta=gparam['phi_2'].value * galsim.radians)
        mono_gal2 = mono_gal2.shift(gparam['x0_2'].value, gparam['y0_2'].value)
        mono_gal2 = mono_gal2.withFlux(gparam['flux_2'].value)

        return gal1, gal2

    def get_image(self, param, ring_beta=None, ring_shear=None, oversample=1):
        """ Draw a galaxy image using GalSim.  Potentially rotate and shear the galaxy as part of a
        ring test.  Optionally draw a high-resolution image.

        @param gparam      An lmfit.Paramters object that will be used to initialize a GalSim object.
        @param ring_beta   Angle around ellipticity ring in ring test.
        @param ring_shear  Shear to apply after rotation as part of ring test. (type=?)
        @param oversample  Integer factor by which to scale output image resolution and size.
        @returns  galsim.Image
        """
        stamp_size = self.stamp_size * oversample
        pixel_scale = self.pixel_scale / float(oversample)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        gal1, gal2 = self._gparam_to_galsim(gparam)
        pix = galsim.Pixel(pixel_scale)
        if ring_beta is not None:
            gal1 = gal1.rotate(ring_beta / 2.0 * galsim.radians)
            gal2 = gal2.rotate(ring_beta / 2.0 * galsim.radians)
        if ring_shear is not None:
            gal1 = gal1.shear(ring_shear)
            gal2 = gal2.shear(ring_shear)
        final1 = galsim.Convolve(gal1, self.PSF1, pix)
        final2 = galsim.Convolve(gal2, self.PSF2, pix)
        final1.draw(image=im, offset=self.offset)
        final2.draw(image=im, add_to_image=True, offset=self.offset)
        return im

    def get_PSF_image(self, oversample=1):
        """ Draw an image of both effective PSFs.  Note that we choose to convolve by the pixel
        response function too here.

        @param oversample  Integer factor by which to scale output image resolution and size.
        @returns (im1, im2)  Both effective PSFs corresponding to both component SEDs.
        """
        stamp_size = self.stamp_size * oversample
        pixel_scale = self.pixel_scale / float(oversample)
        im1 = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        im2 = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        pix = galsim.Pixel(pixel_scale)
        final1 = galsim.Convolve(self.PSF1, pix)
        final1.draw(self.bandpass, image=im1)
        final2 = galsim.Convolve(self.PSF2, pix)
        final2.draw(self.bandpass, image=im2)
        return im1, im2

    def get_uncvl_image(self, gparam, ring_beta=None, ring_shear=None, oversample=1, center=False):
        return NotImplementedError("Unconvolved image impossible for inseparable chromatic profile.")
