import copy

import numpy as np
from scipy.optimize import newton

import galsim

def getmoments(im):
    xgrid, ygrid = np.meshgrid(np.arange(im.array.shape[1])*im.scale + im.getXMin(),
                               np.arange(im.array.shape[0])*im.scale + im.getYMin())
    mx = np.sum(xgrid * im.array) / np.sum(im.array)
    my = np.sum(ygrid * im.array) / np.sum(im.array)
    mxx = np.sum(((xgrid-mx)**2) * im.array) / np.sum(im.array)
    myy = np.sum(((ygrid-my)**2) * im.array) / np.sum(im.array)
    mxy = np.sum((xgrid-mx) * (ygrid-my) * im.array) / np.sum(im.array)
    return mx, my, mxx, myy, mxy

def shear_galaxy(c_ellip, c_gamma):
    '''Compute complex ellipticity after shearing by complex shear `c_gamma`.'''
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)

def Sersic_r_2nd_moment_over_r_e(n):
    ''' Factor to convert the half light radius r_e to the 2nd moment radius defined
    as sqrt(Ixx + Iyy) where Ixx and Iyy are the second central moments of a distribution
    in the perpendicular directions.  Depends on the Sersic index n.  The polynomial
    below is derived from a Mathematica fit to the exact relation, and should be good to
    ~(0.01 - 0.04)% over than range 0.2 < n < 8.0.
    '''
    return 0.98544 + n * (0.391015 + n * (0.0739614 + n * (0.00698666 + n * (0.00212443 + \
                     n * (-0.000154064 + n * 0.0000219636)))))

class GalTool(object):
    def get_image(self, gparam, ring_beta=None, ring_shear=None, oversample_factor=1,
                  N=100):
        stamp_size = self.stamp_size * oversample_factor
        pixel_scale = self.pixel_scale / float(oversample_factor)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        gal = self._gparam_to_galsim(gparam)
        pix = galsim.Pixel(pixel_scale)
        if ring_beta is not None:
            gal.applyRotation(ring_beta / 2.0 * galsim.radians)
        if ring_shear is not None:
            gal.applyShear(ring_shear)
        final = galsim.Convolve(gal, self.PSF, pix)
        final.draw(self.bandpass, image=im, N=N)
        return im

    def get_image2(self, gparam, ring_beta=0.0, ring_shear=None, oversample_factor=1,
                   N=100):
        stamp_size = self.stamp_size * oversample_factor
        pixel_scale = self.pixel_scale / float(oversample_factor)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)

        # explicitly reset gparam
        gparam1 = copy.deepcopy(gparam)
        phi_ring = gparam['phi'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip = gparam['gmag'].value * \
          complex(np.cos(2.0 * phi_ring), np.sin(2.0 * phi_ring))
        # sheared complex ellipticity
        if ring_shear is None:
            gamma = 0.0 + 0.0j
        else:
            gamma = ring_shear.g1 + 1.0j * ring_shear.g2
        s_c_ellip = shear_galaxy(c_ellip, gamma)
        s_gmag = abs(s_c_ellip)
        s_phi = np.angle(s_c_ellip) / 2.0
        # radius rescaling
        # rescale = np.sqrt(1.0 - abs(gamma)**2.0)
        rescale = 1.0

        # rotate center point
        gparam1['x0'].value \
          = gparam['x0'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0'].value * np.sin(ring_beta / 2.0)
        gparam1['y0'].value \
          = gparam['x0'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0'].value * np.cos(ring_beta / 2.0)
        gparam1['gmag'].value = s_gmag
        gparam1['phi'].value = s_phi
        gparam1['hlr'].value = gparam['hlr'].value * rescale

        # now continue as before
        gal = self._gparam_to_galsim(gparam1)
        pix = galsim.Pixel(pixel_scale)
        final = galsim.Convolve(gal, self.PSF, pix)
        final.draw(self.bandpass, image=im, N=N)
        return im

    def get_r2(self, gparam, oversample_factor=1, N=100):
        im = self.get_image(gparam, oversample_factor=oversample_factor, N=N)
        mx, my, mxx, myy, mxy = getmoments(im)
        return mxx + myy

    def get_uncvl_image(self, gparam, ring_beta=None, ring_shear=None, oversample_factor=1,
                        N=100):
        stamp_size = self.stamp_size * oversample_factor
        pixel_scale = self.pixel_scale / float(oversample_factor)
        im = galsim.ImageD(stamp_size, stamp_size, scale=pixel_scale)
        gal = self._gparam_to_galsim(gparam)
        pix = galsim.Pixel(pixel_scale)
        if ring_beta is not None:
            gal.applyRotation(ring_phi / 2.0 * galsim.radians)
        if ring_shear is not None:
            gal.applyShear(ring_shear)
        final = galsim.Convolve(gal, pix)
        final.draw(self.bandpass, image=im, N=N)
        return im

    def get_uncvl_r2(self, gparam, oversample_factor=1):
        im = self.get_uncvl_image(gparam, oversample_factor=oversample_factor)
        mx, my, mxx, myy, mxy = getmoments(im)
        return mxx + myy


class SersicTool(GalTool):
    def __init__(self, SED, bandpass, PSF, stamp_size, pixel_scale):
        self.SED = SED
        self.bandpass = bandpass
        self.PSF = PSF
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale

    def _gparam_to_galsim(self, gparam):
        mono_gal = galsim.Sersic(n=gparam['n'].value,
                                 half_light_radius=gparam['hlr'].value)
        mono_gal.applyShift(gparam['x0'].value, gparam['y0'].value)
        mono_gal.applyShear(g=gparam['gmag'].value, beta=gparam['phi'].value * galsim.radians)
        mono_gal.setFlux(gparam['flux'].value)
        gal = galsim.Chromatic(mono_gal, self.SED)
        return gal

    def set_r2(self, gparam, target_r2, oversample_factor=16):
        def r2_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr'].value *= scale
            current_r2 = self.get_r2(g1, oversample_factor=oversample_factor)
            return current_r2 - target_r2
        scale = newton(r2_resid, 1.0)
        gparam['hlr'].value *= scale
        return gparam

    def get_uncvl_r2(self, gparam):
        return (gparam['hlr'].value *
                Sersic_r_2nd_moment_over_r_e(gparam['n'].value))**2

    def set_uncvl_r2(self, gparam, r2):
        gparam1 = copy.deepcopy(gparam)
        r2_now = self.get_uncvl_r2(gparam)
        scale = np.sqrt(r2 / r2_now)
        gparam1['hlr'].value = gparam['hlr'].value * scale
        return gparam1

    # def set_uncvl_r2(self, gparam, target_r2, oversample_factor=16):
    #     def r2_resid(scale):
    #         g1 = copy.deepcopy(gparam)
    #         g1['hlr'].value *= scale
    #         current_r2 = self.get_uncvl_r2(g1, oversample_factor=oversample_factor)
    #         return current_r2 - target_r2
    #     scale = newton(r2_resid, 1.0)
    #     gparam['hlr'].value *= scale
    #     return gparam

    def get_ring_params(self, gparam, ring_beta, ring_shear):
        gparam1 = copy.deepcopy(gparam)
        rot_phi = gparam['phi'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip = gparam['gmag'].value * \
          complex(np.cos(2.0 * rot_phi), np.sin(2.0 * rot_phi))
        c_gamma = ring_shear.g1 + 1j * ring_shear.g2
        # sheared complex ellipticity
        s_c_ellip = shear_galaxy(c_ellip, c_gamma)
        s_gmag = abs(s_c_ellip)
        s_phi = np.angle(s_c_ellip) / 2.0

        gparam1['x0'].value \
          = gparam['x0'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0'].value * np.sin(ring_beta / 2.0)
        gparam1['y0'].value \
          = gparam['x0'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0'].value * np.cos(ring_beta / 2.0)
        gparam1['gmag'].value = s_gmag
        gparam1['phi'].value = s_phi
        return gparam1


class DoubleSersicTool(GalTool):
    def __init__(self, SED1, SED2, bandpass, PSF, stamp_size, pixel_scale):
        self.SED1 = SED1
        self.SED2 = SED2
        self.bandpass = bandpass
        self.PSF = PSF
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale

    def _gparam_to_galsim(self, gparam):
        mono_gal1 = galsim.Sersic(n=gparam['n_1'].value,
                                 half_light_radius=gparam['hlr_1'].value)
        mono_gal1.applyShift(gparam['x0_1'].value, gparam['y0_1'].value)
        mono_gal1.applyShear(g=gparam['gmag_1'].value, beta=gparam['phi_1'].value * galsim.radians)
        mono_gal1.setFlux(gparam['flux_1'].value)
        gal1 = galsim.Chromatic(mono_gal1, self.SED1)

        mono_gal2 = galsim.Sersic(n=gparam['n_2'].value,
                                 half_light_radius=gparam['hlr_2'].value)
        mono_gal2.applyShift(gparam['x0_2'].value, gparam['y0_2'].value)
        mono_gal2.applyShear(g=gparam['gmag_2'].value, beta=gparam['phi_2'].value * galsim.radians)
        mono_gal2.setFlux(gparam['flux_2'].value)
        gal2 = galsim.Chromatic(mono_gal2, self.SED2)

        gal = gal1 + gal2
        return gal

    def set_r2(self, gparam, target_r2, oversample_factor=16):
        def r2_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr_1'].value *= scale
            g1['hlr_2'].value *= scale
            current_r2 = self.get_r2(g1, oversample_factor=oversample_factor)
            return current_r2 - target_r2
        scale = newton(r2_resid, 1.0)
        gparam['hlr_1'].value *= scale
        gparam['hlr_2'].value *= scale
        return gparam

    def set_uncvl_r2(self, gparam, target_r2, oversample_factor=16):
        def r2_resid(scale):
            g1 = copy.deepcopy(gparam)
            g1['hlr_1'].value *= scale
            g1['hlr_2'].value *= scale
            current_r2 = self.get_uncvl_r2(g1, oversample_factor=oversample_factor)
            return current_r2 - target_r2
        scale = newton(r2_resid, 1.0)
        gparam['hlr_1'].value *= scale
        gparam['hlr_2'].value *= scale
        return gparam

    def get_ring_params(self, gparam, ring_beta, ring_shear):
        c_gamma = ring_shear.g1 + 1j * ring_shear.g2
        gparam1 = copy.deepcopy(gparam)

        # gal1
        rot_phi_1 = gparam['phi_1'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip_1 = gparam['gmag_1'].value * \
          complex(np.cos(2.0 * rot_phi_1), np.sin(2.0 * rot_phi_1))
        # sheared complex ellipticity
        s_c_ellip_1 = shear_galaxy(c_ellip_1, c_gamma)
        s_gmag_1 = abs(s_c_ellip_1)
        s_phi_1 = np.angle(s_c_ellip_1) / 2.0

        gparam1['x0_1'].value \
          = gparam['x0_1'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0_1'].value * np.sin(ring_beta / 2.0)
        gparam1['y0_1'].value \
          = gparam['x0_1'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0_1'].value * np.cos(ring_beta / 2.0)
        gparam1['gmag_1'].value = s_gmag_1
        gparam1['phi_1'].value = s_phi_1

        # gal2
        rot_phi_2 = gparam['phi_2'].value + ring_beta/2.0
        # complex ellipticity
        c_ellip_2 = gparam['gmag_2'].value * \
          complex(np.cos(2.0 * rot_phi_2), np.sin(2.0 * rot_phi_2))
        # sheared complex ellipticity
        s_c_ellip_2 = shear_galaxy(c_ellip_2, c_gamma)
        s_gmag_2 = abs(s_c_ellip_2)
        s_phi_2 = np.angle(s_c_ellip_2) / 2.0

        gparam1['x0_2'].value \
          = gparam['x0_2'].value * np.cos(ring_beta / 2.0) \
          - gparam['y0_2'].value * np.sin(ring_beta / 2.0)
        gparam1['y0_2'].value \
          = gparam['x0_2'].value * np.sin(ring_beta / 2.0) \
          + gparam['y0_2'].value * np.cos(ring_beta / 2.0)
        gparam1['gmag_2'].value = s_gmag_2
        gparam1['phi_2'].value = s_phi_2

        return gparam1
