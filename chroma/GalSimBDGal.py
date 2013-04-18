import copy

import numpy
from scipy.integrate import simps
from scipy.optimize import newton
from lmfit import minimize, report_errors
import galsim

from chroma import BDGal

def GSEuclidPSF(wave, photons, ellipticity=0.0, phi=0.0):
    hlr = lambda wave: 0.7 * (wave / 520.0)**0.6 # pixels
    mpsfs = []
    photons /= simps(photons, wave)
    for w, p in zip(wave, photons):
        mpsfs.append(galsim.Gaussian(flux=p, half_light_radius=hlr(w)))
    PSF = galsim.Add(mpsfs)
    beta = phi * galsim.radians
    PSF.applyShear(g=ellipticity, beta=beta)
    return PSF

class GalSimBDGal(BDGal):
    def __init__(self, gparam0, wave, bulge_photons, disk_photons,
                 PSF_ellip=0.0, PSF_phi=0.0,
                 oversample_factor=7.0):
        self.oversample_factor = oversample_factor
        super(GalSimBDGal, self).__init__(gparam0, wave, bulge_photons, disk_photons,
                                          PSF_ellip=PSF_ellip, PSF_phi=PSF_phi)
        self.build_PSFs()

    def build_PSFs(self):
        self.bulge_PSF = GSEuclidPSF(self.wave, self.bulge_photons,
                                     ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.disk_PSF = GSEuclidPSF(self.wave, self.disk_photons,
                                    ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.composite_PSF = GSEuclidPSF(self.wave, self.composite_photons,
                                         ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.circ_PSF = GSEuclidPSF(self.wave, self.composite_photons,
                                    ellipticity=0.0, phi=0.0)

    def PSF_image(self, PSF):
        PSF_image = galsim.ImageD(119,119)
        PSF.draw(image=PSF_image, dx=1.0/self.oversample_factor)
        return PSF_image.array

    def gal_overimage(self, gparam, bulge_PSF, disk_PSF):
        '''Compute oversampled galaxy image.  Similar to `gal_image()`.

        Useful for computing FWHM of galaxy image at higher resolution than available from just
        `gal_image()`.
        '''
        bulge = galsim.Sersic(n=gparam['b_n'].value, half_light_radius=gparam['b_r_e'].value)
        bulge.applyShift(gparam['b_x0'].value, gparam['b_y0'].value)
        bulge.applyShear(g=gparam['b_gmag'].value, beta=gparam['b_phi'].value * galsim.radians)
        bulge.setFlux(gparam['b_flux'].value)
        disk = galsim.Sersic(n=gparam['d_n'].value, half_light_radius=gparam['d_r_e'].value)
        disk.applyShift(gparam['d_x0'].value, gparam['d_y0'].value)
        disk.applyShear(g=gparam['d_gmag'].value, beta=gparam['d_phi'].value * galsim.radians)
        disk.setFlux(gparam['d_flux'].value)
        pixel = galsim.Pixel(1./self.oversample_factor)
        bulge_cvl = galsim.Convolve(bulge, bulge_PSF, pixel)
        disk_cvl = galsim.Convolve(disk, disk_PSF, pixel)
        gal = bulge_cvl + disk_cvl
        gal_overim = galsim.ImageD(119,119)
        gal.draw(image=gal_overim, dx=1./self.oversample_factor)
        return gal_overim.array

    def gal_image(self, gparam, bulge_PSF, disk_PSF):
        '''Use galsim to make a galaxy image from params in gparam and using the bulge and disk
        PSFs `bulge_PSF` and `disk_PSF`.

        Arguments
        ---------
        gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
                  `b_` prefix for bulge, `d_` prefix for disk.
                  Suffixes are all init arguments for the Sersic object.

        Note that you can specify the composite PSF `c_PSF` for both bulge and disk PSF when using
        during ringtest fits.
        '''
        bulge = galsim.Sersic(n=gparam['b_n'].value, half_light_radius=gparam['b_r_e'].value)
        bulge.applyShift(gparam['b_x0'].value, gparam['b_y0'].value)
        bulge.applyShear(g=gparam['b_gmag'].value, beta=gparam['b_phi'].value * galsim.radians)
        bulge.setFlux(gparam['b_flux'].value)
        disk = galsim.Sersic(n=gparam['d_n'].value, half_light_radius=gparam['d_r_e'].value)
        disk.applyShift(gparam['d_x0'].value, gparam['d_y0'].value)
        disk.applyShear(g=gparam['d_gmag'].value, beta=gparam['d_phi'].value * galsim.radians)
        disk.setFlux(gparam['d_flux'].value)
        pixel = galsim.Pixel(1.0)
        bulge_cvl = galsim.Convolve(bulge, bulge_PSF, pixel)
        disk_cvl = galsim.Convolve(disk, disk_PSF, pixel)
        gal = bulge_cvl + disk_cvl
        gal_im = galsim.ImageD(15, 15)
        gal.draw(image=gal_im, dx=1.0)
        return gal_im.array

def GSEuclidPSFInt(wave, photons, ellipticity=0.0, phi=0.0):
    hlr = lambda wave: 0.7 * (wave / 520.0)**0.6 # pixels
    mpsfs = []
    photons /= simps(photons, wave)
    for w, p in zip(wave, photons):
        mpsfs.append(galsim.Gaussian(flux=p, half_light_radius=hlr(w)))
    PSF = galsim.Add(mpsfs)
    beta = phi * galsim.radians
    PSF.applyShear(g=ellipticity, beta=beta)
    im = galsim.ImageD(119, 119)
    PSF.draw(image=im, dx=1.0/21)
    PSF = galsim.InterpolatedImage(im, dx=1.0/21)
    return PSF

class GalSimBDGalInt(GalSimBDGal):
    def build_PSFs(self):
        self.bulge_PSF = GSEuclidPSFInt(self.wave, self.bulge_photons,
                                        ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.disk_PSF = GSEuclidPSFInt(self.wave, self.disk_photons,
                                       ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.composite_PSF = GSEuclidPSFInt(self.wave, self.composite_photons,
                                            ellipticity=self.PSF_ellip, phi=self.PSF_phi)
        self.circ_PSF = GSEuclidPSFInt(self.wave, self.composite_photons,
                                       ellipticity=0.0, phi=0.0)
