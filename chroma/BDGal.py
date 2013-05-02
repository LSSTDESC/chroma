import copy

import numpy
import scipy
import lmfit

import chroma.utils

class BDGal(object):
    ''' Class to instantiate bulge+disk galaxies.'''
    def __init__(self, gparam0, bd_engine,
                 bulge_PSF=None, disk_PSF=None, composite_PSF=None,
                 wave=None, bulge_photons=None, disk_photons=None,
                 composite_photons=None,
                 PSF_model=None, PSF_kwargs=None):
        ''' Initialize BDGal with specificed bulge/disk parameters.  PSFs may be specified directly
        with `bulge_PSF`, `disk_PSF`, and `composite_PSF`, or alternatively by specifying a
        `PSF_model` together with spectra for the bulge and disk.  If specifying spectra, then a
        composite spectrum can either be specified directly with `composite_photons`, or will be
        created from `bulge_photons`, `disk_photons`, and the b+d flux params in `gparam0`.

        Arguments
        ---------
        gparam0 -- lmfit.Parameters object describing bulge+disk galaxy.  Params for each of the
                   bulge and disk components include:
                       `x0`, `y0` -- center of Sersic
                       `r_e` -- half light radius of Sersic
                       `gmag` -- magnitude of ellipticity
                       `phi` -- position angle of ellipticity
                       `flux` -- flux of component
                   Bulge params carry a `b_` prefix, disk params carry a `d_` prefix.
        bd_engine -- image creation engine.  Possible instances are located in imgen.py
        bulge_PSF, disk_PSF -- PSFs for the bulge/disk compatible with the specified bd_engine.
        composite_PSF -- the PSF to use when fitting the target image constructed with the
                         bulge/disk PSFs.
        wave -- wavelength array (in nm) for both bulge, disk and possible composite spectra.
        bulge_photons -- Spectrum of bulge component, proportional to photons/s/cm^2/A
        disk_photons -- Spectrum of disk component, proportional to photons/s/cm^2/A
        composite_photons -- Spectrum of bulge+disk.
        PSF_model -- callable that will produce PSF given spectrum.  Possible instances
                     are located in PSF_model.py
        PSF_kwargs -- addition arguments for PSF_model
        '''
        self.gparam0 = gparam0
        self.bd_engine = bd_engine
        if bulge_PSF is not None:
            self.bulge_PSF = bulge_PSF
            self.disk_PSF = disk_PSF
            self.composite_PSF = composite_PSF
        else:
            self.wave = wave
            self.bulge_photons = bulge_photons / scipy.integrate.simps(bulge_photons, wave)
            self.disk_photons = disk_photons / scipy.integrate.simps(disk_photons, wave)
            self.bulge_photons = bulge_photons / scipy.integrate.simps(bulge_photons, wave)
            self.disk_photons = disk_photons / scipy.integrate.simps(disk_photons, wave)
            self.PSF_model = PSF_model
            self.PSF_kwargs = PSF_kwargs
            if composite_photons is not None:
                self.composite_photons = composite_photons
            else:
                self.composite_photons = self.bulge_photons * self.gparam0['b_flux'].value \
                  + self.disk_photons * self.gparam0['d_flux'].value
            self.build_PSFs()

    def build_PSFs(self):
        ''' Use `self.PSF_model` to instantiate PSFs from bulge/disk spectra.'''
        self.bulge_PSF = self.PSF_model(self.wave, self.bulge_photons, **self.PSF_kwargs)
        self.disk_PSF = self.PSF_model(self.wave, self.disk_photons, **self.PSF_kwargs)
        self.composite_PSF = self.PSF_model(self.wave, self.composite_photons, **self.PSF_kwargs)

    def build_circ_PSF(self):
        ''' Use `self.PSF_model` to instantiate a circularly symmetric PSF from composite
        bulge+disk spectrum.  Requires that `ellipticity` and `phi` are part of the PSF_kwargs.'''
        PSF_kwargs2 = copy.deepcopy(self.PSF_kwargs)
        PSF_kwargs2['ellipticity']=0.0
        PSF_kwargs2['phi']=0.0
        self.circ_PSF = self.PSF_model(self.wave, self.composite_photons, **PSF_kwargs2)

    def set_FWHM_ratio(self, rpg):
        '''Set the effective radii of the bulge+disk galaxy specified in `self.gparam0` such that the
        ratio of the FWHM of the PSF-convolved galaxy image is `rpg` times the FWHM of the PSF
        itself.  The galaxy is circularized and centered at the origin for this computation
        (ellip -> 0.0) and (x0, y0 -> 0.0, 0.0), and the PSF derived from the composite spectrum and
        set to be circular.
        '''
        FWHM_PSF = self.bd_engine.PSF_FWHM(self.circ_PSF)
        gparam1 = copy.deepcopy(self.gparam0)
        gparam1['b_gmag'].value = 0.0
        gparam1['b_x0'].value = 0.0
        gparam1['b_y0'].value = 0.0
        gparam1['d_gmag'].value = 0.0
        gparam1['d_x0'].value = 0.0
        gparam1['d_y0'].value = 0.0
        def FWHM_gal(scale):
            gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * scale
            gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * scale
            return self.bd_engine.bdcvl_FWHM(gparam1, self.circ_PSF, self.circ_PSF)
        def f(scale):
            return FWHM_gal(scale) - rpg * FWHM_PSF
        scale = scipy.optimize.newton(f, 1.0)
        self.gparam0['b_r_e'].value *= scale
        self.gparam0['d_r_e'].value *= scale

    def gen_target_image(self, gamma, beta):
        ''' Generate a target "truth" image for ring test.

        Arguments
        ---------
        gamma -- the input shear for the ring test.  Complex number.
        beta -- angle around the ellipticity ring.
        '''
        gparam1 = self.gen_init_param(gamma, beta)
        return self.bd_engine.bd_image(gparam1, self.bulge_PSF, self.disk_PSF)

    def gen_init_param(self, gamma, beta):
        ''' Adjust bulge+disk parameters in self.gparam0 to reflect applied shear `gamma` and
        angle around the ring `beta` in a ring test.  Returned parameters are good both for
        creating the target image and for initializing the lmfit minimize routine.
        '''
        gparam1 = copy.deepcopy(self.gparam0)
        b_phi_ring = self.gparam0['b_phi'].value + beta/2.0
        d_phi_ring = self.gparam0['d_phi'].value + beta/2.0
        # bulge complex ellipticity
        b_c_ellip = self.gparam0['b_gmag'].value * \
          complex(numpy.cos(2.0 * b_phi_ring), numpy.sin(2.0 * b_phi_ring))
        # bulge sheared complex ellipticity
        b_s_c_ellip = chroma.utils.shear_galaxy(b_c_ellip, gamma)
        b_s_gmag = abs(b_s_c_ellip)
        b_s_phi = numpy.angle(b_s_c_ellip) / 2.0
        # disk complex ellipticity
        d_c_ellip = self.gparam0['d_gmag'].value * \
          complex(numpy.cos(2.0 * d_phi_ring), numpy.sin(2.0 * d_phi_ring))
        # disk sheared complex ellipticity
        d_s_c_ellip = chroma.utils.shear_galaxy(d_c_ellip, gamma)
        d_s_gmag = abs(d_s_c_ellip)
        d_s_phi = numpy.angle(d_s_c_ellip) / 2.0
        # radius rescaling
        rescale = numpy.sqrt(1.0 - abs(gamma)**2.0)

        gparam1['b_y0'].value \
          = self.gparam0['b_y0'].value * numpy.sin(beta / 2.0) \
          + self.gparam0['b_x0'].value * numpy.cos(beta / 2.0)
        gparam1['b_x0'].value \
          = self.gparam0['b_y0'].value * numpy.cos(beta / 2.0) \
          - self.gparam0['b_x0'].value * numpy.sin(beta / 2.0)
        gparam1['d_y0'].value \
          = self.gparam0['d_y0'].value * numpy.sin(beta / 2.0) \
          + self.gparam0['d_x0'].value * numpy.cos(beta / 2.0)
        gparam1['d_x0'].value \
          = self.gparam0['d_y0'].value * numpy.cos(beta / 2.0) \
          - self.gparam0['d_x0'].value * numpy.sin(beta / 2.0)
        gparam1['b_gmag'].value = b_s_gmag
        gparam1['d_gmag'].value = d_s_gmag
        gparam1['b_phi'].value = b_s_phi
        gparam1['d_phi'].value = d_s_phi
        gparam1['b_r_e'].value = self.gparam0['b_r_e'].value * rescale
        gparam1['d_r_e'].value = self.gparam0['d_r_e'].value * rescale
        return gparam1

    def measure_ellip(self, target_image, init_param):
        ''' Find the best fitting image to the "truth" target image by generating trial images
        using the composite bulge+disk PSF.
        '''
        def resid(param):
            im = self.bd_engine.bd_image(param, self.composite_PSF, self.composite_PSF)
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        gmag = result.params['d_gmag'].value
        phi = result.params['d_phi'].value
        c_ellip = gmag * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip
