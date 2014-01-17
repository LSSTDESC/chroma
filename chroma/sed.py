import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
import extinction

import _mypath
import chroma

class SED(object):
    def __init__(self, wave, flambda):
        self.wave = wave
        self.flambda = flambda
        self.needs_new_interp=True
        self.redshift = 0.0

    def __call__(self, wave, force_new_interp=False):
        interp = self.get_interp(force_new_interp=force_new_interp)
        return interp(wave)

    def get_interp(self, force_new_interp=False):
        if force_new_interp or self.needs_new_interp:
            self.interp = interp1d(self.wave * (1.0 + self.redshift),
                                   self.wave * self.flambda / (1.0 + self.redshift))
            self.needs_new_interp=False
        return self.interp

    def scale(self, mag_norm, bandpass):
        current_mag = self.magnitude(bandpass)
        multiplier = 10**(-0.4 * (mag_norm - current_mag))
        self.flambda *= multiplier
        self.needs_new_interp=True

    def apply_redshift(self, redshift):
        if redshift != self.redshift:
            self.redshift = redshift
            self.needs_new_interp=True

    def apply_extinction(self, A_v, R_v=3.1):
        wgood = (self.wave > 91) & (self.wave < 6000)
        self.wave=self.wave[wgood]
        self.flambda=self.flambda[wgood]
        ext = extinction.reddening(self.wave*10, a_v=A_v, r_v=R_v, model='f99')
        self.flambda /= ext
        self.needs_new_interp=True

    def magnitude(self, bandpass):
        interp = self.get_interp()
        flux = simps(bandpass.throughput * interp(bandpass.wave), bandpass.wave)
        return -2.5 * np.log10(flux) - bandpass.AB_zeropoint()

    def DCR_moment_shifts(self, bandpass, zenith, **kwargs):
        """ Calculates shifts in first and second moments of surface brightness profile due to
        differential chromatic refraction (DCR)."""
        R = chroma.atm_refrac(bandpass.wave, zenith, **kwargs)
        interp = self.get_interp()
        photons = bandpass.throughput * interp(bandpass.wave)
        norm = simps(photons, bandpass.wave)
        Rbar = simps(R * photons, bandpass.wave) / norm
        V = simps((R-Rbar)**2 * photons, bandpass.wave) / norm
        return Rbar, V

    def seeing_shift(self, bandpass, alpha=-0.2, base_wavelength=500.0):
        """ Calculates relative size of PSF that scales like a powerlaw in wavelength.
        """
        interp = self.get_interp()
        photons = bandpass.throughput * interp(bandpass.wave)
        return (simps(photons * (bandpass.wave/500.0)**(2*alpha), bandpass.wave) /
                simps(photons, bandpass.wave))

class Bandpass(object):
    def __init__(self, wave, throughput):
        self.wave = np.array(wave)
        self.throughput = np.array(throughput)
        self.bluelim = self.wave[0]
        self.redlim = self.wave[-1]
        self.interp = interp1d(wave, throughput)

    def __call__(self, wave):
        return self.interp(wave)

    def AB_zeropoint(self, force_new_zeropoint=False):
        if not (hasattr(self, 'zp') or force_new_zeropoint):
            AB_source = 3631e-23 # 3631 Jy -> erg/s/Hz/cm^2
            c = 29979245800.0 # speed of light in cm/s
            nm_to_cm = 1.0e-7
            # convert AB source from erg/s/Hz/cm^2*cm/s/nm^2 -> erg/s/cm^2/nm
            AB_flambda = AB_source * c / self.wave**2 / nm_to_cm
            AB_photons = AB_flambda * self.wave * self.throughput
            AB_flux = simps(AB_photons, self.wave)
            self.zp = -2.5 * np.log10(AB_flux)
        return self.zp

    def truncate(self, rel_throughput=None, bluelim=None, redlim=None):
        if bluelim is None:
            bluelim = self.bluelim
        if redlim is None:
            redlim = self.redlim
        if rel_throughput is not None:
            mx = self.throughput.max()
            w = (self.throughput > mx*rel_throughput).nonzero()
            bluelim = max([min(self.wave[w]), bluelim]) # first `1` in `w`
            redlim = min([max(self.wave[w]), redlim]) # last `1` in `w`
        w = (self.wave >= bluelim) & (self.wave <= redlim)
        self.wave = self.wave[w]
        self.throughput = self.throughput[w]
        self.bluelim = self.wave[0]
        self.redlim = self.wave[-1]
        self.interp = interp1d(self.wave, self.throughput)
