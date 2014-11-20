""" Classes to describe SEDs and bandpasses defined through wavelength samples.
"""

import copy

import numpy as np
from scipy.interpolate import interp1d

import extinction
import dcr

class SampledSED(object):
    """Simple SED object.

    SEDs are callable, returning the flux in photons/nm as a function of wavelength in nm.

    SEDs are immutable; all transformative SED methods return *new* SEDs, and leave their
    originating SEDs unaltered.

    SEDs have `blue_limit` and `red_limit` attributes, which are defined automatically from the
    samples provided at instantiation.  SEDs are considered undefined outside of this range, and
    __call__ will raise an exception if a flux is requested outside of this range.

    SEDs may be multiplied by scalars or scalar functions of wavelength.

    SEDs may be added together.  The resulting SED will only be defined on the wavelength
    region where both of the operand SEDs are defined. `blue_limit` and `red_limit` will be reset
    accordingly.
    """
    def __init__(self, spec, wave_type='nm', flux_type='flambda'):
        """Simple SED object.  This object is callable, returning the flux in
        photons/nm as a function of wavelength in nm.

        The input parameter, `spec`, may be either:
        1. a 2-column file (wave, flux) from which to initialize an scipy.interpolate.interp1d
           object.
        2. a scipy.interpolate.interp1d object which returns flux as a fn of wavelength

        The argument of `spec` will be the wavelength in either nanometers (default) or
        Angstroms depending on the value of `wave_type`.  The output should be the flux density at
        that wavelength.

        The argument `wave_type` specifies the units to assume for wavelength and must be one of
        'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
        is unimportant.

        The argument `flux_type` specifies the type of spectral density and must be one of:
        1. 'flambda':  `spec` is proportional to erg/nm
        2. 'fnu':      `spec` is proportional to erg/Hz
        3. 'fphotons': `spec` is proportional to photons/nm

        Note that the `wave_type` and `flux_type` parameters do not propagate into other methods of
        `SED`.  For instance, SED.__call__ assumes its input argument is in nanometers and returns
        flux proportional to photons/nm.

        @param spec          Argument defining the spectrum at each wavelength.  See above for
                             valid options for this parameter.
        @param flux_type     String specifying what type of spectral density `spec` represents.  See
                             above for valid options for this parameter.
        @param wave_type     String specifying units for wavelength input to `spec`.
        """
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type `{}` in SED.__init__".format(wave_type))

        if isinstance(spec, basestring):
            try:
                w, f = np.genfromtxt(spec).T
            except:
                raise ValueError("Could not create interp1d from file: {}".format(spec))
        else:
            w = self.spec.x
            f = self.spec.y

        self.blue_limit = w[0] / wave_factor
        self.red_limit = w[-1] / wave_factor
        if flux_type == 'flambda':
            self.interp = interp1d(w / wave_factor, f * w)
        elif flux_type == 'fnu':
            self.interp = interp1d(w / wave_factor, f / w)
        elif flux_type == 'fphotons':
            self.interp = interp1d(w / wave_factor, f)
        else:
            raise ValueError("Unknown flux_type `{}` in SampledSED.__init__".format(flux_type))

    def __call__(self, wave):
        """ Return flux density in photons/s/nm as a function of wavelength in nm.
        """
        try:
            return self.interp(wave)
        except ValueError:
            raise ValueError("Wavelength out of range for SED")

    def copy(self):
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            ret.__dict__[k] = copy.deepcopy(v)
        return ret

    def __mul__(self, other):
        """ Multiply SED by either a function of wavelength in nm or a constant.
        """
        ret = self.copy()
        if hasattr(other, '__call__'):
            ret.interp = interp1d(self.interp.x, self.interp.y * other(self.interp.x))
        else:
            ret.interp = interp1d(self.interp.x, self.interp.y * other)
        return ret

    def __rmul__(self, other):
        return self*other

    def __add__(self, other):
        """ Add two SED's together.  The new samples are the union of the operand samples, but only
        in the intersecting region of their wavelength ranges.
        """
        blue_limit = max([self.blue_limit, other.blue_limit])
        red_limit = min([self.red_limit, other.red_limit])
        ret = self.copy()
        waves = np.array(list(set(self.interp.x).union(other.interp.x)))
        waves.sort()
        waves = waves[(waves >= blue_limit) & (waves <= red_limit)]
        fluxes = self(waves) + other(waves)
        ret.interp = interp1d(waves, fluxes)
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        return ret

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        return self + (-1.0 * other)

    def getFlux(self, base):
        """ If base is a number, return the fluxDensity in photons/nm at that wavelength.
            If base is a SampledBandpass, return the integrated flux in photons through that bandpass.
        """
        if isinstance(base, SampledBandpass):
            photons = self(base.interp.x) * base(base.interp.x)
            return np.trapz(photons, base.interp.x)
        else:
            return self(base)

    def createWithFlux(self, base, target):
        """ If base is a number, create a new SampledSED with the target photons/nm at that
        wavelength. If base is a SampledBanspass, create a new SampledSED with the target flux in
        photons through that bandpass.
        """
        current_flux = self(base)
        norm = target / current_flux
        return self * norm

    def createRedshifted(self, redshift):
        """ Create a new SampledSED with redshifted wavelength.
        """
        ret = self.copy()
        ret.interp = interp1d(self.interp.x * (1.0 + redshift), self.interp.y)
        ret.blue_limit = self.blue_limit * (1.0 + redshift)
        ret.red_limit = self.red_limit * (1.0 + redshift)
        return ret

    def getMagnitude(self, bandpass):
        """ Return the AB magnitude through the given bandpass.
        """
        wave_list = bandpass.interp.x
        flux = np.trapz(bandpass(wave_list) * self(wave_list), wave_list)
        return -2.5 * np.log10(flux) - bandpass.getABZeropoint()

    def createWithMagnitude(self, bandpass, target):
        """ Return a new SampledSED with the specified magnitude through the specified bandpass.
        """
        current_mag = self.getMagnitude(bandpass)
        scale = 10**(-0.4 * (target - current_mag))
        return self * scale

    def getDCRMomentShifts(self, bandpass, zenith, **kwargs):
        """ Calculates shifts in first and second moments of surface brightness profile due to
        differential chromatic refraction (DCR)."""
        wave_list = bandpass.interp.x
        R = dcr.get_refraction(wave_list, zenith, **kwargs)
        photons = bandpass(wave_list) * self(wave_list)
        norm = np.trapz(photons, wave_list)
        Rbar = np.trapz(R * photons, wave_list) / norm
        V = np.trapz((R-Rbar)**2 * photons, wave_list) / norm
        return Rbar, V

    def getSeeingShift(self, bandpass, alpha=-0.2, base_wavelength=500.0):
        """ Calculates relative size of PSF that scales like a powerlaw in wavelength.
        """
        wave_list = bandpass.interp.x
        photons = bandpass(wave_list) * self(wave_list)
        return (np.trapz(photons * (wave_list/500.0)**(2*alpha), wave_list) /
                np.trapz(photons, wave_list))

    def createExtincted(self, A_v, R_v=3.1):
        """ Return a new SampledSED with the specified extinction applied.  Note that this will
        truncate the wavelength range to lie between 91 nm and 6000 nm where the extinction
        correction law is defined.
        """
        wave = self.interp.x
        wgood = (wave >= 91) & (wave <= 3300)
        wave = wave[wgood]
        flux = self.interp.y[wgood]
        ext = extinction.reddening(wave * 10, a_v=A_v, r_v=R_v, model='f99')
        ret = self.copy()
        ret.interp = interp1d(wave, flux / ext)
        ret.blue_limit = wave[0]
        ret.red_limit = wave[-1]
        return ret

    def getDCRAngleDensity(self, bandpass, zenith, **kwargs):
        """Return photon density per unit refraction angle through a given filter.
        """
        wave = bandpass.interp.x
        photons = bandpass.interp.y * self(wave)
        R = dcr.get_refraction(wave, zenith, **kwargs)
        dR = np.diff(R)
        dwave = np.diff(wave)
        dwave_dR = dwave / dR # Jacobian
        dwave_dR = np.append(dwave_dR, dwave_dR[-1]) # fudge the last array element
        angle_dens = photons * np.abs(dwave_dR)
        return R, angle_dens

    def addEmissionLines(self):
        # get UV continuum flux
        UV_fphot = self(230.0) # photons / nm / s
        h = 6.62e-27 # ergs / Hz
        UV_fnu = UV_fphot * h * (230.0) # converted to erg/s/Hz

        wave = self.interp.x
        fphot = self.interp.y

        # then add lines appropriately
        lines = ['OII','OIII','Hbeta','Halpha','Lya']
        multipliers = np.r_[1.0, 0.36, 0.61, 1.77, 2.0] * 1.0e13
        waves = [372.7, 500.7, 486.1, 656.3, 121.5] # nm
        velocity = 200.0 # km/s
        for line, m, w in zip(lines, multipliers, waves):
            line_flux = UV_fnu * m # ergs / sec
            hc = 1.986e-9 # erg nm
            line_flux *= w / hc # converted to phot / sec
            sigma = velocity / 299792.458 * w # sigma in Angstroms
            amplitude = line_flux / sigma / np.sqrt(2.0 * np.pi)
            fphot += amplitude * np.exp(-(wave-w)**2/(2*sigma**2))
        ret = self.copy()
        ret.interp = interp1d(wave, fphot)
        return ret


class SampledBandpass(object):
    """Simple bandpass object.

    Bandpasses are callable, returning dimensionless throughput as a function of wavelength in nm.

    Bandpasses are immutable; all transformative methods return *new* Bandpasses, and leave their
    originating Bandpasses unaltered.

    Bandpasses have `blue_limit` and `red_limit` attributes, which are inferred from the
    initializing scipy.interpolate.interp1d or 2-column file.

    Outside of the wavelength interval between `blue_limit` and `red_limit`, the throughput is
    assumed to be zero.
    """
    def __init__(self, throughput, wave_type='nm'):
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type `{}` in SED.__init__".format(wave_type))

        if isinstance(throughput, basestring):
            try:
                w, tp = np.genfromtxt(throughput).T
            except:
                raise ValueError("Could not create interp1d from file: {}".format(spec))
            self.blue_limit = w[0] / wave_factor
            self.red_limit = w[-1] / wave_factor
            self.interp = interp1d(w / wave_factor, tp, bounds_error=False, fill_value=0.0)
        else:
            self.blue_limit = throughput.x[0]
            self.red_limit = throughput.x[1]
            self.interp = interp1d(throughput.x, throughput.y, bounds_error=False, fill_value=0.0)

    def __call__(self, wave):
        """ Return dimensionless throughput as function of wavelength in nm.
        """
        try:
            return self.interp(wave)
        except ValueError:
            raise ValueError("Wavelength out of range for SampledBandpass")

    def copy(self):
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            ret.__dict__[k] = copy.deepcopy(v)
        return ret

    def createTruncated(self, relative_throughput=None, blue_limit=None, red_limit=None):
        """ Return a new SampledBandpass with its wavelength range truncated.

        @param blue_limit             Truncate blue side of bandpass here.
        @param red_limit              Truncate red side of bandpass here.
        @param relative_throughput    Truncate leading and trailing wavelength ranges where the
                                      relative throughput is less than this amount.  Do not remove
                                      any intermediate wavelength ranges.
        @returns   The truncated SampledBandpass.

        """
        if blue_limit is None:
            blue_limit = self.blue_limit
        if red_limit is None:
            red_limit = self.red_limit
        wave = self.interp.x
        tp = self.interp.y
        if relative_throughput is not None:
            w = (tp >= tp.max()*relative_throughput).nonzero()
            blue_limit = max([min(wave[w]), blue_limit])
            red_limit = min([max(wave[w]), red_limit])
        w = (wave >= blue_limit) & (wave <= red_limit)
        ret = self.copy()
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        ret.interp = interp1d(wave[w], tp[w], bounds_error=False, fill_value=0.0)
        return ret

    def getABZeropoint(self):
        """ Return (and cache) the AB zeropoint of the bandpass.
        """
        if not (hasattr(self, 'zp')):
            AB_source = 3631e-23 # 3631 Jy -> erg/s/Hz/cm^2
            c = 29979245800.0 # speed of light in cm/s
            nm_to_cm = 1.0e-7
            wave_list = np.array(self.interp.x)
            # convert AB source from erg/s/Hz/cm^2*cm/s/nm^2 -> erg/s/cm^2/nm
            AB_flambda = AB_source * c / wave_list**2 / nm_to_cm
            AB_photons = AB_flambda * wave_list * self(wave_list)
            AB_flux = np.trapz(AB_photons, wave_list)
            self.zp = -2.5 * np.log10(AB_flux)
        return self.zp

    def createThinned(self, step):
        """ Return a new SampledBandpass with its samples thinned by a factor of `step`.
        """
        wave = self.interp.x[::step]
        if wave[-1] != self.interp.x[-1]:
            wave = np.concatenate([wave, [self.interp.x[-1]]])
        tp = self(wave)
        return SampledBandpass(interp1d(wave, tp, bounds_error=False, fill_value=0.0))
