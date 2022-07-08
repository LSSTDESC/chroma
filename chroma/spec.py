"""@file sampled
For sampled (non-analytic) SEDs and Bandpasses.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

import chroma.dcr
import chroma.extinction


class SED(object):
    """Simple SED object to represent the spectral energy distributions of stars and galaxies.

    SEDs are callable, returning the flux in photons/nm as a function of wavelength in nm.

    SEDs are immutable; all transformative SED methods return *new* SEDs, and leave their
    originating SEDs unaltered.

    SEDs have `blue_limit` and `red_limit` attributes, which may be set to `None` in the case that
    the SED is defined by a python function or lambda `eval` string.  SEDs are considered undefined
    outside of this range, and __call__ will raise an exception if a flux is requested outside of
    this range.

    SEDs may be multiplied by scalars or scalar functions of wavelength.

    SEDs may be added together if they are at the same redshift.  The resulting SED will only be
    defined on the wavelength region where both of the operand SEDs are defined. `blue_limit` and
    `red_limit` will be reset accordingly.

    The input parameter, `spec`, may be one of several possible forms:
    1. a regular python function (or an object that acts like a function)
    2. a numpy.interp1d object
    3. a file from which a numpy.interp1d can be read in
    4. a string which can be evaluated into a function of `wave`
       via `eval('lambda wave : '+spec)
       e.g. spec = '0.8 + 0.2 * (wave-800)`

    The argument of `spec` will be the wavelength in either nanometers (default) or Angstroms
    depending on the value of `wave_type`.  The output should be the flux density at that
    wavelength.  (Note we use `wave` rather than `lambda`, since `lambda` is a python reserved
    word.)

    The argument `wave_type` specifies the units to assume for wavelength and must be one of
    'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
    is unimportant.  If these wavelength options are insufficient, please submit an issue to
    the GalSim github issues page: https://github.com/GalSim-developers/GalSim/issues

    The argument `flux_type` specifies the type of spectral density and must be one of:
    1. 'flambda':  `spec` is proportional to erg/nm
    2. 'fnu':      `spec` is proportional to erg/Hz
    3. 'fphotons': `spec` is proportional to photons/nm

    Note that the `wave_type` and `flux_type` parameters do not propagate into other methods of
    `SED`.  For instance, SED.__call__ assumes its input argument is in nanometers and returns
    flux proportional to photons/nm.

    @param spec          Function defining the spectrum at each wavelength.  See above for
                         valid options for this parameter.
    @param wave_type     String specifying units for wavelength input to `spec`. [default: 'nm']
    @param flux_type     String specifying what type of spectral density `spec` represents.  See
                         above for valid options for this parameter. [default: 'flambda']

    """
    def __init__(self, spec, wave_type='nm', flux_type='flambda'):
        # Figure out input wavelength type
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type '{0}'".format(wave_type))

        # Figure out input flux density type
        if isinstance(spec, str):
            import os
            if os.path.isfile(spec):
                w, f = np.genfromtxt(spec).T
                spec = interp1d(w, f)
            else:
                origspec = spec
                # Don't catch ArithmeticErrors when testing to see if the the result of `eval()`
                # is valid since `spec = '1./(wave-700)'` will generate a ZeroDivisionError (which
                # is a subclass of ArithmeticError) despite being a valid spectrum specification,
                # while `spec = 'blah'` where `blah` is undefined generates a NameError and is not
                # a valid spectrum specification.
                # Are there any other types of errors we should trap here?
                try:
                    spec = eval('lambda wave : ' + spec)   # This can raise SyntaxError
                    spec(700)   # This can raise NameError or ZeroDivisionError
                except ArithmeticError:
                    pass
                except:
                    raise ValueError(
                        "String spec must either be a valid filename or something that " +
                        "can eval to a function of wave. Input provided: {0}".format(origspec))

        if isinstance(spec, interp1d):
            self.blue_limit = spec.x.min() / wave_factor
            self.red_limit = spec.x.max() / wave_factor
            self.wave_list = np.array(spec.x)/wave_factor
        else:
            self.blue_limit = None
            self.red_limit = None
            self.wave_list = np.array([], dtype=float)

        # Do some SED unit conversions to make internal representation proportional to photons/nm.
        # Note that w should have units of nm below.
        c = 2.99792458e17   # speed of light in nm/s
        h = 6.62606957e-27  # Planck's constant in erg seconds
        if flux_type == 'flambda':
            # photons/nm = (erg/nm) * (photons/erg)
            #            = spec(w) * 1/(h nu) = spec(w) * lambda / hc
            self._rest_photons = lambda w: (spec(np.array(w) * wave_factor) * w / (h*c))
        elif flux_type == 'fnu':
            # photons/nm = (erg/Hz) * (photons/erg) * (Hz/nm)
            #            = spec(w) * 1/(h nu) * |dnu/dlambda|
            # [Use dnu/dlambda = d(c/lambda)/dlambda = -c/lambda^2 = -nu/lambda]
            #            = spec(w) * 1/(h lambda)
            self._rest_photons = lambda w: (spec(np.array(w) * wave_factor) / (w * h))
        elif flux_type == 'fphotons':
            # Already basically correct.  Just convert the units of lambda
            self._rest_photons = lambda w: spec(np.array(w) * wave_factor)
        else:
            raise ValueError("Unknown flux_type '{0}'".format(flux_type))
        self.redshift = 0

    def _wavelength_intersection(self, other):
        blue_limit = self.blue_limit
        if other.blue_limit is not None:
            if blue_limit is None:
                blue_limit = other.blue_limit
            else:
                blue_limit = max([blue_limit, other.blue_limit])

        red_limit = self.red_limit
        if other.red_limit is not None:
            if red_limit is None:
                red_limit = other.red_limit
            else:
                red_limit = min([red_limit, other.red_limit])

        return blue_limit, red_limit

    def __call__(self, wave):
        """ Return photon density at wavelength `wave`.

        Note that outside of the wavelength range defined by the `blue_limit` and `red_limit`
        attributes, the SED is considered undefined, and this method will raise an exception if a
        flux at a wavelength outside the defined range is requested.

        @param wave     Wavelength in nanometers at which to evaluate the SED.

        @returns the photon density in units of photons/nm
        """
        if hasattr(wave, '__iter__'):  # Only iterables respond to min(), max()
            wmin = min(wave)
            wmax = max(wave)
        else:  # python scalar
            wmin = wave
            wmax = wave
        extrapolation_slop = 1.e-6  # allow a small amount of extrapolation
        if self.blue_limit is not None:
            if wmin < self.blue_limit - extrapolation_slop:
                raise ValueError("Requested wavelength ({0}) is bluer than blue_limit ({1})"
                                 .format(wmin, self.blue_limit))
        if self.red_limit is not None:
            if wmax > self.red_limit + extrapolation_slop:
                raise ValueError("Requested wavelength ({0}) is redder than red_limit ({1})"
                                 .format(wmax, self.red_limit))
        wave_factor = 1.0 + self.redshift
        # figure out what we received, and return the same thing
        # option 1: a numpy array
        if isinstance(wave, np.ndarray):
            return self._rest_photons(wave / wave_factor)
        # option 2: a tuple
        elif isinstance(wave, tuple):
            return tuple(self._rest_photons(np.array(wave) / wave_factor))
        # option 3: a list
        elif isinstance(wave, list):
            return list(self._rest_photons(np.array(wave) / wave_factor))
        # option 4: a single value
        else:
            return self._rest_photons(wave / wave_factor)

    def __mul__(self, other):
        # SEDs can be multiplied by scalars or functions (callables)
        ret = self.copy()
        if hasattr(other, '__call__'):
            wave_factor = 1.0 + self.redshift
            ret._rest_photons = lambda w: self._rest_photons(w) * other(w * wave_factor)
        else:
            ret._rest_photons = lambda w: self._rest_photons(w) * other
        return ret

    def __rmul__(self, other):
        return self*other

    def __div__(self, other):
        # SEDs can be divided by scalars or functions (callables)
        ret = self.copy()
        if hasattr(other, '__call__'):
            wave_factor = 1.0 + self.redshift
            ret._rest_photons = lambda w: self._rest_photons(w) / other(w * wave_factor)
        else:
            ret._rest_photons = lambda w: self._rest_photons(w) / other
        return ret

    def __rdiv__(self, other):
        # SEDs can be divided by scalars or functions (callables)
        ret = self.copy()
        if hasattr(other, '__call__'):
            wave_factor = 1.0 + self.redshift
            ret._rest_photons = lambda w: other(w * wave_factor) / self._rest_photons(w)
        else:
            ret._rest_photons = lambda w: other / self._rest_photons(w)
        return ret

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __add__(self, other):
        # Add together two SEDs, with the following caveats:
        # 1) The SEDs must have the same redshift.
        # 2) The resulting SED will be defined on the wavelength range set by the overlap of the
        #    wavelength ranges of the two SED operands.
        # 3) If both SEDs maintain a `wave_list` attribute, then the new `wave_list` will be
        #    the union of the old `wave_list`s in the intersecting region.
        # This ensures that SED addition is commutative.

        if self.redshift != other.redshift:
            raise ValueError("Can only add SEDs with same redshift.")
        # Find overlapping wavelength interval
        blue_limit, red_limit = self._wavelength_intersection(other)
        ret = self.copy()
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        ret._rest_photons = lambda w: self._rest_photons(w) + other._rest_photons(w)
        if len(self.wave_list) > 0 and len(other.wave_list) > 0:
            wave_list = np.union1d(self.wave_list, other.wave_list)
            wave_list = wave_list[wave_list <= red_limit]
            wave_list = wave_list[wave_list >= blue_limit]
            ret.wave_list = wave_list
        return ret

    def __sub__(self, other):
        # Subtract two SEDs, with the same caveats as adding two SEDs.
        return self.__add__(-1.0 * other)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def withFluxDensity(self, target_flux_density, wavelength):
        """ Return a new SED with flux density set to `target_flux_density` at wavelength
        `wavelength`.  Note that this normalization is *relative* to the `flux` attribute of the
        chromaticized GSObject.

        @param target_flux_density  The target *relative* normalization in photons / nm.
        @param wavelength           The wavelength, in nm, at which flux density will be set.

        @returns the new normalized SED.
        """
        current_flux_density = self(wavelength)
        factor = target_flux_density / current_flux_density
        ret = self.copy()
        ret._rest_photons = lambda w: self._rest_photons(w) * factor
        return ret

    def withFlux(self, target_flux, bandpass):
        """ Return a new SED with flux through the Bandpass `bandpass` set to `target_flux`. Note
        that this normalization is *relative* to the `flux` attribute of the chromaticized GSObject.

        @param target_flux  The desired *relative* flux normalization of the SED.
        @param bandpass     A Bandpass object defining a filter bandpass.

        @returns the new normalized SED.
        """
        current_flux = self.calculateFlux(bandpass)
        norm = target_flux/current_flux
        ret = self.copy()
        ret._rest_photons = lambda w: self._rest_photons(w) * norm
        return ret

    def withMagnitude(self, target_magnitude, bandpass):
        """ Return a new SED with magnitude through `bandpass` set to `target_magnitude`.  Note
        that this requires `bandpass` to have been assigned a zeropoint using
        `Bandpass.withZeropoint()`.  When the returned SED is multiplied by a GSObject with
        flux=1, the resulting ChromaticObject will have magnitude `target_magnitude` when drawn
        through `bandpass`. Note that the total normalization depends both on the SED and the
        GSObject.  See the galsim.Chromatic docstring for more details on normalization
        conventions.

        @param target_magnitude  The desired *relative* magnitude of the SED.
        @param bandpass          A Bandpass object defining a filter bandpass.

        @returns the new normalized SED.
        """
        current_magnitude = self.calculateMagnitude(bandpass)
        norm = 10**(-0.4*(target_magnitude - current_magnitude))
        ret = self.copy()
        ret._rest_photons = lambda w: self._rest_photons(w) * norm
        return ret

    def atRedshift(self, redshift):
        """ Return a new SED with redshifted wavelengths.

        @param redshift

        @returns the redshifted SED.
        """
        ret = self.copy()
        ret.redshift = redshift
        wave_factor = (1.0 + redshift) / (1.0 + self.redshift)
        ret.wave_list = self.wave_list * wave_factor
        if ret.blue_limit is not None:
            ret.blue_limit = self.blue_limit * wave_factor
        if ret.red_limit is not None:
            ret.red_limit = self.red_limit * wave_factor
        return ret

    def calculateFlux(self, bandpass):
        """ Return the SED flux through a Bandpass `bandpass`.

        @param bandpass   A Bandpass object representing a filter, or None to compute the
                          bolometric flux.  For the bolometric flux the integration limits will be
                          set to (0, infinity) unless overridden by non-`None` SED attributes
                          `blue_limit` or `red_limit`.  Note that SEDs defined using
                          `interp1d`s automatically have `blue_limit` and `red_limit` set.

        @returns the flux through the bandpass.
        """
        if bandpass is None:  # do bolometric flux
            if self.blue_limit is None:
                blue_limit = 0.0
            else:
                blue_limit = self.blue_limit
            if self.red_limit is None:
                red_limit = np.inf  # = infinity in quad
            else:
                red_limit = self.red_limit
            return quad(self._rest_photons, blue_limit, red_limit)
        else:  # do flux through bandpass
            if len(bandpass.wave_list) > 0 or len(self.wave_list) > 0:
                x = np.union1d(bandpass.wave_list, self.wave_list)
                x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
                return np.trapz(bandpass(x) * self(x), x)
            else:
                return quad(lambda w: bandpass(w)*self(w),
                            bandpass.blue_limit, bandpass.red_limit)

    def calculateMagnitude(self, bandpass):
        """ Return the SED magnitude through a Bandpass `bandpass`.  Note that this requires
        `bandpass` to have been assigned a zeropoint using `Bandpass.withZeropoint()`.

        @param bandpass   A Bandpass object representing a filter, or None to compute the
                          bolometric magnitude.  For the bolometric magnitude the integration
                          limits will be set to (0, infinity) unless overridden by non-`None` SED
                          attributes `blue_limit` or `red_limit`.  Note that SEDs defined using
                          `interp1d`s automatically have `blue_limit` and `red_limit` set.

        @returns the bandpass magnitude.
        """
        current_flux = self.calculateFlux(bandpass)
        return -2.5 * np.log10(current_flux) + bandpass.zeropoint

    def thin(self, rel_err=1.e-4, preserve_range=False):
        """ If the SED was initialized with a interp1d or from a file (which internally creates a
        interp1d), then remove tabulated values while keeping the integral over the set of
        tabulated values still accurate to `rel_err`.

        @param rel_err            The relative error allowed in the integral over the SED
                                  [default: 1.e-4]
        @param preserve_range     Should the original range (`blue_limit` and `red_limit`) of the
                                  SED be preserved? (True) Or should the ends be trimmed to
                                  include only the region where the integral is significant? (False)
                                  [default: False]

        @returns the thinned SED.
        """
        if len(self.wave_list) > 0:
            wave_factor = 1.0 + self.redshift
            x = np.array(self.wave_list) / wave_factor
            f = self._rest_photons(x)
            newx, newf = thin_tabulated_values(x, f, rel_err=rel_err, preserve_range=preserve_range)
            ret = self.copy()
            ret.blue_limit = np.min(newx) * wave_factor
            ret.red_limit = np.max(newx) * wave_factor
            ret.wave_list = np.array(newx) * wave_factor
            ret._rest_photons = interp1d(newx, newf)
            return ret

    def calculateDCRMomentShifts(self, bandpass, **kwargs):
        """ Calculates shifts in first and second moments of PSF due to differential chromatic
        refraction (DCR).  I.e., equations (1) and (2) from Plazas and Bernstein (2012)
        (http://arxiv.org/abs/1204.1346).

        @param bandpass             Bandpass through which object is being imaged.
        @param zenith_angle         Angle from object to zenith, in radians
        @param parallactic_angle    Parallactic angle, i.e. the position angle of the zenith,
                                    in radians measured from North through East.  [default: 0]
        @param obj_coord            Celestial coordinates of the object being drawn as a
                                    CelestialCoord. [default: None]
        @param zenith_coord         Celestial coordinates of the zenith as a CelestialCoord.
                                    [default: None]
        @param HA                   Hour angle of the object as an Angle. [default: None]
        @param latitude             Latitude of the observer as an Angle. [default: None]
        @param pressure             Air pressure in kiloPascals.  [default: 69.328 kPa]
        @param temperature          Temperature in Kelvins.  [default: 293.15 K]
        @param H2O_pressure         Water vapor pressure in kiloPascals.  [default: 1.067 kPa]

        @returns a tuple.  The first element is the vector of DCR first moment shifts, and the
                 second element is the 2x2 matrix of DCR second (central) moment shifts.
        """
        if 'zenith_angle' in kwargs:
            zenith_angle = kwargs.pop('zenith_angle')
            parallactic_angle = kwargs.pop('parallactic_angle', 0.0)
        else:
            raise TypeError(
                "Need to specify zenith_angle and parallactic_angle in calculateDCRMomentShifts!")
        # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
        # Check that they're valid
        for kw in kwargs.keys():
            if kw not in ['temperature', 'pressure', 'H2O_pressure']:
                raise TypeError("Got unexpected keyword in calculateDCRMomentShifts: {0}".format(kw))
        # Now actually start calculating things.
        flux = self.calculateFlux(bandpass)
        if len(bandpass.wave_list) > 0:
            x = np.union1d(bandpass.wave_list, self.wave_list)
            x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
            R = dcr.get_refraction(x, zenith_angle, **kwargs)
            photons = self(x)
            throughput = bandpass(x)
            Rbar = np.trapz(throughput * photons * R, x) / flux
            V = np.trapz(throughput * photons * (R-Rbar)**2, x) / flux
        else:
            weight = lambda w: bandpass(w) * self(w)
            Rbar_kernel = lambda w: dcr.get_refraction(w, zenith_angle, **kwargs)
            Rbar = quad(lambda w: weight(w) * Rbar_kernel(w),
                        bandpass.blue_limit, bandpass.red_limit)
            V_kernel = lambda w: (galsim.dcr.get_refraction(w, zenith_angle, **kwargs) - Rbar)**2
            V = quad(lambda w: weight(w) * V_kernel(w),
                     bandpass.blue_limit, bandpass.red_limit)
        # Rbar and V are computed above assuming that the parallactic angle is 0.  Hence we
        # need to rotate our frame by the parallactic angle to get the desired output.
        rot = np.matrix([[np.cos(parallactic_angle), -np.sin(parallactic_angle)],
                         [np.sin(parallactic_angle), np.cos(parallactic_angle)]])
        Rbar = rot * Rbar * np.matrix([0,1]).T
        V = rot * np.matrix([[0, 0], [0, V]]) * rot.T
        return Rbar, V

    def calculateSeeingMomentRatio(self, bandpass, alpha=-0.2, base_wavelength=500):
        """ Calculates the relative size of a PSF compared to the monochromatic PSF size at
        wavelength `base_wavelength`.

        @param bandpass             Bandpass through which object is being imaged.
        @param alpha                Power law index for wavelength-dependent seeing.  [default:
                                    -0.2, the prediction for Kolmogorov turbulence]
        @param base_wavelength      Reference wavelength in nm from which to compute the relative
                                    PSF size.  [default: 500]
        @returns the ratio of the PSF second moments to the second moments of the reference PSF.
        """
        flux = self.calculateFlux(bandpass)
        if len(bandpass.wave_list) > 0:
            x = np.union1d(bandpass.wave_list, self.wave_list)
            x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
            photons = self(x)
            throughput = bandpass(x)
            return np.trapz(photons * throughput * (x/base_wavelength)**(2*alpha), x) / flux
        else:
            weight = lambda w: bandpass(w) * self(w)
            kernel = lambda w: (w/base_wavelength)**(2*alpha)
            return quad(lambda w: weight(w) * kernel(w),
                        bandpass.blue_limit, bandpass.red_limit) / flux

    def calculateLinearMomentShift(self, bandpass, slope, base_wavelength=500):
        """ Calculates the integral:
        \int{F(w) S(w) w (w - base_wavelength)*slope dw} / \int{F(w) S(w) w dw}

        @param bandpass         Bandpass through which object is being imaged.
        @param slope            dIxx/dw or dIyy/dw in square-radians per nanometer
        @param base_wavelength  Reference wavelength in nm from which to change PSF moment

        @returns the above integral
        """
        flux = self.calculateFlux(bandpass)
        if len(bandpass.wave_list) > 0:
            x = np.union1d(bandpass.wave_list, self.wave_list)
            x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
            R = (x - base_wavelength) * slope
            photons = self(x)
            throughput = bandpass(x)
            return np.trapz(throughput * photons * R, x) / flux
        else:
            weight = lambda w: bandpass(w) * self(w)
            kernel = lambda w: (w - base_wavelength)*slope
            return 1./flux * quad(lambda w: weight(w) * kernel(w),
                                  bandpass.blue_limit, bandpass.red_limit)

    def redden(self, A_v, R_v=3.1):
        """ Return a new SED with the specified extinction applied.  Note that this will
        truncate the wavelength range to lie between 91 nm and 6000 nm where the extinction
        correction law is defined.
        """
        return self / (lambda w: extinction.reddening(w * 10, a_v=A_v, r_v=R_v, model='f99'))

    def getDCRAngleDensity(self, bandpass, zenith, **kwargs):
        """Return photon density per unit refraction angle through a given filter.
        """
        wave = bandpass.interp.x
        photons = bandpass.interp.y * self(wave)
        R = dcr.get_refraction(wave, zenith, **kwargs)
        dR = np.diff(R)
        dwave = np.diff(wave)
        dwave_dR = dwave / dR  # Jacobian
        dwave_dR = np.append(dwave_dR, dwave_dR[-1])  # fudge the last array element
        angle_dens = photons * np.abs(dwave_dR)
        return R, angle_dens

    def addEmissionLines(self):
        # get UV continuum flux
        UV_fphot = self(230.0)  # photons / nm / s
        h = 6.62e-27  # ergs / Hz
        UV_fnu = UV_fphot * h * (230.0)  # converted to erg/s/Hz

        # construct line function
        lines = ['OII', 'OIII', 'Hbeta', 'Halpha', 'Lya']
        multipliers = np.r_[1.0, 0.36, 0.61, 1.77, 2.0] * 1.0e13
        waves = [372.7, 500.7, 486.1, 656.3, 121.5]  # nm
        velocity = 200.0  # km/s

        def lines(w):
            out = np.zeros_like(w, dtype=np.float64)
            for lm, lw in zip(multipliers, waves):
                line_flux = UV_fnu * lm  # ergs / sec
                hc = 1.986e-9  # erg nm
                line_flux *= lw / hc  # converted to phot / sec
                sigma = velocity / 299792.458 * lw  # sigma in Angstroms
                amplitude = line_flux / sigma / np.sqrt(2.0 * np.pi)
                out += amplitude * np.exp(-(lw-w)**2/(2*sigma**2))
            return out

        return self + SED(lines, flux_type='fphotons')

        # wave = self.interp.x
        # fphot = self.interp.y

        # # then add lines appropriately
        # lines = ['OII','OIII','Hbeta','Halpha','Lya']
        # multipliers = np.r_[1.0, 0.36, 0.61, 1.77, 2.0] * 1.0e13
        # waves = [372.7, 500.7, 486.1, 656.3, 121.5] # nm
        # velocity = 200.0 # km/s
        # for line, m, w in zip(lines, multipliers, waves):
        #     line_flux = UV_fnu * m # ergs / sec
        #     hc = 1.986e-9 # erg nm
        #     line_flux *= w / hc # converted to phot / sec
        #     sigma = velocity / 299792.458 * w # sigma in Angstroms
        #     amplitude = line_flux / sigma / np.sqrt(2.0 * np.pi)
        #     fphot += amplitude * np.exp(-(wave-w)**2/(2*sigma**2))
        # ret = self.copy()
        # ret.interp = interp1d(wave, fphot)
        # return ret


class Bandpass(object):
    """Simple bandpass object, which models the transmission fraction of incident light as a
    function of wavelength, for either an entire optical path (e.g., atmosphere, reflecting and
    refracting optics, filters, CCD quantum efficiency), or some intermediate piece thereof.
    Bandpasses representing individual components may be combined through the `*` operator to form
    a new Bandpass object representing the composite optical path.

    Bandpasses are callable, returning dimensionless throughput as a function of wavelength in nm.

    Bandpasses are immutable; all transformative methods return *new* Bandpasses, and leave their
    originating Bandpasses unaltered.

    Bandpasses require `blue_limit` and `red_limit` attributes, which may either be explicitly set
    at initialization, or are inferred from the initializing galsim.LookupTable or 2-column file.

    Outside of the wavelength interval between `blue_limit` and `red_limit`, the throughput is
    returned as zero, regardless of the `throughput` input parameter.

    Bandpasses may be multiplied by other Bandpasses, functions, or scalars.

    The Bandpass effective wavelength is stored in the python property `effective_wavelength`. We
    use throughput-weighted average wavelength (which is independent of any SED) as our definition
    for effective wavelength.

    For Bandpasses defined using a LookupTable, a numpy.array of wavelengths, `wave_list`, defining
    the table is maintained.  Bandpasses defined as products of two other Bandpasses will define
    their `wave_list` as the union of multiplicand `wave_list`s, although limited to the range
    between the new product `blue_limit` and `red_limit`.  (This implementation detail may affect
    the choice of integrator used to draw ChromaticObjects.)

    The input parameter, throughput, may be one of several possible forms:
    1. a regular python function (or an object that acts like a function)
    2. a galsim.LookupTable
    3. a file from which a LookupTable can be read in
    4. a string which can be evaluated into a function of `wave`
       via `eval('lambda wave : '+throughput)`
       e.g. throughput = '0.8 + 0.2 * (wave-800)'

    The argument of `throughput` will be the wavelength in either nanometers (default) or
    Angstroms depending on the value of `wave_type`.  The output should be the dimensionless
    throughput at that wavelength.  (Note we use `wave` rather than `lambda`, since `lambda` is a
    python reserved word.)

    The argument `wave_type` specifies the units to assume for wavelength and must be one of
    'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
    is unimportant.  If these wavelength options are insufficient, please submit an issue to
    the GalSim github issues page: https://github.com/GalSim-developers/GalSim/issues

    Note that the `wave_type` parameter does not propagate into other methods of `Bandpass`.
    For instance, Bandpass.__call__ assumes its input argument is in nanometers.

    @param throughput   Function defining the throughput at each wavelength.  See above for
                        valid options for this parameter.
    @param blue_limit   Hard cut off of bandpass on the blue side. [default: None, but required
                        if throughput is not a LookupTable or file.  See above.]
    @param red_limit    Hard cut off of bandpass on the red side. [default: None, but required
                        if throughput is not a LookupTable or file.  See above.]
    @param wave_type    The units to use for the wavelength argument of the `throughput`
                        function. See above for details. [default: 'nm']
    """
    def __init__(self, throughput, blue_limit=None, red_limit=None, wave_type='nm',
                 _wave_list=None):
        # Note that `_wave_list` acts as a private construction variable that overrides the way that
        # `wave_list` is normally constructed (see `Bandpass.__mul__` below)

        # Figure out input throughput type.
        tp = throughput  # For brevity within this function
        if isinstance(tp, str):
            import os
            if os.path.isfile(tp):
                w, t = np.genfromtxt(tp).T
                tp = interp1d(w, t)
            else:
                # Evaluate the function somewhere to make sure it is valid before continuing on.
                if red_limit is not None:
                    test_wave = red_limit
                elif blue_limit is not None:
                    test_wave = blue_limit
                else:
                    # If neither `blue_limit` nor `red_limit` is defined, then the Bandpass should
                    # be able to be evaluated at any wavelength, so check.
                    test_wave = 700
                try:
                    tp = eval('lambda wave : ' + tp)
                    tp(test_wave)
                except:
                    raise ValueError(
                        "String throughput must either be a valid filename or something that " +
                        "can eval to a function of wave. Input provided: {0}".format(throughput))

        # Figure out wavelength type
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type '{0}'".format(wave_type))

        # Assign blue and red limits of bandpass
        if isinstance(tp, interp1d):
            if blue_limit is None:
                blue_limit = tp.x.min()
            if red_limit is None:
                red_limit = tp.x.max()
        else:
            if blue_limit is None or red_limit is None:
                raise AttributeError(
                    "red_limit and blue_limit are required if throughput is not a numpy.interp1d.")

        if blue_limit > red_limit:
            raise ValueError("blue_limit must be less than red_limit")
        self.blue_limit = blue_limit / wave_factor
        self.red_limit = red_limit / wave_factor

        # Sanity check blue/red limit and create self.wave_list
        if isinstance(tp, interp1d):
            self.wave_list = np.array(tp.x)/wave_factor
            # Make sure that blue_limit and red_limit are within interp1d region of support.
            if self.blue_limit < (tp.x.min()/wave_factor):
                raise ValueError("Cannot set blue_limit to be less than throughput " +
                                 "interp1d.x.min()")
            if self.red_limit > (tp.x.max()/wave_factor):
                raise ValueError("Cannot set red_limit to be greater than throughput " +
                                 "interp1d.x.max()")
            # Make sure that blue_limit and red_limit are part of wave_list.
            if self.blue_limit not in self.wave_list:
                np.insert(self.wave_list, 0, self.blue_limit)
            if self.red_limit not in self.wave_list:
                np.insert(self.wave_list, -1, self.red_limit)
        else:
            self.wave_list = np.array([], dtype=np.float)

        # Manual override!  Be careful!
        if _wave_list is not None:
            self.wave_list = _wave_list

        self.func = lambda w: tp(np.array(w) * wave_factor)

        self.zeropoint = None

    def __mul__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = self.wave_list

        if isinstance(other, (Bandpass, SED)):
            if len(other.wave_list) > 0:
                wave_list = np.union1d(wave_list, other.wave_list)
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        # product of Bandpass instance and Bandpass subclass instance
        if isinstance(other, Bandpass) and type(self) != type(other):
            ret = Bandpass(lambda w: other(w)*self(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)
        # otherwise, preserve type of self
        else:
            ret = self.copy()
            ret.blue_limit = blue_limit
            ret.red_limit = red_limit
            ret.wave_list = wave_list
            ret.zeropoint = None
            if hasattr(ret, '_effective_wavelength'):
                del ret._effective_wavelength  # this will get lazily recomputed when needed
            if hasattr(other, '__call__'):
                ret.func = lambda w: other(w)*self(w)
            else:
                ret.func = lambda w: other*self(w)
        return ret

    def __rmul__(self, other):
        return self*other

    # Doesn't check for divide by zero, so be careful.
    def __div__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = self.wave_list

        if isinstance(other, Bandpass):
            if len(other.wave_list) > 0:
                wave_list = np.union1d(wave_list, other.wave_list)
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        # product of Bandpass instance and Bandpass subclass instance
        if isinstance(other, Bandpass) and type(self) != type(other):
            ret = Bandpass(lambda w: self(w)/other(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)
        # otherwise, preserve type of self
        else:
            ret = self.copy()
            ret.blue_limit = blue_limit
            ret.red_limit = red_limit
            ret.wave_list = wave_list
            ret.zeropoint = None
            if hasattr(ret, '_effective_wavelength'):
                del ret._effective_wavelength  # this will get lazily recomputed when needed
            if hasattr(other, '__call__'):
                ret.func = lambda w: self(w)/other(w)
            else:
                ret.func = lambda w: self(w)/other
        return ret

    # Doesn't check for divide by zero, so be careful.
    def __rdiv__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = self.wave_list

        if isinstance(other, Bandpass):
            if len(other.wave_list) > 0:
                wave_list = np.union1d(wave_list, other.wave_list)
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        # product of Bandpass instance and Bandpass subclass instance
        if isinstance(other, Bandpass) and type(self) != type(other):
            ret = Bandpass(lambda w: other(w)/self(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)
        # otherwise, preserve type of self
        else:
            ret = self.copy()
            ret.blue_limit = blue_limit
            ret.red_limit = red_limit
            ret.wave_list = wave_list
            ret.zeropoint = None
            if hasattr(ret, '_effective_wavelength'):
                del ret._effective_wavelength  # this will get lazily recomputed when needed
            if hasattr(other, '__call__'):
                ret.func = lambda w: other(w)/self(w)
            else:
                ret.func = lambda w: other/self(w)
        return ret

    # Doesn't check for divide by zero, so be careful.
    def __truediv__(self, other):
        return self.__div__(other)

    # Doesn't check for divide by zero, so be careful.
    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __call__(self, wave):
        """ Return dimensionless throughput of bandpass at given wavelength in nanometers.

        Note that outside of the wavelength range defined by the `blue_limit` and `red_limit`
        attributes, the throughput is assumed to be zero.

        @param wave   Wavelength in nanometers.

        @returns the dimensionless throughput.
        """
        # figure out what we received, and return the same thing
        # option 1: a NumPy array
        if isinstance(wave, np.ndarray):
            wgood = (wave >= self.blue_limit) & (wave <= self.red_limit)
            ret = np.zeros(wave.shape, dtype=np.float)
            np.place(ret, wgood, self.func(wave[wgood]))
            return ret
        # option 2: a tuple
        elif isinstance(wave, tuple):
            return tuple([self.func(w) if (w >= self.blue_limit and w <= self.red_limit) else 0.0
                          for w in wave])
        # option 3: a list
        elif isinstance(wave, list):
            return [self.func(w) if (w >= self.blue_limit and w <= self.red_limit) else 0.0
                    for w in wave]
        # option 4: a single value
        else:
            return self.func(wave) if (wave >= self.blue_limit and wave <= self.red_limit) else 0.0

    @property
    def effective_wavelength(self):
        """ Calculate, store, and return the effective wavelength for this bandpass.  We define
        the effective wavelength as the throughput-weighted average wavelength, which is
        SED-independent.  Units are nanometers.
        """
        if not hasattr(self, '_effective_wavelength'):
            if len(self.wave_list) > 0:
                f = self.func(self.wave_list)
                self._effective_wavelength = (np.trapz(f * self.wave_list, self.wave_list) /
                                              np.trapz(f, self.wave_list))
            else:
                self._effective_wavelength = (quad(lambda w: self.func(w) * w,
                                                   self.blue_limit,
                                                   self.red_limit) /
                                              quad(self.func,
                                                   self.blue_limit,
                                                   self.red_limit))
        return self._effective_wavelength

    def withZeropoint(self, zeropoint, effective_diameter=None, exptime=None):
        """ Assign a zeropoint to this Bandpass.

        The first argument `zeropoint` can take a variety of possible forms:
        1. a number, which will be the zeropoint
        2. a galsim.SED.  In this case, the zeropoint is set such that the magnitude of the supplied
           SED through the bandpass is 0.0
        3. the string 'AB'.  In this case, use an AB zeropoint.
        4. the string 'Vega'.  Use a Vega zeropoint.
        5. the string 'ST'.  Use a HST STmag zeropoint.
        For 3, 4, and 5, the effective diameter of the telescope and exposure time of the
        observation are also required.

        @param zeropoint            see above for valid input options
        @param effective_diameter   Effective diameter of telescope aperture in cm. This number must
                                    account for any central obscuration, i.e. for a diameter d and
                                    linear obscuration fraction obs, the effective diameter is
                                    d*sqrt(1-obs^2). [default: None, but required if zeropoint is
                                    'AB', 'Vega', or 'ST'].
        @param exptime              Exposure time in seconds. [default: None, but required if
                                    zeropoint is 'AB', 'Vega', or 'ST'].
        @returns new Bandpass with zeropoint set.
        """
        if isinstance(zeropoint, str):
            if effective_diameter is None or exptime is None:
                raise ValueError("Cannot calculate Zeropoint from string {0} without " +
                                 "telescope effective diameter or exposure time.")
            if zeropoint.upper() == 'AB':
                AB_source = 3631e-23  # 3631 Jy in units of erg/s/Hz/cm^2
                c = 2.99792458e17  # speed of light in nm/s
                AB_flambda = AB_source * c / self.wave_list**2
                AB_sed = SED(interp1d(self.wave_list, AB_flambda))
                flux = AB_sed.calculateFlux(self)
            # If zeropoint.upper() is 'ST', then use HST STmags:
            # http://www.stsci.edu/hst/acs/analysis/zeropoints
            elif zeropoint.upper() == 'ST':
                ST_flambda = 3.63e-8  # erg/s/cm^2/nm
                ST_sed = SED(interp1d(self.wave_list, ST_flambda))
                flux = ST_sed.calculateFlux(self)
            # If zeropoint.upper() is 'VEGA', then load vega spectrum stored in repository,
            # and use that for zeropoint spectrum.
            # elif zeropoint.upper()=='VEGA':
            #     import os
            #     vegafile = os.path.join(galsim.meta_data.share_dir, "vega.txt")
            #     sed = galsim.SED(vegafile)
            #     flux = sed.calculateFlux(self)
            else:
                raise ValueError("Do not recognize Zeropoint string {0}.".format(zeropoint))
            flux *= np.pi*effective_diameter**2/4 * exptime
            new_zeropoint = 2.5 * np.log10(flux)
        # If `zeropoint` is an `SED`, then compute the SED flux through the bandpass, and
        # use this to create a magnitude zeropoint.
        elif isinstance(zeropoint, SED):
            flux = zeropoint.calculateFlux(self)
            new_zeropoint = 2.5 * np.log10(flux)
        # If zeropoint is a number, then use that
        elif isinstance(zeropoint, (float, int)):
            new_zeropoint = zeropoint
        # But if zeropoint is none of these, raise an exception.
        else:
            raise ValueError(
                "Don't know how to handle zeropoint of type: {0}".format(type(zeropoint)))
        ret = self.copy()
        ret.zeropoint = new_zeropoint
        return ret

    def truncate(self, blue_limit=None, red_limit=None, relative_throughput=None):
        """Return a bandpass with its wavelength range truncated.

        This function truncate the range of the bandpass either explicitly (with `blue_limit` or
        `red_limit` or both) or automatically, just trimming off leading and trailing wavelength
        ranges where the relative throughput is less than some amount (`relative_throughput`).

        This second option using relative_throughpt is only available for bandpasses initialized
        with a LookupTable or from a file, not when using a regular python function or a string
        evaluation.

        This function does not remove any intermediate wavelength ranges, but see thin() for
        a method that can thin out the intermediate values.

        @param blue_limit       Truncate blue side of bandpass here. [default: None]
        @param red_limit        Truncate red side of bandpass here. [default: None]
        @param relative_throughput  Truncate leading or trailing wavelengths that are below
                                this relative throughput level.  (See above for details.)
                                [default: None]

        @returns the truncated Bandpass.
        """
        if blue_limit is None:
            blue_limit = self.blue_limit
        if red_limit is None:
            red_limit = self.red_limit
        blue_limit = max([blue_limit, self.blue_limit])
        red_limit = min([red_limit, self.red_limit])

        if len(self.wave_list) > 0:
            if relative_throughput is not None:
                wave = np.array(self.wave_list)
                tp = self.func(wave)
                w = (tp >= tp.max()*relative_throughput).nonzero()
                blue_limit = max([min(wave[w]), blue_limit])
                red_limit = min([max(wave[w]), red_limit])
        elif relative_throughput is not None:
            raise ValueError(
                "Can only truncate with relative_throughput argument if throughput is " +
                "a LookupTable")
        # preserve type
        ret = self.copy()
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        if hasattr(ret, '_effective_wavelength'):
            del ret._effective_wavelength
        return ret

    def thin(self, rel_err=1.e-4, preserve_range=False):
        """Thin out the internal wavelengths of a Bandpass that uses a LookupTable.

        If the bandpass was initialized with a LookupTable or from a file (which internally
        creates a LookupTable), this function removes tabulated values while keeping the integral
        over the set of tabulated values still accurate to the given relative error.

        That is, the integral of the bandpass function is preserved to a relative precision
        of `rel_err`, while eliminating as many internal wavelength values as possible.  This
        process will usually help speed up integrations using this bandpass.  You should weigh
        the speed improvements against your fidelity requirements for your particular use
        case.

        @param rel_err            The relative error allowed in the integral over the throughput
                                  function. [default: 1.e-4]
        @param preserve_range     Should the original range (`blue_limit` and `red_limit`) of the
                                  Bandpass be preserved? (True) Or should the ends be trimmed to
                                  include only the region where the integral is significant? (False)
                                  [default: False]

        @returns the thinned Bandpass.
        """
        if len(self.wave_list) > 0:
            x = self.wave_list
            f = self(x)
            newx, newf = thin_tabulated_values(x, f, rel_err=rel_err,
                                               preserve_range=preserve_range)
            # preserve type
            ret = self.copy()
            ret.func = interp1d(newx, newf)
            ret.blue_limit = np.min(newx)
            ret.red_limit = np.max(newx)
            ret.wave_list = np.array(newx)
            if hasattr(ret, '_effective_wavelength'):
                del ret._effective_wavelength
            return ret


def thin_tabulated_values(x, f, rel_err=1.e-4, preserve_range=False):
    """
    Remove items from a set of tabulated f(x) values so that the error in the integral is still
    accurate to a given relative accuracy.

    The input `x,f` values can be lists, NumPy arrays, or really anything that can be converted
    to a NumPy array.  The new lists will be output as python lists.

    @param x                The `x` values in the f(x) tabulation.
    @param f                The `f` values in the f(x) tabulation.
    @param rel_err          The maximum relative error to allow in the integral from the removal.
                            [default: 1.e-4]
    @param preserve_range   Should the original range of `x` be preserved? (True) Or should the ends
                            be trimmed to include only the region where the integral is
                            significant? (False)  [default: False]

    @returns a tuple of lists `(x_new, y_new)` with the thinned tabulation.
    """
    x = np.array(x)
    f = np.array(f)

    # Check for valid inputs
    if len(x) != len(f):
        raise ValueError("len(x) != len(f)")
    if rel_err <= 0 or rel_err >= 1:
        raise ValueError("rel_err must be between 0 and 1")
    if not (np.diff(x) >= 0).all():
        raise ValueError("input x is not sorted.")

    # Check for trivial noop.
    if len(x) <= 2:
        # Nothing to do
        return

    # Start by calculating the complete integral of |f|
    total_integ = np.trapz(abs(f), x)
    if total_integ == 0:
        return np.array([x[0], x[-1]]), np.array([f[0], f[-1]])
    thresh = rel_err * total_integ

    if not preserve_range:
        # Remove values from the front that integrate to less than thresh.
        integ = 0.5 * (abs(f[0]) + abs(f[1])) * (x[1] - x[0])
        k0 = 0
        while k0 < len(x)-2 and integ < thresh:
            k0 = k0+1
            integ += 0.5 * (abs(f[k0]) + abs(f[k0+1])) * (x[k0+1] - x[k0])
        # Now the integral from 0 to k0+1 (inclusive) is a bit too large.
        # That means k0 is the largest value we can use that will work as the staring value.

        # Remove values from the back that integrate to less than thresh.
        k1 = len(x)-1
        integ = 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        while k1 > k0 and integ < thresh:
            k1 = k1-1
            integ += 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        # Now the integral from k1-1 to len(x)-1 (inclusive) is a bit too large.
        # That means k1 is the smallest value we can use that will work as the ending value.

        x = x[k0:k1+1]  # +1 since end of range is given as one-past-the-end.
        f = f[k0:k1+1]

    # Start a new list with just the first item so far
    newx = [x[0]]
    newf = [f[0]]

    k0 = 0  # The last item currently in the new array
    k1 = 1  # The current item we are considering to skip or include
    while k1 < len(x)-1:
        # We are considering replacing all the true values between k0 and k1+1 (non-inclusive)
        # with a linear approxmation based on the points at k0 and k1+1.
        lin_f = f[k0] + (f[k1+1]-f[k0])/(x[k1+1]-x[k0]) * (x[k0:k1+2] - x[k0])
        # Integrate | f(x) - lin_f(x) | from k0 to k1+1, inclusive.
        integ = np.trapz(abs(f[k0:k1+2] - lin_f), x[k0:k1+2])
        # If the integral of the difference is < thresh, we can skip this item.
        if integ < thresh:
            # OK to skip item k1
            k1 = k1 + 1
        else:
            # Have to include this one.
            newx.append(x[k1])
            newf.append(f[k1])
            k0 = k1
            k1 = k1 + 1

    # Always include the last item
    newx.append(x[-1])
    newf.append(f[-1])

    return newx, newf
