''' Version 0.0 of a class to handle phosim objects.  The most useful
method at the moment is the ability to predict the value of the phosim
normalization which will produce a given number of output photons.
The airmass dependence of this is not tested, but should be relatively
weak.

TODO
----

phosim_dir is hardcoded.  There has to be a better way.

Should probably use SED and Bandpass classes.  For instance, the ones
from eLSST at Washington?

Dust stuff not implemented.

Shape parameters not implemented.

AB_mag method should respect mag_norm.

Is it possible to use filter data built into phosim directory
structure?  Is the atmosphere extinction present somewhere?
'''

import numpy as np

class PhosimObject(object):
    ''' A class to build and output phosim instance catalog objects.
    Mostly useful right now to solve the normalization problem, where
    phosim chooses to normalize by the AB magnitude at 500 nm
    (rest-frame?), but what we usually desire is a broadband filter
    magnitude.
    '''
    def __init__(self, _id=None, RA=None, dec=None, mag_norm=None, SED_name=None,
                 redshift=0.0, gamma1=0.0, gamma2=0.0, mu=0.0,
                 delta_RA=0.0, delta_dec=0.0, source_type='point',
                 spatial_pars=None, dust_rest_name=None, dust_rest_pars=None,
                 dust_obs_name=None, dust_obs_pars=None):
        self._id = _id
        self.RA = RA
        self.dec = dec
        self.mag_norm = mag_norm
        self.SED_name = SED_name
        self.redshift = redshift
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        self.delta_RA = delta_RA
        self.delta_dec = delta_dec
        self.source_type = source_type
        self.spatial_pars = spatial_pars
        self.dust_rest_name = dust_rest_name
        self.dust_rest_pars = dust_rest_pars
        self.dust_obs_name = dust_obs_name
        self.dust_obs_pars = dust_obs_pars
        self.phosim_dir = PhosimObject.phosim_dir
        self._current_redshift = None
        self._current_SED_name = None
        self._current_mag_norm = None
        self._current_SED_wave = None
        self._current_SED_flux = None

    import socket
    if socket.gethostname() == 'Trogdor':
        phosim_dir = '/Users/josh/phosim-3.2.1/'
    else:
        phosim_dir = '/nfs/slac/g/ki/ki19/jmeyers3/phosim-3.2.1/'

    def get_SED(self):
        '''Returns the redshifted wavelength array and flux array of
        object.  Loads data on first call and caches result.  Updates
        cache if redshift or SED_name have been changed.
        '''
        if (self._current_redshift != self.redshift) or (self._current_SED_name != self.SED_name):
            self._current_redshift = self.redshift
            self._current_SED_name = self.SED_name
            self._current_mag_norm = self.mag_norm
            SED_data = np.genfromtxt(self.phosim_dir+'data/SEDs/'+self.SED_name)
            self._SED_wave = SED_data[:,0] * (1.0 + self.redshift)
            self._SED_flambda = SED_data[:,1]
        return self._SED_wave, self._SED_flambda


    def _AB(self, SED_wave, flambda, wave0):
        '''Returns the AB magnitude at wave0 (nm) of spectrum
        specified by SED_wave, flambda.
        '''
        c = 2.99792458e18 # Angstrom Hz
        fnu = flambda * (SED_wave * 10)**2 / c # wave from nm -> Angstrom
        AB = -2.5 * np.log10(np.interp(wave0, SED_wave, fnu)) - 48.6
        return AB

    def AB_mag(self, wavenm):
        '''Returns the AB magnitude of the (redshifted) spectrum at
        wavelength wavenm (specified in nanometers).
        '''
        SED_wave, flambda = self.get_SED() # nm, erg/s/cm2/A
        return self._AB(SED_wave, flambda, wavenm)

    def AB_mag_rest(self, wavenm):
        '''Returns the AB magnitude of the (unredshifted) spectrum at
        wavelength wavenm (specified in nanometers).
        '''
        SED_wave, flambda = self.get_SED() # nm, erg/s/cm2/A
        SED_wave_rest = SED_wave / (1.0 + self.redshift) # Need to unredshift wavelength array...
        return self._AB(SED_wave_rest, flambda, wavenm)

    def integrated_flux(self, filter_wave, filter_throughput, exp_time=15.0, eff_diameter=670):
        '''Integrates product of SED and filter throughput, and
        multiplies by typical LSST exposure time (in seconds) and
        collecting area (specified by effective diameter in cm) to
        estimate the number of photons collected by CCD.
        '''
        SED_wave, flambda = self.get_SED()
        wave_union = np.union1d(SED_wave, filter_wave) #note union1d sorts its output
        flambda_i = np.interp(wave_union, SED_wave, flambda)
        throughput_i = np.interp(wave_union, filter_wave, filter_throughput)
        dwave = np.diff(wave_union)
        dwave = np.append(dwave, dwave[-1])
        dwave *= 10 # nm -> Ang
        hc = 1.98644521e-9 # (Plancks_constant * speed_of_light) in erg nm
        photon_rate = (flambda_i * throughput_i * wave_union * dwave / hc).sum() # photons/sec/cm2
        return photon_rate * np.pi * (eff_diameter / 2)**2 * exp_time # photons!

    def set_flux_goal(self, goal, filter_wave, filter_throughput):
        '''phosim normalization is AB at 500 nm of the unredshifted
        spectrum, with a (1+z) factor inserted.  Josh is not sure
        where the (1+z) factor comes from, but this is empirically
        supported from running phosim.
        '''
        AB = self.AB_mag_rest(500.0) - 2.5 * np.log10(1.0 + self.redshift)
        flux = self.integrated_flux(filter_wave, filter_throughput)
        self.mag_norm = AB + 2.5 * np.log10(flux * 0.805 / goal) # 0.805 = fudge factor
        return self.mag_norm

    def out_string(self):
        obj_format = 'object {} {} {} {} {} {} {} {} {} {} {}'
        out_string = obj_format.format(self._id, self.RA,
                                       self.dec, self.mag_norm,
                                       self.SED_name, self.redshift,
                                       self.gamma1, self.gamma2,
                                       self.mu,
                                       self.delta_RA, self.delta_dec)
        out_string += ' star none none' #HACK! for now
        return out_string
