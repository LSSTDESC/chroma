import scipy.integrate

def _AB(wave, flambda, AB_wave):
    """Returns the AB magnitude at `AB_wave` (nm) of spectrum specified by
    `wave` (in nm), and `flambda` (in erg/s/cm^2/Ang).
    """
    speed_of_light = 2.99792458e18 # units are Angstrom Hz
    fNu = flambda * (wave * 10)**2 / speed_of_light # `* 10` is wave from nm -> Angstrom
    AB = -2.5 * numpy.log10(numpy.interp(AB_wave, wave, fNu)) - 48.6
    return AB

def _integrated_flux(wave, flambda, f_wave, f_throughput, exptime=15.0, eff_diam=670):
    """Integrates product of SED and filter throughput, and multiplies
    by typical LSST exposure time (in seconds) and collecting area
    (specified by effective diameter in cm) to estimate the number of
    photons collected by CCD.

    Units
    -----

    wave, f_wave : nm (corresponds to SED and filter resp.)
    flambda : erg/s/cm^2/Ang
    f_throughput : dimensionless
    exptime : seconds
    eff_diam : cm (effective diameter of aperture)
    """
    wave_union = numpy.union1d(wave, f_wave) #note union1d sorts its output
    flambda_i = numpy.interp(wave_union, wave, flambda)
    throughput_i = numpy.interp(wave_union, f_wave, f_throughput)

    hc = 1.98644521e-9 # (PlanckConstant * speedOfLight) in erg nm
    integrand = flambda_i * throughput_i * wave_union / hc
    differential = wave_union * 10 # nm -> Ang
    photon_rate = scipy.integrate.simps(integrand, differential)
    return photon_rate * numpy.pi * (eff_diam / 2)**2 * exptime # total photons


def mag_norm_to_LSST_flux(SED_file, filter_file, mag_norm, redshift=0.0):
    """Predict LSST PhoSim flux (in total number of collected photons)
    for an object with SED specified by `SED_file` through a filter
    specified by `filter_file`, and a PhoSim normalization of `mag_norm`.

    The format of the SED_file is 2 columns with first column the
    wavelength in nm, and the second column the flambda flux in
    erg/s/cm2/Ang.

    The format of the filter_file is 2 columns with first column the
    wavelength in nm and the second column the throughput (assumed to
    be everything: sky, filter, CCD, etc.) in fraction of surviving
    photons.
    """
    SED_data = numpy.genfromtxt(SED_file)
    wave, flambda = SED_data[:,0], SED_data[:,1]
    f_data = numpy.genfromtxt(filter_file)
    f_wave, f_throughput = filter_data[:,0], filter_data[:,1]

    AB = _AB(wave, flambda, 500.0)
    flux = _integrated_flux(wave * (1.0 + redshift), flambda / (1.0 + redshift), f_wave, f_throughput)
    return flux * 10**(-0.4 * (mag_norm - AB)) * 0.805 #empirical fudge factor!
