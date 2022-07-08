import numpy

def air_refractive_index(wave, pressure=69.328, temperature=293.15, H2O_pressure=1.067):
    """Return the refractive index of air as function of wavelength.

    Uses the formulae given in Filippenko (1982), which appears to come from Edlen (1953),
    and Coleman, Bozman, and Meggers (1960).  The units of the original formula are non-SI,
    being mmHg for pressure (and water vapor pressure), and degrees C for temperature.  This
    function accepts SI units however, and transforms them when plugging in the formula.
    Default values for `pressure`, `temperature`, and `H2O_pressure` are taken from LSST PhoSim
    defaults.

    Arguments
    ---------
    wave -- wavelength in nanometers
    pressure -- in kiloPascals (default 69.328 kPa = 520 mmHg)
    temperature -- in Kelvin (default 293.15 kPa = 20 C)
    H2O_pressure -- in kiloPascals (default 1.067 kPa = 8 mmHg)
    """

    sigma_squared = 1.0 / (wave * 1.e-3)**2.0 # inverse wavenumber squared in um^-2

    n_minus_one = (64.328 + (29498.1 / (146.0 - sigma_squared))
                   + (255.4 / (41.0 - sigma_squared))) * 1.e-6
    P = pressure * 7.50061683 # kPa -> mmHg
    T = temperature - 273.15 # K -> C
    W = H2O_pressure * 7.50061683 # kPa -> mmHg
    n_minus_one *= P * (1.0 + (1.049 - 0.0157 * T) * 1.e-6 * P) / (720.883 * (1.0 + 0.003661 * T))
    n_minus_one -= (0.0624 - 0.000680 * sigma_squared)/(1.0 + 0.003661 * T) * W * 1.e-6
    return n_minus_one + 1.0

def air_refractive_index2(wave, pressure=69.328, temperature=293.15, H2O_pressure=1.067):
    """Return the refractive index of air as function of wavelength.

    Uses the formulae given by Allens Astrophysical Quantities (Cox et al. 2001), including effects
    due to `pressure`, `temperature`, and the partial pressure of water vapor: `H2O_pressure`.
    Default values for `pressure`, `temperature`, and `H2O_pressure` are taken from LSST PhoSim
    defaults.

    Arguments
    ---------
    wave -- wavelength in nanometers
    pressure -- in kiloPascals (default 69.328 kPa = 520 mmHg)
    temperature -- in Kelvin (default 293.15 kPa = 20 C)
    H2O_pressure -- in kiloPascals (default 1.067 kPa = 8 mmHg)
    """

    sigma_squared = 1.0 / wave**2.0
    n_minus_one = (64.328 + (29498.1e-6 / (146e-6 - sigma_squared))
                   + (255.4e-6 / (41e-6 - sigma_squared))) * 1.e-6
    p_ref = 101.325 # kPA
    t_ref = 288.15 # K
    n_minus_one *= (pressure / p_ref) / (temperature / t_ref)
    n_minus_one -= 43.49e-6 * (1 - 7.956e3 * sigma_squared) * H2O_pressure / p_ref
    return n_minus_one + 1.0

def get_refraction(wave, zenith, **kwargs):
    """Compute refraction angle (in radians) from space to atmosphere.

    Uses formulae from Allen's Astrophysical Quantities (Cox et al. 2001).  Result depends on the
    inpute wavelength `wave` and the zenith angle `zenith`.  Only valid for zenith angles less than
    ~80 degrees.

    Arguments
    ---------
    wave -- wavelength in nanometers
    zenith -- the zenith angle of the incoming photon (actual or refracted?) in radians.

    **kwargs
    --------
    pressure, temperature, H2O_pressure forwarded to air_refractive_index()
    """

    n_squared = air_refractive_index(wave, **kwargs)**2.0
    r0 = (n_squared - 1.0) / (2.0 * n_squared)
    return r0 * numpy.tan(zenith)
