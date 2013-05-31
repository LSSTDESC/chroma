import numpy
import scipy.integrate

def air_refractive_index(wave, pressure=69.328, temperature=293.15, H2O_pressure=1.067):
    '''Return the refractive index of air as function of wavelength.

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
    '''

    sigma_squared = 1.0 / wave**2.0
    n_minus_one = (64.328 + (29498.1e-6 / (146e-6 - sigma_squared))
                   + (255.4e-6 / (41e-6 - sigma_squared))) * 1.e-6
    p_ref = 101.325 #kPA
    t_ref = 288.15 #K
    n_minus_one *= (pressure / p_ref) / (temperature / t_ref)
    n_minus_one -= 43.49e-6 * (1 - 7.956e3 * sigma_squared) * H2O_pressure / p_ref
    return n_minus_one + 1.0

def atm_refrac(wave, zenith, **kwargs):
    '''Compute refraction angle (in radians) from space to atmosphere.

    Uses formulae from Allen's Astrophysical Quantities (Cox et al. 2001).  Result depends on the
    inpute wavelength `wave` and the zenith angle `zenith`.  Only valid for zenith angles less than
    ~80 degrees.

    Arguments
    ---------
    wave -- wavelength in nanometers
    zenith -- the zenith angle of the incoming photon (actual or refracted?) in radians.

    **kwargs
    --------
    pressure, temperature, partialPressureWater forwarded to air_refractive_index()
    '''

    n_squared = air_refractive_index(wave, **kwargs)**2.0
    r0 = (n_squared - 1.0) / (2.0 * n_squared)
    return r0 * numpy.tan(zenith)

def disp_moments(wave, photons, **kwargs):
    '''Compute the first and second central moments of refraction distribution from SED.

    The `photons` is the relative density of photons at wavelengths in `wave`.  This should be the
    surviving photon distribution, i.e. an SED multiplied by a filter throughput function, and by
    the wavelength to convert ergs -> photons.  The moments are then computed as integrals over
    wavelength as in Plazas and Bernstein (2012).

    Arguments
    ---------
    wave -- wavelength array in nanometers
    photons -- object SED*throughput*wave.  Units proportional to photons/sec/cm^2/A

    **kwargs
    --------
    zenith -> atm_refrac()
    pressure, temperature, H2O_pressure -> air_refractive_index()

    Returns
    -------
    (M1, M2) -- tuple containing first moment (mean) and second central moment (variance)
    '''

    R = atm_refrac(wave, **kwargs)
    norm = scipy.integrate.simps(photons, wave)
    Rbar = scipy.integrate.simps(photons * R, wave) / norm
    V = scipy.integrate.simps(photons * ((R - Rbar)**2.0), wave) / norm
    return Rbar, V

def wave_dens_to_angle_dens(wave, wave_dens, **kwargs):
    '''Utility to convert f_lambda to f_R.

    Converts a spectral density object with units of blah/Angstrom to an object with units of
    blah/radian.  For example, a generic SED usually will be given in units proportional to
    erg/s/cm^2/A, i.e. energy intensity per unit wavelength.  This function will compute the
    refraction for each wavelength, and then convert the spectrum into an object with units of
    erg/s/cm^2/rad, i.e. energy intensity per unit refraction angle.  This is useful for convolving
    with a (zenith) atmospheric PSF to estimate the PSF (away from zenith) including dispersion.

    Arguments
    ---------
    wave -- wavelength in nanometers
    wave_dens -- object with units of something/A

    **kwargs
    --------
    zenith -> atm_disp()
    pressure, temperature, H2O_pressure -> air_refractive_index()

    Returns
    -------
    R -- refraction for each input wavelength in radians
    angledens -- the rebinned density in something/radian
    '''

    R = atm_refrac(wave, **kwargs)
    dR = numpy.diff(R)
    dwave = numpy.diff(wave)
    dwave_dR = dwave / dR #Jacobian
    dwave_dR = numpy.append(dwave_dR, dwave_dR[-1]) #fudge the last array element
    angle_dens = wave_dens * numpy.abs(dwave_dR)
    return R, angle_dens

def disp_moments_R(wave, photons, **kwargs):
    '''Same as disp_moments, but integrates against refraction instead of wavelength; sanity check'''
    R, photons_per_dR = wave_dens_to_angle_dens(wave, photons, **kwargs)
    norm = scipy.integrate.simps(photons_per_dR, R)
    Rbar = scipy.integrate.simps(R * photons_per_dR, R)/norm
    V = scipy.integrate.simps((R - Rbar)**2.0 * photons_per_dR, R)/norm
    return Rbar, V

def weighted_second_moment(wave, photons, sigma, Rbar=None, V=None, **kwargs):
    def moffat1d(FWHM, beta):
        alpha = FWHM / (2.0 * numpy.sqrt(2.0**(1.0 / beta) - 1.0))
        def f(x):
            u = (x/alpha)**2.0
            p = 1.0 / ((u + 1.0)**beta)
            return p / p.max()
        return f

    def gaussian1d(sigma, center):
        return lambda x: numpy.exp(-0.5 * ((x - center) / sigma)**2.0)

    def zenith_PSF(wave, flux, moffat_FWHM=0.705, moffat_beta=2.67,
                   Rbar=None, V=None,
                   **kwargs):
        # the dispersion contribution to the PSF
        R, photons_per_dR = wave_dens_to_angle_dens(wave, photons, **kwargs)
        asort = R.argsort()
        R, photons_per_dR = R[asort], photons_per_dR[asort]
        #scale output range using larger of moffatFWHM and FWHM of dispersion component
        if Rbar is None or V is None:
            M = disp_moments(wave, photons, **kwargs)
            Rbar = M[0]
            V = M[1]
        #scale in radians
        scale = numpy.sqrt(numpy.log(256.0) * V) # radians
        moffat_FWHM_rad = moffat_FWHM * numpy.pi / 180.0 / 3600
        if moffat_FWHM_rad > scale : scale = moffat_FWHM_rad
        r_min = Rbar - 2.5 * scale #radians
        r_max = Rbar + 2.5 * scale #radians
        R_fine = numpy.arange(r_min, r_max, 0.001 * numpy.pi / 180.0 / 3600) #radians
        photons_per_dR_fine = numpy.interp(R_fine, R, photons_per_dR)
        #check units below
        moffat = moffat1d(moffat_FWHM_rad, moffat_beta)
        moffat_PSF = moffat(R_fine - 0.5 * (R_fine[0] + R_fine[-1])) #center in window
        moffat_PSF /= moffat_PSF.sum()
        zen_PSF = numpy.convolve(photons_per_dR_fine, moffat_PSF, mode="same")
        return R_fine, zen_PSF

    if Rbar is None or V is None:
        M = disp_moments(wave, photons, **kwargs)
        Rbar = M[0]
        V = M[1]
    R, zen_PSF = zenith_PSF(wave, photons, Rbar=Rbar, V=V, **kwargs)
    sigma_rad = sigma * numpy.pi / 180 / 3600
    gaussian = gaussian1d(sigma_rad, Rbar)
    norm = scipy.integrate.simps(gaussian(R) * zen_PSF, R)
    return scipy.integrate.simps(gaussian(R) * zen_PSF * (R - Rbar)**2, R)/norm


if __name__ == '__main__':
    fdata = numpy.genfromtxt('../data/filters/LSST_r.dat')
    wave, fthroughput = fdata[:,0], fdata[:,1]
    sdata = numpy.genfromtxt('../data/SEDs/ukg5v.ascii')
    swave, flux = sdata[:,0], sdata[:,1]
    flux_i = numpy.interp(wave, swave, flux)
    photons = flux_i * fthroughput * wave
    M = disp_moments(wave, photons, zenith=45.0 * numpy.pi/180.0)
    print 'First and second moments of ukg5v star through LSST_r filter at 45 degrees zenith'
    print 'Computing these two ways.  Better match!'
    print M[0] * 206265, M[1] * 206265**2
    M = disp_moments_R(wave, photons, zenith=45.0 * numpy.pi/180.0)
    print M[0] * 206265, M[1] * 206265**2
