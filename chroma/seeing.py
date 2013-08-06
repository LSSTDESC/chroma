import scipy.integrate

def relative_second_moment_radius(wave, photons):
    ''' Returns the second moment radius of PSF with specified SED, normalized at 500nm.
    '''
    return (scipy.integrate.simps(photons * (wave/500) ** -0.4, wave) /
            scipy.integrate.simps(photons, wave))
