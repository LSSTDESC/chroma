import numpy

def relative_second_moment_radius(wave, photons, alpha=-0.2):
    ''' Returns the second moment radius of PSF with specified SED, normalized at 500nm.
    '''
    return (numpy.trapz(photons * (wave/500) ** (2*alpha), wave) /
            numpy.trapz(photons, wave))
