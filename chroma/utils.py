import numpy

def get_SED_photons(SED_file, filter_file, redshift, oob_thresh=1.e-5):
    '''Return wave and photon-flux of filtered spectrum.

    Arguments
    ---------
    SED_file -- filename containing two column data:
                column 1 -- wavelength in nm
                column 2 -- flux proportional to erg/s/cm^2/Ang
    filter_file -- filename containing two column data:
                   column 1 -- wavelength in nm
                   column 2 -- fraction of above-atmosphere photons eventually accepted.
    redshift -- redshift the SED (assumed to be initially rest-frame) this amount
    oob_thresh -- out-of-band threshold at which to clip throughput
    '''
    fdata = numpy.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]

    sdata = numpy.genfromtxt(SED_file)
    swave, flux = sdata[:,0] * (1.0 + redshift), sdata[:,1]
    flux_i = numpy.interp(fwave, swave, flux)
    photons = flux_i * throughput * fwave
    w = numpy.where(photons > oob_thresh * photons.max())[0]
    return fwave[w.min():w.max()], photons[w.min():w.max()]

def get_composite_SED_photons(SED_files, weights, filter_file, redshift):
    '''Return wave and photon-flux of filtered composite spectrum.

    Composite here means different SEDs are added together with corresponding `weights`.
    The weights are applied after normalizing by number of surviving photons for each
    SED. Inputs are similar to get_SED_photons.

    Arguments
    ---------
    SED_files -- iterable of strings, each a filename containing a two-column SED
                 column 1 -- wavelength in nm
                 column 2 -- flux proportional to ers/s/cm^2/A
    weights -- iterable containing fraction of spatially integrated flux for each SED component.
               note that weights are applied after redshifting
    filter_file -- filename for file containing two-columns:
                 column 1 -- wavelength in nm
                 column 2 -- fraction of above-atmosphere photons eventually accepted.
    redshift -- redshift each of the (rest-frame!) SEDs this amount.
    '''
    fdata = np.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]

    photons = np.zeros_like(fwave)
    for SED_file, weight in zip(SED_files, weights):
        sdata = np.genfromtxt(SED_file)
        swave, flux = sdata[:,0] * (1.0 + redshift), sdata[:,1]
        flux_i = np.interp(fwave, swave, flux)
        photons1 = flux_i * throughput * fwave
        w = np.where(photons1 < 1.e-5 * photons1.max())[0]
        photons1[w] = 0.0
        photons1 *= weight / simps(photons1, fwave)
        photons += photons1
    w = np.where(photons > 0.0)[0]
    return fwave[w.min():w.max()], photons[w.min():w.max()]

def shear_galaxy(c_ellip, c_gamma):
    '''Compute complex ellipticity after shearing by complex shear `c_gamma`.'''
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)
