import numpy
from astropy.utils.console import ProgressBar

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
    fdata = numpy.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]

    photons = numpy.zeros_like(fwave)
    for SED_file, weight in zip(SED_files, weights):
        sdata = numpy.genfromtxt(SED_file)
        swave, flux = sdata[:,0] * (1.0 + redshift), sdata[:,1]
        flux_i = numpy.interp(fwave, swave, flux)
        photons1 = flux_i * throughput * fwave
        w = numpy.where(photons1 < 1.e-5 * photons1.max())[0]
        photons1[w] = 0.0
        photons1 *= weight / simps(photons1, fwave)
        photons += photons1
    w = numpy.where(photons > 0.0)[0]
    return fwave[w.min():w.max()], photons[w.min():w.max()]

def shear_galaxy(c_ellip, c_gamma):
    '''Compute complex ellipticity after shearing by complex shear `c_gamma`.'''
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)

def ringtest(gamma, n_ring, gen_target_image, gen_init_param, measure_ellip):
    ''' Performs a shear calibration ringtest.

    Produces "true" images uniformly spread along a ring in ellipticity space using the supplied
    `gen_target_image` function.  Then tries to fit these images, (returning ellipticity estimates)
    using the supplied `measure_ellip` function with the fit initialized by the supplied
    `gen_init_param` function.

    The "true" images are sheared by `gamma` (handled by passing through to `gen_target_image`).
    Images are generated in pairs separated by 180 degrees on the ellipticity plane to minimize shape
    noise.

    Ultimately returns an estimate of the applied shear (`gamma_hat`), which can then be compared to
    the input shear `gamma` in an external function to estimate shear calibration parameters.
    '''

    betas = numpy.linspace(0.0, 2.0 * numpy.pi, n_ring, endpoint=False)
    ellip0s = []
    ellip180s = []
    with ProgressBar(2 * n_ring) as bar:
        for beta in betas:
            #measure ellipticity at beta along the ring
            target_image0 = gen_target_image(gamma, beta)
            init_param0 = gen_init_param(gamma, beta)
            ellip0 = measure_ellip(target_image0, init_param0)
            ellip0s.append(ellip0)
            bar.update()

            #repeat with beta on opposite side of the ring (i.e. +180 deg)
            target_image180 = gen_target_image(gamma, beta + numpy.pi)
            init_param180 = gen_init_param(gamma, beta + numpy.pi)
            ellip180 = measure_ellip(target_image180, init_param180)
            ellip180s.append(ellip180)
            bar.update()
    gamma_hats = [0.5 * (e0 + e1) for e0, e1 in zip(ellip0s, ellip180s)]
    gamma_hat = numpy.mean(gamma_hats)
    return gamma_hat
