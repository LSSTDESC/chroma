import numpy
import scipy

import astropy.utils.console

def get_photons(SED_file, filter_file, redshift):
    fwave, throughput = numpy.genfromtxt(filter_file).T
    swave, flux = numpy.genfromtxt(SED_file).T
    swave *= (1.0 + redshift)
    flux_i = numpy.interp(fwave, swave, flux)
    photons = flux_i * throughput * fwave
    return fwave, photons

def shear_galaxy(c_ellip, c_gamma):
    '''Compute complex ellipticity after shearing by complex shear `c_gamma`.'''
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)

def ringtest(gamma, n_ring, gen_target_image, gen_init_param, measure_ellip, silent=False):
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
    if not silent:
        with astropy.utils.console.ProgressBar(2 * n_ring) as bar:
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
    else:
        for beta in betas:
            #measure ellipticity at beta along the ring
            target_image0 = gen_target_image(gamma, beta)
            init_param0 = gen_init_param(gamma, beta)
            ellip0 = measure_ellip(target_image0, init_param0)
            ellip0s.append(ellip0)

            #repeat with beta on opposite side of the ring (i.e. +180 deg)
            target_image180 = gen_target_image(gamma, beta + numpy.pi)
            init_param180 = gen_init_param(gamma, beta + numpy.pi)
            ellip180 = measure_ellip(target_image180, init_param180)
            ellip180s.append(ellip180)
            gamma_hats = [0.5 * (e0 + e1) for e0, e1 in zip(ellip0s, ellip180s)]

    gamma_hat = numpy.mean(gamma_hats)
    return gamma_hat

def FWHM(data, pixsize=1.0):
    '''Compute the full-width at half maximum of a symmetric 2D distribution.  Assumes that measuring
    along the x-axis is sufficient (ignores all but one row, the one containing the distribution
    maximum).  Scales result by `pixsize` for non-unit width pixels.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    '''
    height = data.max()
    w = numpy.where(data == height)
    y0, x0 = w[0][0], w[1][0]
    xs = numpy.arange(data.shape[0], dtype=numpy.float64) * pixsize
    low = numpy.interp(0.5*height, data[x0, 0:x0], xs[0:x0])
    high = numpy.interp(0.5*height, data[x0+1, -1:x0:-1], xs[-1:x0:-1])
    return abs(high-low)

def moments(data, pixsize=1.0):
    '''Compute first and second (quadrupole) moments of `data`.  Scales result by `pixsize` for
    non-unit width pixels.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    '''
    xs, ys = numpy.meshgrid(numpy.arange(data.shape[0], dtype=numpy.float64) * pixsize,
                            numpy.arange(data.shape[0], dtype=numpy.float64) * pixsize)
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs-xbar)**2).sum() / total
    Iyy = (data * (ys-ybar)**2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return xbar, ybar, Ixx, Iyy, Ixy

def AHM(data, pixsize=1.0, height=None):
    ''' Compute area above half maximum as a potential replacement for FWHM.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    height -- optional maximum height of data (defaults to sample maximum).
    '''
    if height is None:
        height = data.max()
    return (data > (0.5 * height)).sum() * scale**2
