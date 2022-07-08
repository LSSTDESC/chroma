import sys

import numpy as np

try:
    from astropy.utils.console import ProgressBar
except:

    def isiterable(obj):
        """Returns `True` if the given object is iterable."""

        try:
            iter(obj)
            return True
        except TypeError:
            return False

    class ProgressBar(object):
        """A somewhat simple console progress bar in case user doesn't have astropy."""

        def __init__(self, total_or_items, file=sys.stdout):
            if isiterable(total_or_items):
                self._items = iter(total_or_items)
                self._total = len(total_or_items)
            else:
                self._total = total_or_items
            self.file_ = file
            self._i = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return True

        def __iter__(self):
            return self

        def next(self):
            try:
                ret = next(self._items)
            except StopIteration:
                self.__exit__(None, None, None)
                raise
            else:
                self.update()
                return ret

        def update(self):
            self.file_.write(".")
            self.file_.flush()
            if self._i % 100 == 0:
                print("{} of {}".format(self._i, self._total))
            self._i += 1


def Sersic_r2_over_hlr(n):
    """Factor to convert the half light radius `hlr` to the 2nd moment radius `r^2` defined as
    sqrt(Ixx + Iyy) where Ixx and Iyy are the second central moments of a distribution in
    perpendicular directions.  Depends on the Sersic index n.  The polynomial below is derived from
    a Mathematica fit to the exact relation, and should be good to ~(0.01 - 0.04)% over the range
    0.2 < n < 8.0.

    @param n Sersic index
    @returns ratio sqrt(r^2) / hlr
    """
    return 0.985444 + n * (
        0.391016
        + n
        * (
            0.0739602
            + n
            * (0.00698719 + n * (0.00212432 + n * (-0.000154052 + n * 0.0000219632)))
        )
    )


def component_Sersic_r2(ns, weights, hlrs):
    """Calculate second moment radius of concentric multi-Sersic galaxy.

    @param  ns       List of Sersic indices in model.
    @param  weights  Relative flux of each component
    @param  hlrs     List of half-light-radii of each component
    @returns         Second moment radius = sqrt(r^2)
    """
    t = np.array(weights).sum()
    ws = [w / t for w in weights]
    r2s = [Sersic_r2_over_hlr(n) * hlr for n, hlr in zip(ns, hlrs)]
    return np.sqrt(reduce(lambda x, y: x + y, [r2**2 * w for r2, w in zip(r2s, ws)]))


def apply_shear(c_ellip, c_gamma):
    """Compute complex ellipticity after shearing by complex shear `c_gamma`."""
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)


def ringtest(
    gamma, n_ring, gen_target_image, gen_init_param, measure_ellip, silent=False
):
    """Performs a shear calibration ringtest.

    Produces "true" images uniformly spread along a ring in ellipticity space using the supplied
    `gen_target_image` function.  Then tries to fit these images, (returning ellipticity estimates)
    using the supplied `measure_ellip` function with the fit initialized by the supplied
    `gen_init_param` function.

    The "true" images are sheared by `gamma` (handled by passing through to `gen_target_image`).
    Images are generated in pairs separated by 180 degrees on the ellipticity plane to minimize shape
    noise.

    Ultimately returns an estimate of the applied shear (`gamma_hat`), which can then be compared to
    the input shear `gamma` in an external function to estimate shear calibration parameters.
    """

    betas = np.linspace(0.0, np.pi, n_ring, endpoint=False)
    ellip0s = []
    ellip180s = []

    def work():
        # measure ellipticity at beta along the ring
        target_image0 = gen_target_image(gamma, beta)
        init_param0 = gen_init_param(gamma, beta)
        ellip0 = measure_ellip(target_image0, init_param0)
        ellip0s.append(ellip0)

        # repeat with beta on opposite side of the ring (i.e. +180 deg)
        target_image180 = gen_target_image(gamma, beta + np.pi)
        init_param180 = gen_init_param(gamma, beta + np.pi)
        ellip180 = measure_ellip(target_image180, init_param180)
        ellip180s.append(ellip180)

    # Use fancy console updating if astropy is installed and not silenced
    if not silent:
        with ProgressBar(n_ring) as bar:
            for beta in betas:
                work()
                bar.update()
    else:
        for beta in betas:
            work()

    gamma_hats = [0.5 * (e0 + e1) for e0, e1 in zip(ellip0s, ellip180s)]
    gamma_hat = np.mean(gamma_hats)
    return gamma_hat


def measure_shear_calib(gparam, gen_target_image, get_ring_params, measurer, nring=3):
    """Measure the shear calibration parameters m1, m2, c1, and c2 by performing ring tests for
    two values of true shear.

    @param    gparam            Parameters describing the galaxy to be simulated.
    @param    gen_target_image  Python function that returns a GalSim.Image given a `gparam`
                                argument.
    @param    get_ring_params   Python function that returns gparam initial guesses as a function
                                of shear and angle around the ellipticity ring.
    @param    measurer          A chroma.EllipMeasurer instance to measure the ellipticity of each
                                target image around the ring.
    @param    nring             Number of angles around the ellipticity ring for which to generate
                                images.
    @returns  ((m1, m2), (c1, c2))
    """
    # Do ring test for two values of the complex reduced shear `gamma`, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = ringtest(
        gamma0, nring, gen_target_image, get_ring_params, measurer, silent=True
    )
    # c is the same as the estimated reduced shear `gamma_hat` when the input reduced shear
    # is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = ringtest(
        gamma1, nring, gen_target_image, get_ring_params, measurer, silent=True
    )
    # solve for m
    m0 = (gamma1_hat.real - c[0]) / gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1]) / gamma1.imag - 1.0
    m = m0, m1

    return m, c


def moments(image):
    """Compute first and second (quadrupole) moments of `image`.  Scales result for non-unit width
    pixels.

    @param image   galsim.Image to analyze
    @returns       x0, y0, Ixx, Iyy, Ixy - first and second moments of image.
    """
    data = image.array
    scale = image.scale
    xs, ys = np.meshgrid(
        np.arange(data.shape[0], dtype=np.float64) * scale,
        np.arange(data.shape[0], dtype=np.float64) * scale,
    )
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs - xbar) ** 2).sum() / total
    Iyy = (data * (ys - ybar) ** 2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return xbar, ybar, Ixx, Iyy, Ixy


def my_imshow(im, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%8e @ [%4i, %4i]" % (im[y, x], x, y)
        except IndexError:
            return ""

    img = ax.imshow(im, **kwargs)
    ax.format_coord = format_coord
    return img


# def FWHM(data, pixsize=1.0):
#     """Compute the full-width at half maximum of a symmetric 2D distribution.  Assumes that measuring
#     along the x-axis is sufficient (ignores all but one row, the one containing the distribution
#     maximum).  Scales result by `pixsize` for non-unit width pixels.

#     Arguments
#     ---------
#     data -- array to analyze
#     pixsize -- linear size of a pixel
#     """
#     height = data.max()
#     w = np.where(data == height)
#     y0, x0 = w[0][0], w[1][0]
#     xs = np.arange(data.shape[0], dtype=np.float64) * pixsize
#     low = np.interp(0.5*height, data[x0, 0:x0], xs[0:x0])
#     high = np.interp(0.5*height, data[x0+1, -1:x0:-1], xs[-1:x0:-1])
#     return abs(high-low)

# def AHM(data, pixsize=1.0, height=None):
#     """ Compute area above half maximum as a potential replacement for FWHM.

#     Arguments
#     ---------
#     data -- array to analyze
#     pixsize -- linear size of a pixel
#     height -- optional maximum height of data (defaults to sample maximum).
#     """
#     if height is None:
#         height = data.max()
#     return (data > (0.5 * height)).sum() * scale**2
