import numpy as np
from astropy.utils.console import ProgressBar

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

    betas = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
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
            target_image180 = gen_target_image(gamma, beta + np.pi)
            init_param180 = gen_init_param(gamma, beta + np.pi)
            ellip180 = measure_ellip(target_image180, init_param180)
            ellip180s.append(ellip180)
            bar.update()
    gamma_hats = [0.5 * (e0 + e1) for e0, e1 in zip(ellip0s, ellip180s)]
    gamma_hat = np.mean(gamma_hats)
    return gamma_hat
