''' Functions for handling Bulge+Disk galaxies such as used in Voigt+12.
'''

import numpy as np
import copy
from VoigtImageFactory import VoigtImageFactory
from Voigt12PSF import Voigt12PSF
from scipy.integrate import simps
from sersic import Sersic
from scipy.optimize import newton
from lmfit import minimize, report_errors

def get_SED_photons(SED_file, filter_file, redshift):
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
    '''
    fdata = np.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]

    sdata = np.genfromtxt(SED_file)
    swave, flux = sdata[:,0] * (1.0 + redshift), sdata[:,1]
    flux_i = np.interp(fwave, swave, flux)
    photons = flux_i * throughput * fwave
    w = np.where(photons > 1.e-5 * photons.max())[0]
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

def build_PSFs(filter_file, bulge_flux,
               bulge_SED_file, disk_SED_file,
               redshift,
               PSF_ellip=0.0, PSF_phi=0.0):
    ''' Build bulge, disk, and composite PSFs from SEDs and filter function.

    Assume that disk_flux = 1.0 - bulge_flux.
    Forwards arguments to get(_composite)_SED_photons, and Voigt12PSF init.
    '''
    bwave, bphotons = get_SED_photons(bulge_SED_file, filter_file, redshift)
    b_PSF = Voigt12PSF(bwave, bphotons, ellipticity=PSF_ellip, phi=PSF_phi)
    dwave, dphotons = get_SED_photons(disk_SED_file, filter_file, redshift)
    d_PSF = Voigt12PSF(dwave, dphotons, ellipticity=PSF_ellip, phi=PSF_phi)
    cwave, cphotons = get_composite_SED_photons([bulge_SED_file, disk_SED_file],
                                                [bulge_flux, 1.0 - bulge_flux],
                                                filter_file, redshift)
    c_PSF = Voigt12PSF(cwave, cphotons, ellipticity=PSF_ellip, phi=PSF_phi)
    circ_c_PSF = Voigt12PSF(cwave, cphotons)
    return b_PSF, d_PSF, c_PSF, circ_c_PSF

def shear_galaxy(c_ellip, c_gamma):
    '''Compute complex ellipticity after shearing by complex shear `c_gamma`.'''
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)

def gal_image(gparam, b_PSF, d_PSF, im_fac):
    '''Use image_factory `im_fac` to make a galaxy image from params in gparam and using
    the bulge and disk psfs `b_PSF` and `d_PSF`.

    Arguments
    ---------
    gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
              `b_` prefix for bulge, `d_` prefix for disk.
              Suffixes are all init arguments for the Sersic object.

    Note that you can specify the composite PSF `c_PSF` for both bulge and disk PSF when using during
    ringtest fits.
    '''
    bulge = Sersic(gparam['b_y0'].value,
                   gparam['b_x0'].value,
                   gparam['b_n'].value,
                   r_e=gparam['b_r_e'].value,
                   gmag=gparam['b_gmag'].value,
                   phi=gparam['b_phi'].value,
                   flux=gparam['b_flux'].value)
    disk = Sersic(gparam['d_y0'].value,
                  gparam['d_x0'].value,
                  gparam['d_n'].value,
                  r_e=gparam['d_r_e'].value,
                  gmag=gparam['d_gmag'].value,
                  phi=gparam['d_phi'].value,
                  flux=gparam['d_flux'].value)
    return im_fac.get_image([(bulge, b_PSF), (disk, d_PSF)])

def gal_overimage(gparam, b_PSF, d_PSF, im_fac):
    '''Compute oversampled galaxy image.  Similar to `gal_image()`.

    Useful for computing FWHM of galaxy image at higher resolution than available from just
    `gal_image()`.
    '''
    bulge = Sersic(gparam['b_y0'].value,
                   gparam['b_x0'].value,
                   gparam['b_n'].value,
                   r_e=gparam['b_r_e'].value,
                   gmag=gparam['b_gmag'].value,
                   phi=gparam['b_phi'].value,
                   flux=gparam['b_flux'].value)
    disk = Sersic(gparam['d_y0'].value,
                  gparam['d_x0'].value,
                  gparam['d_n'].value,
                  r_e=gparam['d_r_e'].value,
                  gmag=gparam['d_gmag'].value,
                  phi=gparam['d_phi'].value,
                  flux=gparam['d_flux'].value)
    return im_fac.get_overimage([(bulge, b_PSF), (disk, d_PSF)])

def target_image_fn_generator(gparam, b_PSF, d_PSF, im_fac):
    '''Return a function that can be passed to `ringtest()` which produces a target image as a
    function of applied shear `gamma` and angle along the ring `beta`.

    Arguments are passed through to `gal_image()`
    '''

    gen_init_param = init_param_generator(gparam)
    def f(gamma, beta):
        gparam1 = gen_init_param(gamma, beta)
        return gal_image(gparam1, b_PSF, d_PSF, im_fac)
    return f

def init_param_generator(gparam):
    '''Return a function that can be passed to `ringtest()` which generates initial conditions for
    fit to the "true image" using the incorrect PSF as a function of applied shear `gamma` and angle
    along the ring `beta`.
    '''
    # parameters which will change with applied shear gamma or angle along ring beta
    # extract their initial values here
    b_y0 = gparam['b_y0'].value
    b_x0 = gparam['b_x0'].value
    b_gmag = gparam['b_gmag'].value
    b_phi = gparam['b_phi'].value
    b_r_e = gparam['b_r_e'].value
    d_y0 = gparam['d_y0'].value
    d_x0 = gparam['d_x0'].value
    d_gmag = gparam['d_gmag'].value
    d_phi = gparam['d_phi'].value
    d_r_e = gparam['d_r_e'].value

    def gen_init_param(gamma, beta):
        gparam1 = copy.deepcopy(gparam)
        b_phi_ring = b_phi + beta/2.0
        d_phi_ring = d_phi + beta/2.0
        # bulge complex ellipticity
        b_c_ellip = b_gmag * complex(np.cos(2.0 * b_phi_ring), np.sin(2.0 * b_phi_ring))
        # bulge sheared complex ellipticity
        b_s_c_ellip = shear_galaxy(b_c_ellip, gamma)
        b_s_gmag = abs(b_s_c_ellip)
        b_s_phi = np.angle(b_s_c_ellip) / 2.0
        # disk complex ellipticity
        d_c_ellip = d_gmag * complex(np.cos(2.0 * d_phi_ring), np.sin(2.0 * d_phi_ring))
        # disk sheared complex ellipticity
        d_s_c_ellip = shear_galaxy(d_c_ellip, gamma)
        d_s_gmag = abs(d_s_c_ellip)
        d_s_phi = np.angle(d_s_c_ellip) / 2.0
        # radius rescaling
        rescale = np.sqrt(1.0 - abs(gamma)**2.0)

        gparam1['b_y0'].value = b_y0 * np.sin(beta / 2.0) + b_x0 * np.cos(beta / 2.0)
        gparam1['b_x0'].value = b_y0 * np.cos(beta / 2.0) - b_x0 * np.sin(beta / 2.0)
        gparam1['d_y0'].value = d_y0 * np.sin(beta / 2.0) + d_x0 * np.cos(beta / 2.0)
        gparam1['d_x0'].value = d_y0 * np.cos(beta / 2.0) - d_x0 * np.sin(beta / 2.0)
        gparam1['b_gmag'].value = b_s_gmag
        gparam1['d_gmag'].value = d_s_gmag
        gparam1['b_phi'].value = b_s_phi
        gparam1['d_phi'].value = d_s_phi
        gparam1['b_r_e'].value = b_r_e * rescale
        gparam1['d_r_e'].value = d_r_e * rescale
        return gparam1
    return gen_init_param

def fit_image_fn_generator(c_PSF, im_fac):
    def f(gparam):
        return gal_image(gparam, c_PSF, c_PSF, im_fac)
    return f

def ellip_measurement_generator(c_PSF, im_fac):
    '''Return a function that can be passed to `ringtest()` which computes the best-fit ellipticity
    of a sheared galaxy along the ring as a function of the "true" image `target_image` and some
    initial parameters for the fit.
    '''
    def measure_ellip(target_image, init_param):
        image_gen = fit_image_fn_generator(c_PSF, im_fac)
        resid = lambda param: (image_gen(param) - target_image).flatten()
        result = minimize(resid, init_param)
        gmag = result.params['d_gmag'].value
        phi = result.params['d_phi'].value
        c_ellip = gmag * complex(np.cos(2.0  * phi), np.sin(2.0 * phi))
        return c_ellip
    return measure_ellip

def FWHM(data, scale=1.0):
    '''Compute the full-width at half maximum of a symmetric 2D distribution.  Assumes that measuring
    along the x-axis is sufficient (ignores all but one row, the one containing the distribution
    maximum).  Scales result by scale for non-unit width pixels.
    '''
    height = data.max()
    w = np.where(data == height)
    y0, x0 = w[0][0], w[1][0]
    xs = np.arange(data.shape[0], dtype=np.float64)/scale
    low = np.interp(0.5*height, data[x0, 0:x0], xs[0:x0])
    high = np.interp(0.5*height, data[x0+1, -1:x0:-1], xs[-1:x0:-1])
    return abs(high-low)

def set_FWHM_ratio(gparam, rpg, circ_c_PSF, im_fac):
    '''Set the effective radii of the bulge+disk galaxy specified in `gparam` such that the ratio
    of the FWHM of the PSF-convolved galaxy image is `rpg` times the FWHM of the PSF itself. The
    galaxy is circularized and centered at the origin for this computation (ellip -> 0.0) and
    (x0, y0 -> 0.0, 0.0).  The supplied PSF `circ_c_PSF` is assumed to be circularly symmetric.
    '''
    FWHM_PSF = FWHM(im_fac.get_PSF_image(circ_c_PSF), scale=im_fac.oversample_factor)
    gparam2 = copy.deepcopy(gparam)
    gparam2['b_gmag'].value = 0.0
    gparam2['b_x0'].value = 0.0
    gparam2['b_y0'].value = 0.0
    gparam2['d_gmag'].value = 0.0
    gparam2['d_x0'].value = 0.0
    gparam2['d_y0'].value = 0.0
    def FWHM_gal(scale):
        gparam2['b_r_e'].value = gparam['b_r_e'].value * scale
        gparam2['d_r_e'].value = gparam['d_r_e'].value * scale
        image = gal_overimage(gparam2, circ_c_PSF, circ_c_PSF, im_fac)
        return FWHM(image, scale=im_fac.oversample_factor)
    def f(scale):
        return FWHM_gal(scale) - rpg * FWHM_PSF
    scale = newton(f, 1.0)
    # report_errors(gparam)
    # import ipdb; ipdb.set_trace()
    gparam['b_r_e'].value *= scale
    gparam['d_r_e'].value *= scale

if __name__ == '__main__':
    # Returns some objects that can be passed to ringtest for interactively checking that the code
    # is working as expected.
    from lmfit import Parameter, Parameters, Minimizer, minimize

    gparam = Parameters()
    # bulge
    gparam.add('b_x0', value=2.1)
    gparam.add('b_y0', value=3.3)
    gparam.add('b_n', value=4.0, vary=False)
    gparam.add('b_r_e', value=2.7)
    gparam.add('b_flux', value=0.25)
    gparam.add('b_gmag', value=0.4)
    gparam.add('b_phi', value=0.0)
    # disk
    gparam.add('d_x0', expr='b_x0')
    gparam.add('d_y0', expr='b_y0')
    gparam.add('d_n', value=1.0, vary=False)
    gparam.add('d_r_e', value=2.7 * 1.1)
    gparam.add('d_flux', expr='1.0 - b_flux')
    gparam.add('d_gmag', expr='b_gmag')
    gparam.add('d_phi', expr='b_phi')

    dummyfit = Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()

    filter_file = '../data/filters/voigt12_350.dat'
    bulge_SED_file = '../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../data/SEDs/CWW_Sbc_ext.ascii'

    b_PSF, d_PSF, c_PSF, circ_c_PSF = build_PSFs(filter_file, 0.25, bulge_SED_file,
                                                 disk_SED_file, 0.9, PSF_ellip=0.05)
    # im_fac = VoigtImageFactory(size=51, oversample_factor=3)
    im_fac = VoigtImageFactory()
    set_FWHM_ratio(gparam, 1.4, circ_c_PSF, im_fac)
    gen_target_image = target_image_fn_generator(gparam, b_PSF, d_PSF, im_fac)
    gen_init_param = init_param_generator(gparam)
    measure_ellip = ellip_measurement_generator(c_PSF, im_fac)
