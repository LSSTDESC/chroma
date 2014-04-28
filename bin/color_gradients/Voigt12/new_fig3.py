import os
import sys
import logging
import copy

import scipy.optimize
import numpy as np
import lmfit
import galsim

pixel_scale = 0.2
data_dir = '../../../data/'

def fiducial_galaxy():
    '''Bulge + disk parameters of the fiducial galaxy described in Voigt+12.'''
    gparam = lmfit.Parameters()
    # bulge
    gparam.add('b_x0', value=0.1)
    gparam.add('b_y0', value=0.3)
    gparam.add('b_n', value=4.0, vary=False)
    gparam.add('b_hlr', value=1.1 * 1.1)
    gparam.add('b_flux', value=0.25)
    gparam.add('b_g', value=0.2, min=0.0, max=1.0)
    gparam.add('b_phi', value=0.0)
    # disk
    gparam.add('d_x0', expr='b_x0')
    gparam.add('d_y0', expr='b_y0')
    gparam.add('d_n', value=1.0, vary=False)
    gparam.add('d_hlr', value=1.1)
    gparam.add('d_flux', expr='1.0 - b_flux')
    gparam.add('d_g', expr='b_g')
    gparam.add('d_phi', expr='b_phi')
    # initialize constrained variables
    dummyfit = lmfit.Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()
    return gparam

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

    betas = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    ellip0s = []
    ellip180s = []

    def work():
        #measure ellipticity at beta along the ring
        target_image0 = gen_target_image(gamma, beta)
        init_param0 = gen_init_param(gamma, beta)
        ellip0 = measure_ellip(target_image0, init_param0)
        ellip0s.append(ellip0)

        #repeat with beta on opposite side of the ring (i.e. +180 deg)
        target_image180 = gen_target_image(gamma, beta + np.pi)
        init_param180 = gen_init_param(gamma, beta + np.pi)
        ellip180 = measure_ellip(target_image180, init_param180)
        ellip180s.append(ellip180)

    if not silent:
        with chroma.ProgressBar(n_ring) as bar:
            for beta in betas:
                work()
                bar.update()
    else:
        for beta in betas:
            work()

    gamma_hats = [0.5 * (e0 + e1) for e0, e1 in zip(ellip0s, ellip180s)]
    gamma_hat = np.mean(gamma_hats)
    return gamma_hat

def shear_galaxy(c_ellip, c_gamma):
    '''Compute complex ellipticity after shearing by complex shear `c_gamma`.'''
    return (c_ellip + c_gamma) / (1.0 + c_gamma.conjugate() * c_ellip)

def ring_params(gparam, gamma, beta):
    ''' Adjust bulge+disk parameters in `gparam0` to reflect applied shear `gamma` and
    angle around the ring `beta` in a ring test.  Returned parameters are good both for
    creating the target image and for initializing the lmfit minimize routine.
    '''
    gparam1 = copy.deepcopy(gparam)
    b_phi_ring = gparam['b_phi'].value + beta/2.0
    d_phi_ring = gparam['d_phi'].value + beta/2.0
    # bulge complex ellipticity
    b_c_ellip = gparam['b_g'].value * complex(np.cos(2.0 * b_phi_ring), np.sin(2.0 * b_phi_ring))
    # bulge sheared complex ellipticity
    b_s_c_ellip = shear_galaxy(b_c_ellip, gamma)
    b_s_g = abs(b_s_c_ellip)
    b_s_phi = np.angle(b_s_c_ellip) / 2.0
    # disk complex ellipticity
    d_c_ellip = gparam['d_g'].value * complex(np.cos(2.0 * d_phi_ring), np.sin(2.0 * d_phi_ring))
    # disk sheared complex ellipticity
    d_s_c_ellip = shear_galaxy(d_c_ellip, gamma)
    d_s_g = abs(d_s_c_ellip)
    d_s_phi = np.angle(d_s_c_ellip) / 2.0
    # radius rescaling
    rescale = np.sqrt(1.0 - abs(gamma)**2.0)

    gparam1['b_x0'].value \
      = gparam['b_x0'].value * np.cos(beta / 2.0) \
      - gparam['b_y0'].value * np.sin(beta / 2.0)
    gparam1['b_y0'].value \
      = gparam['b_x0'].value * np.sin(beta / 2.0) \
      + gparam['b_y0'].value * np.cos(beta / 2.0)
    gparam1['d_x0'].value \
      = gparam['d_x0'].value * np.cos(beta / 2.0) \
      - gparam['d_y0'].value * np.sin(beta / 2.0)
    gparam1['d_y0'].value \
      = gparam['d_x0'].value * np.sin(beta / 2.0) \
      + gparam['d_y0'].value * np.cos(beta / 2.0)
    gparam1['b_g'].value = b_s_g
    gparam1['d_g'].value = d_s_g
    gparam1['b_phi'].value = b_s_phi
    gparam1['d_phi'].value = d_s_phi
    gparam1['b_hlr'].value = gparam['b_hlr'].value * rescale
    gparam1['d_hlr'].value = gparam['d_hlr'].value * rescale
    return gparam1

def measure_shear_calib(gparam, bandpass, b_SED, d_SED, c_SED, PSF):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''

    pix = galsim.Pixel(pixel_scale)
    # generate target image using ringed gparam and PSFs
    def gen_target_image(gamma, beta):
        ring_param = ring_params(gparam, gamma, beta)
        profile = param_to_gal(ring_param, b_SED, d_SED)
        final = galsim.Convolve(profile, PSF, pix)
        image = galsim.ImageD(15, 15, scale=pixel_scale)
        final.draw(bandpass, image=image)
        return image

    # function to measure ellipticity of target_image by trying to match the pixels
    # but using the "wrong" PSF (the composite PSF for both bulge and disk).
    def measure_ellip(target_image, init_param):
        def resid(param):
            profile = param_to_gal(param, c_SED, c_SED)
            final = galsim.Convolve(profile, PSF, pix)
            image = galsim.ImageD(15, 15, scale=pixel_scale)
            final.draw(bandpass, image=image)
            return (image.array - target_image.array).flatten()
        result = lmfit.minimize(resid, init_param)
        g = result.params['d_g'].value
        phi = result.params['d_phi'].value
        c_ellip = g * complex(np.cos(2.0 * phi), np.sin(2.0 * phi))
        return c_ellip

    def get_ring_params(gamma, beta):
        return ring_params(gparam, gamma, beta)

    # Ring test for two values of gamma, solve for m and c.
    gamma0 = 0.0 + 0.0j
    gamma0_hat = ringtest(gamma0, 3, gen_target_image, get_ring_params, measure_ellip)
    # c is just gamma_hat when input gamma_true is (0.0, 0.0)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = ringtest(gamma1, 3, gen_target_image, get_ring_params, measure_ellip)
    # solve for m
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1

    return m, c

# def FWHM(image):
#     '''Compute the full-width at half maximum of a symmetric 2D galsim image.  Assumes that measuring
#     along the x-axis is sufficient (ignores all but one row, the one containing the distribution
#     maximum).

#     Arguments
#     ---------
#     image -- galsim.Image instance
#     '''
#     height = image.array.max()
#     w = np.where(image.array == height)
#     y0, x0 = w[0][0], w[1][0]
#     xs = np.arange(image.array.shape[0], dtype=np.float64) * image.scale
#     low = np.interp(0.5*height, image.array[x0, 0:x0], xs[0:x0])
#     high = np.interp(0.5*height, image.array[x0+1, -1:x0:-1], xs[-1:x0:-1])
#     return abs(high-low)

def gaussian(height, center_x, center_y, width_x, width_y):
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x - x)/width_x)**2 + ((center_y - y)/width_y)**2)/2)

def moments(image):
    data = image.array
    total = data.sum()
    x_idx, y_idx = np.indices(data.shape)
    x = (x_idx*data).sum()/total
    y = (y_idx*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(image):
    data = image.array
    params = moments(image)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = scipy.optimize.leastsq(errorfunction, params)
    return p

def FWHM(image):
    p = fitgaussian(image)
    return np.sqrt(p[3]*p[4])*image.scale

def param_to_gal(gparam, b_SED, d_SED):
    mono_bulge = galsim.Sersic(n=gparam['b_n'].value,
                               half_light_radius=gparam['b_hlr'].value * pixel_scale,
                               flux=gparam['b_flux'].value)
    mono_bulge.applyShift(gparam['b_x0'].value * pixel_scale,
                          gparam['b_y0'].value * pixel_scale)
    mono_bulge.applyShear(g=gparam['b_g'].value, beta=gparam['b_phi'].value * galsim.radians)
    bulge = galsim.Chromatic(mono_bulge, b_SED)
    mono_disk = galsim.Sersic(n=gparam['d_n'].value,
                              half_light_radius=gparam['d_hlr'].value * pixel_scale,
                              flux=gparam['d_flux'].value)
    mono_disk.applyShift(gparam['d_x0'].value * pixel_scale,
                         gparam['d_y0'].value * pixel_scale)
    mono_disk.applyShear(g=gparam['d_g'].value, beta=gparam['d_phi'].value * galsim.radians)
    disk = galsim.Chromatic(mono_disk, d_SED)
    gal = bulge+disk
    return gal

def param_to_circ_gal(gparam, b_SED, d_SED):
    mono_bulge = galsim.Sersic(n=gparam['b_n'].value,
                               half_light_radius=gparam['b_hlr'].value * pixel_scale,
                               flux=gparam['b_flux'].value)
    bulge = galsim.Chromatic(mono_bulge, b_SED)
    mono_disk = galsim.Sersic(n=gparam['d_n'].value,
                              half_light_radius=gparam['d_hlr'].value * pixel_scale,
                              flux=gparam['d_flux'].value)
    disk = galsim.Chromatic(mono_disk, d_SED)
    gal = bulge+disk
    return gal

def profile_FWHM(profile, bandpass):
    image = galsim.ImageD(15*16, 15*16, scale=pixel_scale/16.0)
    profile.draw(bandpass, image=image)
    return FWHM(image)

def fig3_fiducial():
    '''Calculate `m` and `c` as a function of filter width for the fiducial bulge+disk galaxy.'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Running on fiducial galaxy parameters'
    print

    # build SEDs
    b_wave, b_flambda = np.genfromtxt(bulge_SED_file).T
    b_SED = galsim.SED(wave=b_wave, flambda=b_flambda)
    b_SED.setRedshift(redshift)

    d_wave, d_flambda = np.genfromtxt(disk_SED_file).T
    d_SED = galsim.SED(wave=d_wave, flambda=d_flambda)
    d_SED.setRedshift(redshift)

    # Define the PSF
    circ_base_PSF = galsim.Gaussian(half_light_radius=0.7 * pixel_scale)
    circ_PSF = galsim.ChromaticShiftAndDilate(circ_base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)
    base_PSF = circ_base_PSF.createSheared(g=PSF_ellip, beta=PSF_phi * galsim.radians)
    PSF = galsim.ChromaticShiftAndDilate(base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_fiducial.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        # build filter bandpass
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
        f_wave, f_throughput = np.genfromtxt(filter_file).T
        bandpass = galsim.Bandpass(f_wave, f_throughput)
        # scale SEDs and create composite SED
        b_SED.setFlux(bandpass, 1.0)
        d_SED.setFlux(bandpass, 1.0)
        c_SED = gparam['b_flux'].value * b_SED + gparam['d_flux'].value * d_SED

        # get PSF FWHM (for composite SED) to set galaxy scale
        star = galsim.Chromatic(galsim.Gaussian(fwhm=1.e-8), SED=c_SED)
        PSF_FWHM = profile_FWHM(galsim.Convolve(circ_PSF, star), bandpass)

        # Solve for galaxy scale that gives FWHM(convolved galaxy) / FWHM(PSF) = 1.4
        def resid(scale):
            gparam1 = fiducial_galaxy()
            gparam1['b_hlr'].value *= scale
            gparam1['d_hlr'].value *= scale
            circ_gal = param_to_circ_gal(gparam1, c_SED, c_SED)
            circ_final = galsim.Convolve(circ_PSF, circ_gal)
            gal_FWHM = profile_FWHM(circ_final, bandpass)
            return gal_FWHM/PSF_FWHM - 1.4
        scale = scipy.optimize.newton(resid, 1.0)
        gparam['b_hlr'].value *= scale
        gparam['d_hlr'].value *= scale

        #hack
        #gparam['b_hlr'].value = 1.13792527602
        #gparam['d_hlr'].value = 1.03447752365
        #hack

        print gparam['b_hlr'].value
        print gparam['d_hlr'].value
        #continue

        m, c = measure_shear_calib(gparam, bandpass, b_SED, d_SED, c_SED, PSF)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()


def fig3_redshift():
    '''Calculate `m` and `c` as a function of filter width, but change the redshift.'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    redshift = 1.4

    print
    print 'Varying the redshift'
    print

    # build SEDs
    b_wave, b_flambda = np.genfromtxt(bulge_SED_file).T
    b_SED = galsim.SED(wave=b_wave, flambda=b_flambda)
    b_SED.setRedshift(redshift)

    d_wave, d_flambda = np.genfromtxt(disk_SED_file).T
    d_SED = galsim.SED(wave=d_wave, flambda=d_flambda)
    d_SED.setRedshift(redshift)

    # Define the PSF
    circ_base_PSF = galsim.Gaussian(half_light_radius=0.7 * pixel_scale)
    circ_PSF = galsim.ChromaticShiftAndDilate(circ_base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)
    base_PSF = circ_base_PSF.createSheared(g=PSF_ellip, beta=PSF_phi * galsim.radians)
    PSF = galsim.ChromaticShiftAndDilate(base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_redshift.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        # build filter bandpass
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
        f_wave, f_throughput = np.genfromtxt(filter_file).T
        bandpass = galsim.Bandpass(f_wave, f_throughput)
        # scale SEDs and create composite SED
        b_SED.setFlux(bandpass, 1.0)
        d_SED.setFlux(bandpass, 1.0)
        c_SED = gparam['b_flux'].value * b_SED + gparam['d_flux'].value * d_SED

        # get PSF FWHM (for composite SED) to set galaxy scale
        star = galsim.Chromatic(galsim.Gaussian(fwhm=1.e-8), SED=c_SED)
        PSF_FWHM = profile_FWHM(galsim.Convolve(circ_PSF, star), bandpass)

        # Solve for galaxy scale that gives FWHM(convolved galaxy) / FWHM(PSF) = 1.4
        def resid(scale):
            gparam1 = fiducial_galaxy()
            gparam1['b_hlr'].value *= scale
            gparam1['d_hlr'].value *= scale
            circ_gal = param_to_circ_gal(gparam1, c_SED, c_SED)
            circ_final = galsim.Convolve(circ_PSF, circ_gal)
            gal_FWHM = profile_FWHM(circ_final, bandpass)
            return gal_FWHM/PSF_FWHM - 1.4
        scale = scipy.optimize.newton(resid, 1.0)
        gparam['b_hlr'].value *= scale
        gparam['d_hlr'].value *= scale

        #hack
        #gparam['b_hlr'].value = 1.13792527602
        #gparam['d_hlr'].value = 1.03447752365
        #hack

        print gparam['b_hlr'].value
        print gparam['d_hlr'].value
        #continue

        m, c = measure_shear_calib(gparam, bandpass, b_SED, d_SED, c_SED, PSF)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()


def fig3_bulge_radius():
    '''Calculate `m` and `c` as a function of filter width, but adjust the bulge radius such that
    b_hlr/d_hlr = 0.4.'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the bulge radius'
    print

    # build SEDs
    b_wave, b_flambda = np.genfromtxt(bulge_SED_file).T
    b_SED = galsim.SED(wave=b_wave, flambda=b_flambda)
    b_SED.setRedshift(redshift)

    d_wave, d_flambda = np.genfromtxt(disk_SED_file).T
    d_SED = galsim.SED(wave=d_wave, flambda=d_flambda)
    d_SED.setRedshift(redshift)

    # Define the PSF
    circ_base_PSF = galsim.Gaussian(half_light_radius=0.7 * pixel_scale)
    circ_PSF = galsim.ChromaticShiftAndDilate(circ_base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)
    base_PSF = circ_base_PSF.createSheared(g=PSF_ellip, beta=PSF_phi * galsim.radians)
    PSF = galsim.ChromaticShiftAndDilate(base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_bulge_radius.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        gparam['b_hlr'].value = gparam['b_hlr'].value * 0.4/1.1
        # build filter bandpass
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
        f_wave, f_throughput = np.genfromtxt(filter_file).T
        bandpass = galsim.Bandpass(f_wave, f_throughput)
        # scale SEDs and create composite SED
        b_SED.setFlux(bandpass, 1.0)
        d_SED.setFlux(bandpass, 1.0)
        c_SED = gparam['b_flux'].value * b_SED + gparam['d_flux'].value * d_SED

        # get PSF FWHM (for composite SED) to set galaxy scale
        star = galsim.Chromatic(galsim.Gaussian(fwhm=1.e-8), SED=c_SED)
        PSF_FWHM = profile_FWHM(galsim.Convolve(circ_PSF, star), bandpass)

        # Solve for galaxy scale that gives FWHM(convolved galaxy) / FWHM(PSF) = 1.4
        def resid(scale):
            gparam1 = fiducial_galaxy()
            gparam1['b_hlr'].value *= scale
            gparam1['d_hlr'].value *= scale
            circ_gal = param_to_circ_gal(gparam1, c_SED, c_SED)
            circ_final = galsim.Convolve(circ_PSF, circ_gal)
            gal_FWHM = profile_FWHM(circ_final, bandpass)
            return gal_FWHM/PSF_FWHM - 1.4
        scale = scipy.optimize.newton(resid, 1.0)
        gparam['b_hlr'].value *= scale
        gparam['d_hlr'].value *= scale

        #hack
        #gparam['b_hlr'].value = 1.13792527602
        #gparam['d_hlr'].value = 1.03447752365
        #hack

        print gparam['b_hlr'].value
        print gparam['d_hlr'].value
        #continue

        m, c = measure_shear_calib(gparam, bandpass, b_SED, d_SED, c_SED, PSF)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()


def fig3_disk_spectrum():
    '''Calculate `m` and `c` as a function of filter width, but change the disk spectrum.'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Im_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the disk spectrum'
    print

    # build SEDs
    b_wave, b_flambda = np.genfromtxt(bulge_SED_file).T
    b_SED = galsim.SED(wave=b_wave, flambda=b_flambda)
    b_SED.setRedshift(redshift)

    d_wave, d_flambda = np.genfromtxt(disk_SED_file).T
    d_SED = galsim.SED(wave=d_wave, flambda=d_flambda)
    d_SED.setRedshift(redshift)

    # Define the PSF
    circ_base_PSF = galsim.Gaussian(half_light_radius=0.7 * pixel_scale)
    circ_PSF = galsim.ChromaticShiftAndDilate(circ_base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)
    base_PSF = circ_base_PSF.createSheared(g=PSF_ellip, beta=PSF_phi * galsim.radians)
    PSF = galsim.ChromaticShiftAndDilate(base_PSF, dilate_fn = lambda w:(w/520.0)**0.6)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_disk_spectrum.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        # build filter bandpass
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
        f_wave, f_throughput = np.genfromtxt(filter_file).T
        bandpass = galsim.Bandpass(f_wave, f_throughput)
        # scale SEDs and create composite SED
        b_SED.setFlux(bandpass, 1.0)
        d_SED.setFlux(bandpass, 1.0)
        c_SED = gparam['b_flux'].value * b_SED + gparam['d_flux'].value * d_SED

        # get PSF FWHM (for composite SED) to set galaxy scale
        star = galsim.Chromatic(galsim.Gaussian(fwhm=1.e-8), SED=c_SED)
        PSF_FWHM = profile_FWHM(galsim.Convolve(circ_PSF, star), bandpass)

        # Solve for galaxy scale that gives FWHM(convolved galaxy) / FWHM(PSF) = 1.4
        def resid(scale):
            gparam1 = fiducial_galaxy()
            gparam1['b_hlr'].value *= scale
            gparam1['d_hlr'].value *= scale
            circ_gal = param_to_circ_gal(gparam1, c_SED, c_SED)
            circ_final = galsim.Convolve(circ_PSF, circ_gal)
            gal_FWHM = profile_FWHM(circ_final, bandpass)
            return gal_FWHM/PSF_FWHM - 1.4
        scale = scipy.optimize.newton(resid, 1.0)
        gparam['b_hlr'].value *= scale
        gparam['d_hlr'].value *= scale

        #hack
        #gparam['b_hlr'].value = 1.13792527602
        #gparam['d_hlr'].value = 1.03447752365
        #hack

        print gparam['b_hlr'].value
        print gparam['d_hlr'].value
        #continue

        m, c = measure_shear_calib(gparam, bandpass, b_SED, d_SED, c_SED, PSF)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

# def fig3_fiducial():
#     logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
#     logger = logging.getLogger("new_fig3")

#     b_wave, b_flambda = np.genfromtxt(bulge_SED_file).T
#     b_SED = galsim.SED(wave=b_wave, flambda=b_flambda)

#     d_wave, d_flambda = np.genfromtxt(disk_SED_file).T
#     d_SED = galsim.SED(wave=d_wave, flambda=d_flambda)

#     gparam0 = fiducial_galaxy()

#     print
#     print 'Running on fiducial galaxy parameters'
#     print

#     if not os.path.isdir('output/'):
#         os.mkdir('output/')
#     file_ = open('output/fig3_fiducial.dat', 'w')
#     for fw in [150, 250, 350, 450]:
#         gparam = gparam0.copy()
#         filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
#         f_wave, f_throughput = np.genfromtxt(filter_file).T
#         bandpass = galsim.Bandpass(f_wave, f_throughput)

#         # Step 0: set flux normalization of SEDs to 1.0, we'll control the relative
#         # flux of the bulge and disk through their monochromatic GSObject
#         # `flux` attributes.
#         b_SED.setFlux(bandpass, 1.0)
#         d_SED.setFlux(bandpass, 1.0)

#         # Create composite SED
#         c_wave = b_wave
#         c_flux = gparam0['b_flux'].value * b_SED(c_wave) + gparam0['d_flux'].value * d_SED(c_wave)
#         c_SED = galsim.SED(wave=c_wave, fphotons=c_flux)


#         # Step 7: Create target image generator
#         def gen_target_image(gamma, beta):
#             mono_bulge = galsim.Sersic(n=gparam['b_n'].value,
#                                        half_light_radius=gparam['b_hlr'].value * pixel_scale,
#                                        flux=gparam['b_flux'].value)
#             mono_bulge.applyShift(gparam['b_x0'].value * pixel_scale,
#                                   gparam['b_y0'].value * pixel_scale)
#             bulge = galsim.Chromatic(mono_bulge, b_SED)
#             bulge.applyShear(g=gparam['b_g'].value, beta=gparam['b_phi'].value * galsim.radians)
#             mono_disk = galsim.Sersic(n=gparam['d_n'].value,
#                                       half_light_radius=gparam['d_hlr'].value * pixel_scale,
#                                       flux=gparam['d_flux'].value)
#             mono_disk.applyShift(gparam['d_x0'].value * pixel_scale,
#                                  gparam['d_y0'].value * pixel_scale)
#             disk = galsim.Chromatic(mono_disk, d_SED)
#             disk.applyShear(g=gparam['d_g'].value, beta=gparam['d_phi'].value * galsim.radians)
#             gal = bulge+disk
#             final = galsim.Convolve([gal, PSF, pix])
#             target_image = galsim.ImageD(15, 15, scale=pixel_scale)
#             final.draw(bandpass, image=target_image)
#             return target_image

#         # I'm here...

#         # Step 7: Create function to make galaxy as a function of g
#         def test_image(g):
#             lmfit.report_errors(g)
#             mono_bulge = galsim.Sersic(n=g['b_n'].value,
#                                        half_light_radius=g['b_hlr'].value * pixel_scale,
#                                        flux=g['b_flux'].value)
#             mono_bulge.applyShift(g['b_x0'].value * pixel_scale,
#                                   g['b_y0'].value * pixel_scale)
#             bulge = galsim.Chromatic(mono_bulge, c_SED)
#             bulge.applyShear(g=g['b_g'].value, beta=g['b_phi'].value * galsim.radians)
#             mono_disk = galsim.Sersic(n=g['d_n'].value,
#                                       half_light_radius=g['d_hlr'].value * pixel_scale,
#                                       flux=g['d_flux'].value)
#             mono_disk.applyShift(g['d_x0'].value * pixel_scale,
#                                  g['d_y0'].value * pixel_scale)
#             disk = galsim.Chromatic(mono_disk, c_SED)
#             disk.applyShear(g=g['d_g'].value, beta=g['d_phi'].value * galsim.radians)
#             gal = bulge+disk
#             final = galsim.Convolve([gal, PSF, pix])
#             im = galsim.ImageD(15, 15, scale=pixel_scale)
#             final.draw(bandpass, image=im)
#             return im

#         # Step 8: Create residual function to optimize!
#         def image_resid(g):
#             image = test_image(g)
#             # imshow(hstack([image.array, target_image.array, target_image.array - image.array]))
#             # show()
#             return (image.array - target_image.array).flatten()

#         # Step 9: Optimize!
#         result = lmfit.minimize(image_resid, gparam)
#         lmfit.report_errors(gparam)


if __name__ == '__main__':
    fig3_fiducial()
    fig3_redshift()
    fig3_bulge_radius()
    fig3_disk_spectrum()
