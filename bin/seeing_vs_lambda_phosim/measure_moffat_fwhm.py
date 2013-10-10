import sys
import pickle

import numpy
import numpy.lib.recfunctions
import lmfit
#import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits

def encode_obshistid(mode, mono_wave, filter_name, zenith, seed):
    mode_digits = str(mode)
    wave_digits = '{:04d}'.format(int(round(mono_wave)))
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = '{:03d}'.format(seed - 1000)
    return mode_digits + wave_digits + filter_digit + zenith_digit + seed_digit

def moffat2d(params):
    fwhm = params['fwhm'].value
    beta = params['beta'].value
    q = params['q'].value
    phi = params['phi'].value
    flux = params['flux'].value
    x0 = params['x0'].value
    y0 = params['y0'].value

    alpha = fwhm / (2.0 * numpy.sqrt(2.0**(1.0 / beta) - 1.0))
    a = alpha / numpy.sqrt(q)
    b = alpha * numpy.sqrt(q)
    cph = numpy.cos(phi)
    sph = numpy.sin(phi)
    C11 = cph**2 / a**2 + sph**2 / b**2
    C12 = 0.5 * (1.0 / a**2 - 1.0 / b**2) * numpy.sin(2.0 * phi)
    C22 = sph**2 / a**2 + cph**2 / b**2

    coeff = flux * (beta - 1.0) / (numpy.pi * alpha**2)

    def f(y, x):
        u = 1.0 + (x - x0)**2 * C11 + 2.0 * (x - x0) * (y - y0) * C12 + (y - y0)**2 * C22
        return coeff * u**(-beta)
    return f

def moments(data, pixsize=1.0):
    '''Compute first and second (quadrupole) moments of `data`.  Scales result by `pixsize` for
    non-unit width pixels.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    '''
    ys, xs = numpy.mgrid[0:data.shape[0], 0:data.shape[1]] * pixsize
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs-xbar)**2).sum() / total
    Iyy = (data * (ys-ybar)**2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return total, xbar, ybar, Ixx, Iyy, Ixy

def moments_to_ellipse(Ixx, Ixy, Iyy):
    r = numpy.sqrt(numpy.sqrt(4.0 * (Ixx*Iyy - Ixy**2))) # determinant
    phi = 0.5 * numpy.arctan2(Iyy - Ixx, 2*Ixy)
    discriminant = 4.0 / r**4 * (Ixx + Iyy)**2 - 4.0
    q = 0.5 * ((2 / r**2) * (Ixx + Iyy) + numpy.sqrt(discriminant))
    return r, q, phi

def measure_moffat_fwhm(mode, mono_wave, filter_name, zenith, seed):
    if mode > 2:
        print 'invalid mode'
        sys.exit()
    out = numpy.zeros(1, dtype=[('fwhm', 'f4'),
                                ('fwhm_err', 'f4'),
                                ('beta', 'f4'),
                                ('beta_err', 'f4'),
                                ('q', 'f4'),
                                ('q_err', 'f4'),
                                ('phi','f4'),
                                ('phi_err','f4'),
                                ('flux','f4'),
                                ('flux_err','f4')])
    out[0] = (numpy.NaN,)*10
    if mono_wave == 500:
        return out
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    obshistid = encode_obshistid(mode, mono_wave, filter_name, zenith, seed)
    image_file = 'output/lsst_e_{}_f{}_R22_S11_E000.fits.gz'
    image_file = image_file.format(obshistid, filter_number[filter_name])

    try:
        hdulist = fits.open(image_file)
    except:
        return out
    w = wcs.WCS(hdulist[0].header)

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    values = numpy.empty(len(RAs) * len(DECs),
                         dtype=[('fwhm','f4'),
                                ('beta','f4'),
                                ('q','f4'),
                                ('phi','f4'),
                                ('flux','f4'),
                                ('x0','f4'),
                                ('y0', 'f4')])

    i=0
    for RA in RAs:
        for DEC in DECs:
            center = w.wcs_world2pix(numpy.array([[RA, DEC]], numpy.float_),0)
            thumb = hdulist[0].data[center[0,1]-30:center[0,1]+30, center[0,0]-30:center[0,0]+30]
            flux, xbar, ybar, Ixx, Iyy, Ixy = moments(thumb)
            if flux == 0.0:
                return out
            r, q, phi = moments_to_ellipse(Ixx, Ixy, Iyy)

            params = lmfit.Parameters()
            params.add('fwhm', value=r * 0.8, min=0.0, max=5.0)
            params.add('beta', value=2.7, min=2.0, max=5.0)
            params.add('q', value=q, min=0.0, max=1.0)
            params.add('phi', value=4.0 * numpy.pi, min=0.0, max=8.0 * numpy.pi, vary=False)
            params.add('flux', value=flux, min=0.0)
            params.add('x0', value=xbar)
            params.add('y0', value=ybar)

            xs, ys = numpy.meshgrid(numpy.arange(thumb.shape[1]), numpy.arange(thumb.shape[0]))

            def resid(p):
                return (thumb - moffat2d(p)(ys, xs)).flatten()
            result = lmfit.minimize(resid, params)
            params['phi'].vary=True
            result = lmfit.minimize(resid, params)

            while result.params['phi'].value > 2.0 * numpy.pi:
                result.params['phi'].value -= 2.0 * numpy.pi

            values[i] = (result.params['fwhm'].value,
                         result.params['beta'].value,
                         result.params['q'].value,
                         result.params['phi'].value,
                         result.params['flux'].value,
                         result.params['x0'].value,
                         result.params['y0'].value)
            i += 1

    print mode, mono_wave, filter_name
    outstring = 'fwhm = {:5.3f} +/- {:5.3f}    flux = {:010f}'
    print outstring.format(numpy.mean(values['fwhm']),
                           numpy.std(values['fwhm']),
                           numpy.mean(values['flux']))

    out['fwhm'] = numpy.mean(values['fwhm'])
    out['fwhm_err'] = numpy.std(values['fwhm'])
    out['beta'] = numpy.mean(values['beta'])
    out['beta_err'] = numpy.std(values['beta'])
    out['q'] = numpy.mean(values['q'])
    out['q_err'] = numpy.std(values['q'])
    out['phi'] = numpy.mean(values['phi'])
    out['phi_err'] = numpy.std(values['phi'])
    out['flux'] = numpy.mean(values['flux'])
    out['flux_err'] = numpy.std(values['flux'])

    return out

def main():
    waves = {'u': numpy.arange(300, 401, 25),
             'g': numpy.arange(375, 576, 25),
             'r': numpy.arange(525, 726, 25),
             'i': numpy.arange(650, 851, 25),
             'z': numpy.arange(775, 976, 25),
             'Y': numpy.arange(875, 1100, 25)}
    # waves = {'u': numpy.arange(325, 351, 25)}

    nfiles = sum(map(len, waves.values())) * 2

    values = numpy.empty(nfiles,
                         dtype=[('wave', 'i4'),
                                ('filter', 'a1'),
                                ('mode', 'i4'),
                                ('fwhm', 'f4'),
                                ('fwhm_err', 'f4'),
                                ('beta', 'f4'),
                                ('beta_err', 'f4'),
                                ('q', 'f4'),
                                ('q_err', 'f4'),
                                ('phi','f4'),
                                ('phi_err','f4'),
                                ('flux','f4'),
                                ('flux_err','f4')])

    values[:] = numpy.nan

    i = 0
    for f, ws in waves.iteritems():
        for w in ws:
            r1 = measure_moffat_fwhm(1, w, f, 0, 1000)
            values[i] = (w, f, 1,
                         r1['fwhm'], r1['fwhm_err'],
                         r1['beta'], r1['beta_err'],
                         r1['q'], r1['q_err'],
                         r1['phi'], r1['phi_err'],
                         r1['flux'], r1['flux_err'])
            i += 1
            r1 = measure_moffat_fwhm(2, w, f, 0, 1000)
            values[i] = (w, f, 2,
                         r1['fwhm'], r1['fwhm_err'],
                         r1['beta'], r1['beta_err'],
                         r1['q'], r1['q_err'],
                         r1['phi'], r1['phi_err'],
                         r1['flux'], r1['flux_err'])
            i += 1

    # ignore 500nm since spectrum here is wrong.
    values['fwhm'][values['wave'] == 500] = numpy.nan
    values['fwhm_err'][values['wave'] == 500] = numpy.nan
    values['beta'][values['wave'] == 500] = numpy.nan
    values['beta_err'][values['wave'] == 500] = numpy.nan
    values['q'][values['wave'] == 500] = numpy.nan
    values['q_err'][values['wave'] == 500] = numpy.nan
    values['phi'][values['wave'] == 500] = numpy.nan
    values['phi_err'][values['wave'] == 500] = numpy.nan
    values['flux'][values['wave'] == 500] = numpy.nan
    values['flux_err'][values['wave'] == 500] = numpy.nan

    pickle.dump(values, open('seeing_vs_wave.pik', 'w'))

if __name__ == '__main__':
    main()
