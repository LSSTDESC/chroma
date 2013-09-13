import sys
import pickle

import numpy
import lmfit
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits

def encode_obshistid(SED_type, filter_name, zenith, seed, redshift):
    SED_types = {'G5v':'1', 'star':'2', 'gal':'3'}
    SED_digit = SED_types[SED_type]
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    redshift_digits = '{:02d}'.format(int(round((redshift / 0.03))))
    return SED_digit + filter_digit + zenith_digit + seed_digit + redshift_digits

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

def measure_moffat_fwhm(SED_type, filter_name, zenith, seed, redshift):
    # what will be the output of this function?
    # FWHM_x, FWHM_y, beta for each star?

    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    obshistid = encode_obshistid(SED_type, filter_name, zenith, seed, redshift)
    print obshistid

    if SED_type == 'G5v':
        obj_types = ['G5v']*8
    elif SED_type == 'star':
        obj_types = ['uko5v',
                     'ukb5iii',
                     'uka5v',
                     'ukf5v',
                     'ukg5v',
                     'ukk5v',
                     'ukm5v',
                     'ukg5v'] #extra G5v star to make 8
    elif SED_type == 'gal':
        obj_types = ['E',
                     'Sa',
                     'Sb',
                     'Sbc',
                     'Scd',
                     'Im',
                     'SB1',
                     'SB6']
    else:
        print 'error'
        sys.exit()

    image_file = 'output/lsst_e_{}_f{}_R22_S11_E000.fits.gz'
    image_file = image_file.format(obshistid, filter_number[filter_name])

    try:
        hdulist = fits.open(image_file)
    except:
        return (numpy.nan, numpy.nan, numpy.nan)
    w = wcs.WCS(hdulist[0].header)

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    values = numpy.empty(len(RAs) * len(DECs),
                         dtype=[('type', numpy.str_, 8),
                                ('fwhm','f4'),
                                ('beta', 'f4'),
                                ('q', 'f4'),
                                ('phi', 'f4'),
                                ('flux', 'f4')])

    i=0
    for RA, typ in zip(RAs, obj_types):
        for DEC in DECs:
            center = w.wcs_world2pix(numpy.array([[RA, DEC]], numpy.float_),0)
            thumb = hdulist[0].data[center[0,1]-30:center[0,1]+30, center[0,0]-30:center[0,0]+30]
            flux, xbar, ybar, Ixx, Iyy, Ixy = moments(thumb)
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

            values[i] = (typ,
                         result.params['fwhm'].value,
                         result.params['beta'].value,
                         result.params['q'].value,
                         result.params['phi'].value,
                         result.params['flux'].value)
            i += 1

    return values

def main():
    star_r = measure_moffat_fwhm('star', 'r', 0.0, 1000, 0.0)
    pickle.dump(star_r, open('pickles/star.r.pik', 'w'))
    star_i = measure_moffat_fwhm('star', 'i', 0.0, 1000, 0.0)
    pickle.dump(star_i, open('pickles/star.i.pik', 'w'))
    G5v_r = measure_moffat_fwhm('G5v', 'r', 0.0, 1000, 0.0)
    pickle.dump(G5v_r, open('pickles/G5v.r.pik', 'w'))
    G5v_i = measure_moffat_fwhm('G5v', 'i', 0.0, 1000, 0.0)
    pickle.dump(G5v_i, open('pickles/G5v.i.pik', 'w'))
    for z in numpy.arange(0.0, 3.0, 0.03):
        rdat = measure_moffat_fwhm('gal', 'r', 0.0, 1000, z)
        pickle.dump(rdat, open('pickles/gal.r.{:02d}.pik'.format(int(round(z / 0.03))), 'w'))
        idat = measure_moffat_fwhm('gal', 'i', 0.0, 1000, z)
        pickle.dump(idat, open('pickles/gal.i.{:02d}.pik'.format(int(round(z / 0.03))), 'w'))

if __name__ == '__main__':
    main()
