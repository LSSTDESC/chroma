import copy

import numpy
import lmfit

import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits

def encode_obshistid(mono_wave, filter_name, zenith, seed):
    #first four digits are wavelength in nm
    wave_digits = str(int(round(mono_wave)))
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return wave_digits + filter_digit + zenith_digit + seed_digit

def moffat2d(params):
    fwhm_x = params['fwhm_x'].value
    fwhm_y = params['fwhm_y'].value
    beta = params['beta'].value
    peak = params['peak'].value
    x0 = params['x0'].value
    y0 = params['y0'].value

    alpha_x = fwhm_x / (2.0 * numpy.sqrt(2.0**(1.0 / beta) - 1.0))
    alpha_y = fwhm_y / (2.0 * numpy.sqrt(2.0**(1.0 / beta) - 1.0))
    def f(y, x):
        u = ((x - x0) / alpha_x)**2.0 + ((y - y0) / alpha_y)**2.0
        p = 1.0 / ((u + 1.0)**beta)
        return peak*p/p.max()
    return f

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

def measure_moffat_fwhm(mono_wave, filter_name, zenith, seed):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    obshistid = encode_obshistid(mono_wave, filter_name, zenith, seed)
    image_file = 'output/eimage_{}_f{}_R22_S11_E000.fits.gz'
    image_file = image_file.format(obshistid, filter_number[filter_name])

    print 'opening: {}'.format(image_file)
    hdulist = fits.open(image_file)
    w = wcs.WCS(hdulist[0].header)

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    values = {'fwhm_x':numpy.empty(0), 'fwhm_y':numpy.empty(0), 'beta':numpy.empty(0)}
    for RA in RAs:
        for DEC in DECs:
            center = w.wcs_world2pix(numpy.array([[RA, DEC]], numpy.float_),0)
            thumb = hdulist[0].data[center[0,1]-30:center[0,1]+30, center[0,0]-30:center[0,0]+30]

            xbar, ybar, Ixx, Iyy, Ixy = moments(thumb)

            params = lmfit.Parameters()
            params.add('fwhm_x', value=numpy.sqrt(Ixx))
            params.add('fwhm_y', value=numpy.sqrt(Iyy))
            params.add('beta', value=2.5)
            params.add('peak', value=thumb.max())
            params.add('x0', xbar)
            params.add('y0', ybar)

            def resid(p):
                profile = moffat2d(p)
                xs = numpy.arange(thumb.shape[1])
                ys = numpy.arange(thumb.shape[0])
                xs, ys = numpy.meshgrid(xs, ys)
                profile_im = profile(ys, xs)
                return (profile_im - thumb).flatten()
            result = lmfit.minimize(resid, params)
            values['fwhm_x'] = numpy.append(values['fwhm_x'], result.params['fwhm_x'].value)
            values['fwhm_y'] = numpy.append(values['fwhm_y'], result.params['fwhm_y'].value)
            values['beta'] = numpy.append(values['beta'], result.params['beta'].value)

    print mono_wave
    outstring = 'fwhm_x = {} +/- {}, fwhm_y = {} +/- {}, beta = {} +/- {}'
#    import ipdb; ipdb.set_trace()
    outstring = outstring.format(numpy.mean(values['fwhm_x']), numpy.std(values['fwhm_x']),
                                 numpy.mean(values['fwhm_y']), numpy.std(values['fwhm_y']),
                                 numpy.mean(values['beta']), numpy.std(values['beta']))
    print outstring
    return (numpy.mean(values['fwhm_x']), numpy.std(values['fwhm_x']),
            numpy.mean(values['fwhm_y']), numpy.std(values['fwhm_y']),
            numpy.mean(values['beta']), numpy.std(values['beta']))

values0 = {'waves':numpy.empty(0),
           'fwhm_x':numpy.empty(0), 'fwhm_x_err':numpy.empty(0),
           'fwhm_y':numpy.empty(0), 'fwhm_y_err':numpy.empty(0),
           'beta':numpy.empty(0), 'beta_err':numpy.empty(0)}
values = {'u':copy.deepcopy(values0),
          'g':copy.deepcopy(values0),
          'r':copy.deepcopy(values0),
          'i':copy.deepcopy(values0),
          'z':copy.deepcopy(values0),
          'Y':copy.deepcopy(values0)}

for w in numpy.arange(300, 401, 25):
    result = measure_moffat_fwhm(w, 'u', 0, 1000)
    values['u']['waves'] = numpy.append(values['u']['waves'], w)
    values['u']['fwhm_x'] = numpy.append(values['u']['fwhm_x'], result[0])
    values['u']['fwhm_y'] = numpy.append(values['u']['fwhm_y'], result[2])
    values['u']['beta'] = numpy.append(values['u']['beta'], result[4])
    values['u']['fwhm_x_err'] = numpy.append(values['u']['fwhm_x_err'], result[1])
    values['u']['fwhm_y_err'] = numpy.append(values['u']['fwhm_y_err'], result[3])
    values['u']['beta_err'] = numpy.append(values['u']['beta_err'], result[5])
for w in numpy.arange(400, 551, 25):
    result = measure_moffat_fwhm(w, 'g', 0, 1000)
    values['g']['waves'] = numpy.append(values['g']['waves'], w)
    values['g']['fwhm_x'] = numpy.append(values['g']['fwhm_x'], result[0])
    values['g']['fwhm_y'] = numpy.append(values['g']['fwhm_y'], result[2])
    values['g']['beta'] = numpy.append(values['g']['beta'], result[4])
    values['g']['fwhm_x_err'] = numpy.append(values['g']['fwhm_x_err'], result[1])
    values['g']['fwhm_y_err'] = numpy.append(values['g']['fwhm_y_err'], result[3])
    values['g']['beta_err'] = numpy.append(values['g']['beta_err'], result[5])
for w in numpy.arange(550, 701, 25):
    result = measure_moffat_fwhm(w, 'r', 0, 1000)
    values['r']['waves'] = numpy.append(values['r']['waves'], w)
    values['r']['fwhm_x'] = numpy.append(values['r']['fwhm_x'], result[0])
    values['r']['fwhm_y'] = numpy.append(values['r']['fwhm_y'], result[2])
    values['r']['beta'] = numpy.append(values['r']['beta'], result[4])
    values['r']['fwhm_x_err'] = numpy.append(values['r']['fwhm_x_err'], result[1])
    values['r']['fwhm_y_err'] = numpy.append(values['r']['fwhm_y_err'], result[3])
    values['r']['beta_err'] = numpy.append(values['r']['beta_err'], result[5])
for w in numpy.arange(675, 826, 25):
    result = measure_moffat_fwhm(w, 'i', 0, 1000)
    values['i']['waves'] = numpy.append(values['i']['waves'], w)
    values['i']['fwhm_x'] = numpy.append(values['i']['fwhm_x'], result[0])
    values['i']['fwhm_y'] = numpy.append(values['i']['fwhm_y'], result[2])
    values['i']['beta'] = numpy.append(values['i']['beta'], result[4])
    values['i']['fwhm_x_err'] = numpy.append(values['i']['fwhm_x_err'], result[1])
    values['i']['fwhm_y_err'] = numpy.append(values['i']['fwhm_y_err'], result[3])
    values['i']['beta_err'] = numpy.append(values['i']['beta_err'], result[5])
for w in numpy.arange(800, 951, 25):
    result = measure_moffat_fwhm(w, 'z', 0, 1000)
    values['z']['waves'] = numpy.append(values['z']['waves'], w)
    values['z']['fwhm_x'] = numpy.append(values['z']['fwhm_x'], result[0])
    values['z']['fwhm_y'] = numpy.append(values['z']['fwhm_y'], result[2])
    values['z']['beta'] = numpy.append(values['z']['beta'], result[4])
    values['z']['fwhm_x_err'] = numpy.append(values['z']['fwhm_x_err'], result[1])
    values['z']['fwhm_y_err'] = numpy.append(values['z']['fwhm_y_err'], result[3])
    values['z']['beta_err'] = numpy.append(values['z']['beta_err'], result[5])
for w in numpy.arange(900, 1101, 25):
    result = measure_moffat_fwhm(w, 'Y', 0, 1000)
    values['Y']['waves'] = numpy.append(values['Y']['waves'], w)
    values['Y']['fwhm_x'] = numpy.append(values['Y']['fwhm_x'], result[0])
    values['Y']['fwhm_y'] = numpy.append(values['Y']['fwhm_y'], result[2])
    values['Y']['beta'] = numpy.append(values['Y']['beta'], result[4])
    values['Y']['fwhm_x_err'] = numpy.append(values['Y']['fwhm_x_err'], result[1])
    values['Y']['fwhm_y_err'] = numpy.append(values['Y']['fwhm_y_err'], result[3])
    values['Y']['beta_err'] = numpy.append(values['Y']['beta_err'], result[5])

# fwhm plot

fig = plt.figure()
ax = plt.subplot(111)
ax.errorbar(values['u']['waves'], values['u']['fwhm_x'], values['u']['fwhm_x_err'],
            ls='none', marker='o', color='violet')
ax.errorbar(values['u']['waves']+1, values['u']['fwhm_y'], values['u']['fwhm_y_err'],
            ls='none', marker='o', color='violet')
ax.errorbar(values['g']['waves'], values['g']['fwhm_x'], values['g']['fwhm_x_err'],
            ls='none', marker='o', color='blue')
ax.errorbar(values['g']['waves']+1, values['g']['fwhm_y'], values['g']['fwhm_y_err'],
            ls='none', marker='o', color='blue')
ax.errorbar(values['r']['waves'], values['r']['fwhm_x'], values['r']['fwhm_x_err'],
            ls='none', marker='o', color='green')
ax.errorbar(values['r']['waves']+1, values['r']['fwhm_y'], values['r']['fwhm_y_err'],
            ls='none', marker='o', color='green')
ax.errorbar(values['i']['waves'], values['i']['fwhm_x'], values['i']['fwhm_x_err'],
            ls='none', marker='o', color='red')
ax.errorbar(values['i']['waves']+1, values['i']['fwhm_y'], values['i']['fwhm_y_err'],
            ls='none', marker='o', color='red')
ax.errorbar(values['z']['waves'], values['z']['fwhm_x'], values['z']['fwhm_x_err'],
            ls='none', marker='o', color='brown')
ax.errorbar(values['z']['waves']+1, values['z']['fwhm_y'], values['z']['fwhm_y_err'],
            ls='none', marker='o', color='brown')
ax.errorbar(values['Y']['waves'], values['Y']['fwhm_x'], values['Y']['fwhm_x_err'],
            ls='none', marker='o', color='black')
ax.errorbar(values['Y']['waves']+1, values['Y']['fwhm_y'], values['Y']['fwhm_y_err'],
            ls='none', marker='o', color='black')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('FWHM (pixels)')

x=numpy.linspace(300, 1100, 100)
y=values['r']['fwhm_x'][0] * (x/550.)**(-0.2)
ax.plot(x, y)
plt.show()

fig2 = plt.figure()
ax = plt.subplot(111)
ax.errorbar(values['u']['waves'], values['u']['beta'], values['u']['beta_err'],
            ls='none', marker='o', color='violet')
ax.errorbar(values['g']['waves'], values['g']['beta'], values['g']['beta_err'],
            ls='none', marker='o', color='blue')
ax.errorbar(values['r']['waves'], values['r']['beta'], values['r']['beta_err'],
            ls='none', marker='o', color='green')
ax.errorbar(values['i']['waves'], values['i']['beta'], values['i']['beta_err'],
            ls='none', marker='o', color='red')
ax.errorbar(values['z']['waves'], values['z']['beta'], values['z']['beta_err'],
            ls='none', marker='o', color='brown')
ax.errorbar(values['Y']['waves'], values['Y']['beta'], values['Y']['beta_err'],
            ls='none', marker='o', color='black')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel(r'$\beta$')
plt.show()
