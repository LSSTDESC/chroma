import numpy
import pyfits
import matplotlib.pyplot as plt
import lmfit

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

def estimate_moffat_params(imfile, catfile):
    im = pyfits.getdata(imfile)
    cat = pyfits.getdata(catfile)

    betas = numpy.empty(0)
    fwhm_xs = numpy.empty(0)
    fwhm_ys = numpy.empty(0)
    fwhms = numpy.empty(0)

    for obj in cat:
        xwin = obj['XWIN_IMAGE']
        ywin = obj['YWIN_IMAGE']
        thumb = im[ywin-30:ywin+30, xwin-30:xwin+30]

        params = lmfit.Parameters()
        params.add('fwhm_x', value=5.0)
        params.add('fwhm_y', value=5.0)
        params.add('beta', value=2.5)
        params.add('peak', value=10000)
        params.add('x0', 30)
        params.add('y0', 30)

        def resid(p):
            profile = moffat2d(p)
            xs = numpy.arange(thumb.shape[1])
            ys = numpy.arange(thumb.shape[0])
            xs, ys = numpy.meshgrid(xs, ys)
            profile_im = profile(ys, xs)
            return (profile_im - thumb).flatten()
        result = lmfit.minimize(resid, params)
        #lmfit.report_errors(params)
        print '{} {} {}'.format(result.params['beta'].value,
                                result.params['fwhm_x'].value,
                                result.params['fwhm_y'].value)
        betas = numpy.append(betas, result.params['beta'].value)
        fwhm_xs = numpy.append(fwhm_xs, result.params['fwhm_x'].value)
        fwhm_ys = numpy.append(fwhm_ys, result.params['fwhm_y'].value)
        fwhms = numpy.append(fwhms, numpy.sqrt(result.params['fwhm_x'].value * \
                                               result.params['fwhm_y'].value))

    print 'averages'
    print '{} {} {}'.format(betas.mean(),
                            fwhm_xs.mean(),
                            fwhm_ys.mean())
    print 'geometric average fwhm'
    print fwhms.mean()






if __name__ == '__main__':
    imfile = 'output/eimage_123000_f2_R22_S11_E000.fits.gz'
    catfile = 'output/123000_cat_V.fits.gz'
    estimate_moffat_params(imfile, catfile)
