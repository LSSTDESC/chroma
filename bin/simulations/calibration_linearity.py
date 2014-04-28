import copy

import numpy
import lmfit
import scipy.integrate
import matplotlib.pyplot as plt

import _mypath
import chroma

data_dir = '../../../data/'

# wiki.python.org/moin/PythonDecoratorLibrary#Memoize
import collections
import functools
class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


def measure_gamma_hat(gparam, gal_PSF, star_PSF, s_engine, gamma0):
    galtool = chroma.GalTools.SGalTool(s_engine)

    def gen_target_image(gamma, beta):
        ring_gparam = galtool.get_ring_params(gparam, gamma, beta)
        return s_engine.get_image(ring_gparam, gal_PSF)

    def measure_ellip(target_image, init_param):
        def resid(param):
            im = s_engine.get_image(param, star_PSF)
            return (im - target_image).flatten()
        result = lmfit.minimize(resid, init_param)
        g = result.params['g'].value
        phi = result.params['phi'].value
        c_ellip = g * complex(numpy.cos(2.0 * phi), numpy.sin(2.0 * phi))
        return c_ellip

    def get_ring_params(gamma, beta):
        return galtool.get_ring_params(gparam, gamma, beta)

    return chroma.utils.ringtest(gamma0, 3,
                                 gen_target_image,
                                 get_ring_params,
                                 measure_ellip, silent=True)

def calibration_linearity(filter_name, gal, z, star, n, e, zenith=30*numpy.pi/180):
    s_engine = chroma.ImageEngine.GalSimSEngine(size=41)
    PSF_model = chroma.PSF_model.GSAtmPSF
    PSF_ellip = 0.0
    PSF_phi = 0.0
    filter_file = data_dir+'filters/LSST_{}.dat'.format(filter_name)
    gal_SED_file = data_dir+'SEDs/{}.ascii'.format(gal)
    star_SED_file = data_dir+'SEDs/{}.ascii'.format(star)

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    # swave = swave[::50]
    # sphotons = sphotons[::50]
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = PSF_model(swave, sphotons, zenith=zenith)
    smom = chroma.disp_moments(swave, sphotons, zenith=zenith)

    galtool = chroma.GalTools.SGalTool(s_engine)
    gparam = galtool.default_galaxy()
    gparam['g'].value = e
    gparam['n'].value = n
    # gparam['g'].value = 0.0

    # normalize size to second moment (before PSF convolution)
    print 'n: {}'.format(gparam['n'].value)
    print 'fiducial r_e: {}'.format(gparam['r_e'].value)
    print 'setting second moment radius to 0.27 arcseconds = 1.35 pixels'
    gparam = galtool.set_uncvl_r2(gparam, (0.27/0.2)**2) # (0.27 arcsec)^2 -> pixels^2
    print 'output r2: {}'.format(galtool.get_uncvl_r2(gparam))
    print 'output r: {}'.format(numpy.sqrt(galtool.get_uncvl_r2(gparam)))
    print 'output r_e:{}'.format(gparam['r_e'].value)

    gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, z)
    gphotons /= scipy.integrate.simps(gphotons, gwave)
    gmom = chroma.disp_moments(gwave, gphotons, zenith=zenith)
    m_PB12 = - (gmom[1] - smom[1]) * (3600 * 180 / numpy.pi)**2 / (0.27**2)
    c_PB12 = complex(m_PB12/2, 0)
    m_PB12 = complex(m_PB12, m_PB12)

    gal_PSF = PSF_model(gwave, gphotons, zenith=zenith)


    @memoized
    def gamma_hat(gamma):
        gparam1 = copy.deepcopy(gparam)
        return measure_gamma_hat(gparam1, gal_PSF, star_PSF, s_engine, gamma)

    g1s = numpy.arange(0.0, 0.061, 0.01)
    g2s = numpy.arange(0.0, 0.061, 0.01)
    markers = 's*oD+^x'

    # g1 plot:
    f1 = plt.figure(figsize=(8,6))
    f1.subplots_adjust(hspace=0)
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.get_xaxis().set_visible(False)
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax1.set_xlim(-0.01, 0.07)
    ax1.set_ylim(-0.01, 0.07)
    ax2.set_xlim(-0.01, 0.07)
    ax1.set_ylabel('$\hat{\gamma_1}$')
    ax2.set_xlabel('$\gamma_1$')
    ax2.set_ylabel('obs - PB12')
    ax1.set_title('n = {}, e = {}'.format(n, e))

    # PB12 analytic results
    ax1.plot(g1s, [c_PB12.real + (1.0 + m_PB12.real) * g for g in g1s])

    vals = numpy.array([])
    xs = numpy.array([])
    ys = numpy.array([])
    for g2, m in zip(g2s, markers):
        gs = [complex(g1, g2) for g1 in g1s]
        ax1.scatter([g.real for g in gs], [gamma_hat(g).real for g in gs],
                    marker=m, label=g2)
        resids = [gamma_hat(g).real - (c_PB12.real + (1.0 + m_PB12.real) * g.real) for g in gs]
        ax2.scatter([g.real for g in gs], resids, marker=m)
        #HACK!
        if g2 == 0.0:
            vals = numpy.append(vals, resids[1:])
            xs = numpy.append(xs, [g.real for g in gs])
            ys = numpy.append(ys, [gamma_hat(g).real for g in gs])
        else:
            vals = numpy.append(vals, resids)
            xs = numpy.append(xs, [g.real for g in gs])
            ys = numpy.append(ys, [gamma_hat(g).real for g in gs])
    ax1.legend(title='$\gamma_2$', fontsize=9)

    # fit line to the data points
    A = numpy.vstack([xs, numpy.ones(len(xs))]).T
    m, c = numpy.linalg.lstsq(A, ys)[0]
    m -= 1
    # overplot fit
    pts = numpy.array([0.00, 0.06])
    ax1.plot(pts, pts * (1.0 + m) + c, c='green')
    # overprint results
    text = 'PB12     : $\hat{{\gamma_1}} = (1 + {0:8.5f}) \gamma_1 + {1:8.5f}$'
    text = text.format(m_PB12.real, c_PB12.real)
    ax1.text(-0.005, 0.06, text, color='blue')
    text = 'ring test: $\hat{{\gamma_1}} = (1 + {0:8.5f}) \gamma_1 + {1:8.5f}$'.format(m, c)
    ax1.text(-0.005, 0.05, text, color='green')

    # compute yrange for residuals plot
    # yrange = [-numpy.abs(vals).max(), numpy.abs(vals).max()]
    yrange = [vals.min(), vals.max()]
    yspan = yrange[1] - yrange[0]
    yrange = [yrange[0]-yspan*0.1, yrange[1] + yspan*0.1]
    ax2.set_ylim(yrange)
    plt.savefig('output/cal.g1.n{}.e{}.pdf'.format(n, e), dpi=220)


    # g2 plot:
    f2 = plt.figure(figsize=(8,6), dpi=100)
    f2.subplots_adjust(hspace=0)
    ax3 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax3.get_xaxis().set_visible(False)
    ax4 = plt.subplot2grid((3,1), (2,0))
    ax3.set_xlim(-0.01, 0.07)
    ax3.set_ylim(-0.01, 0.07)
    ax4.set_xlim(-0.01, 0.07)
    ax3.set_ylabel('$\hat{\gamma_2}$')
    ax4.set_xlabel('$\gamma_2$')
    ax4.set_ylabel('obs - PB12')
    ax3.set_title('n = {}, e = {}'.format(n, e))
    ax3.plot(g1s, [c_PB12.imag + (1.0 + m_PB12.imag) * g for g in g1s])

    vals = numpy.array([])
    xs = numpy.array([])
    ys = numpy.array([])
    for g1, m in zip(g1s, markers):
        gs = [complex(g1, g2) for g2 in g2s]
        ax3.scatter([g.imag for g in gs], [gamma_hat(g).imag for g in gs],
                    marker=m, label=g1)
        resids = [gamma_hat(g).imag - (c_PB12.imag + (1.0 + m_PB12.imag) * g.imag) for g in gs]
        ax4.scatter([g.imag for g in gs], resids, marker=m)
        vals = numpy.append(vals, resids)
        xs = numpy.append(xs, [g.imag for g in gs])
        ys = numpy.append(ys, [gamma_hat(g).imag for g in gs])
    ax3.legend(title='$\gamma_1$', fontsize=9)

    # fit line to the data points
    A = numpy.vstack([xs, numpy.ones(len(xs))]).T
    m, c = numpy.linalg.lstsq(A, ys)[0]
    m -= 1
    # overplot fit
    pts = numpy.array([0.00, 0.06])
    ax3.plot(pts, pts * (1.0 + m) + c, c='green')
    # overprint results
    text = 'PB12     : $\hat{{\gamma_2}} = (1 + {0:8.5f}) \gamma_2 + {1:8.5f}$'
    text = text.format(m_PB12.imag, c_PB12.imag)
    ax3.text(-0.005, 0.06, text, color='blue')
    text = 'ring test: $\hat{{\gamma_2}} = (1 + {0:8.5f}) \gamma_2 + {1:8.5f}$'.format(m, c)
    ax3.text(-0.005, 0.05, text, color='green')

    # yrange = [-numpy.abs(vals).max(), numpy.abs(vals).max()]
    yrange = [vals.min(), vals.max()]
    yspan = yrange[1] - yrange[0]
    yrange = [yrange[0]-yspan*0.1, yrange[1] + yspan*0.1]
    ax4.set_ylim(yrange)
    plt.savefig('output/cal.g2.n{}.e{}.pdf'.format(n, e), dpi=220)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print 'usage: python calibration_linearity.py n e'
        print
        print 'n : Sersic index'
        print 'e : ellipticity of galaxy |e| used in ring tests'
        sys.exit()
    else:
        calibration_linearity('r', 'CWW_E_ext', 1.25, 'ukg5v',
                              float(sys.argv[1]), float(sys.argv[2]))
