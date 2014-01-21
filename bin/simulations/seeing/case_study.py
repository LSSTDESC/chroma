import copy

import lmfit
import scipy.integrate
import numpy
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
from matplotlib.backends.backend_pdf import PdfPages
import galsim

import _mypath
import chroma

def my_imshow(my_img,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%5e @ [%4i, %4i]' % (my_img[y, x], x, y)
        except IndexError:
            return ''
    img = ax.imshow(my_img,**kwargs)
    ax.format_coord = format_coord
    return img

def quadrupole(data, pixsize=1.0):
    xs, ys = numpy.mgrid[0:data.shape[0], 0:data.shape[1]] * pixsize
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs - xbar)**2).sum() / total
    Iyy = (data * (ys - ybar)**2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return numpy.array([Ixx, Iyy, Ixy])

def ellip(data, pixsize=1.0):
    Ixx, Iyy, Ixy = quadrupole(data, pixsize=pixsize)
    denom = Ixx + Iyy + 2.0 * numpy.sqrt(Ixx * Iyy - Ixy**2)
    return complex(Ixx - Iyy, 2.0 * Ixy) / denom

def apodized_quadrupole(data, sigma=5.0, pixsize=1.0):
    xs, ys = numpy.mgrid[0:data.shape[0], 0:data.shape[1]] * pixsize
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    weight = numpy.exp(-0.5 * ((xs - xbar)**2 + (ys - ybar)**2) / sigma**2)
    data1 = data * weight
    total1 = data1.sum()
    Ixx = (data1 * (xs - xbar)**2).sum() / total1
    Iyy = (data1 * (ys - ybar)**2).sum() / total1
    Ixy = (data1 * (xs - xbar) * (ys - ybar)).sum() / total1
    return numpy.array([Ixx, Iyy, Ixy])

def diagnostic_ringtest(gamma, n_ring, gparam, star_PSF, gal_PSF, bd=False):
    s_engine = chroma.ImageEngine.GalSimSEngine(size=31,
                                                gsp=galsim.GSParams(maximum_fft_size=32768))
    bd_engine = chroma.ImageEngine.GalSimBDEngine(size=31,
                                                  gsp=galsim.GSParams(maximum_fft_size=32768))
    galtool = chroma.GalTools.SGalTool(s_engine)
    bdtool = chroma.GalTools.BDGalTool(bd_engine)

    star_PSF_overim = s_engine.get_PSF_image(star_PSF, pixsize=1./7)
    star_mom = chroma.utils.moments(star_PSF_overim, pixsize=1./7)

    print
    print 'stellar PSF quadrupole moments'
    print '-----------------------'
    print 'Ixx = {:8f}'.format(star_mom[2])
    print 'Iyy = {:8f}'.format(star_mom[3])
    print 'Ixy = {:8f}'.format(star_mom[4])

    gal_PSF_overim = s_engine.get_PSF_image(gal_PSF, pixsize=1./7)
    gal_mom = chroma.utils.moments(gal_PSF_overim, pixsize=1./7)

    print
    print 'galactic PSF quadrupole moments'
    print '-----------------------'
    print 'Ixx = {:8f}'.format(gal_mom[2])
    print 'Iyy = {:8f}'.format(gal_mom[3])
    print 'Ixy = {:8f}'.format(gal_mom[4])

    outfile = 'output/case_study_g{:6.2f}.n{}.bd{}.pdf'
    outfile = outfile.format(gamma, gparam['n'].value, bd)
    pp = PdfPages(outfile)

    betas = numpy.linspace(0.0, 2.0 * numpy.pi, 2 * n_ring, endpoint=False)
    out = numpy.empty(len(betas), dtype=[('ellip', 'c8'),
                                         ('chisqr','f8'),
                                         ('FoD','f8')])
    iout = 0
    for beta in betas:
        # generate target image
        ring_gparam = galtool.get_ring_params(gparam, gamma, beta)
        uncvlim = s_engine.get_uncvl_image(ring_gparam, pixsize=1./7)
        overim = s_engine.get_image(ring_gparam, gal_PSF, pixsize=1./7)
        im = s_engine.get_image(ring_gparam, gal_PSF)

        # fit the target image
        def resid(param):
            testim = bd_engine.get_image(param, star_PSF, star_PSF)
            return (testim - im).flatten()
        def initparam(param):
            outparam = lmfit.Parameters()
            outparam.add('b_x0', value=param['x0'].value)
            outparam.add('b_y0', value=param['y0'].value)
            outparam.add('b_n', value=param['n'].value, vary=False)
            outparam.add('b_r_e', value=param['r_e'].value)
            outparam.add('b_gmag', value=param['gmag'].value, min=0.0, max=1.0)
            outparam.add('b_phi', value=param['phi'].value)
            outparam.add('d_x0', expr='b_x0')
            outparam.add('d_y0', expr='b_y0')
            outparam.add('d_n', value=1.0, vary=False)
            outparam.add('d_r_e', value=param['r_e'].value)
            outparam.add('d_gmag', expr='b_gmag', min=0.0, max=1.0)
            outparam.add('d_phi', expr='b_phi')
            if bd:
                outparam.add('b_flux', value=1.0)
                outparam.add('d_flux', expr='1.0 - b_flux')
            else:
                outparam.add('b_flux', value=1.0, vary=False)
                outparam.add('d_flux', value=0.0, vary=False)
            return outparam
        fit = lmfit.minimize(resid, initparam(ring_gparam))
        fitparam = fit.params
        fit_uncvlim = bd_engine.get_uncvl_image(fitparam, pixsize=1./7)
        fit_overim = bd_engine.get_image(fitparam, star_PSF, star_PSF, pixsize=1./7)
        fit_im = bd_engine.get_image(fitparam, star_PSF, star_PSF)

        # compute unweighted quadrupole moments of 'truth' image
        Q_g = quadrupole(uncvlim, pixsize=1./7)
        Q_o = quadrupole(overim, pixsize=1./7)
        Q_p = quadrupole(im)
        e_model = ring_gparam['gmag'].value * complex(numpy.cos(2.0 * ring_gparam['phi'].value),
                                                      numpy.sin(2.0 * ring_gparam['phi'].value))
        # unweighted ellipticities
        e_gal = ellip(uncvlim, pixsize=1./7)
        e_obs = ellip(overim, pixsize=1./7)
        e_obsx = ellip(im)

        # repeat for best fit galaxy
        Qfit_g = quadrupole(fit_uncvlim, pixsize=1./7)
        Qfit_o = quadrupole(fit_overim, pixsize=1./7)
        Qfit_p = quadrupole(fit_im)
        efit_model = fitparam['b_gmag'].value * complex(numpy.cos(2.0 * fitparam['b_phi'].value),
                                                        numpy.sin(2.0 * fitparam['b_phi'].value))
        # and corresponding ellipticities
        efit_gal = ellip(fit_uncvlim, pixsize=1./7)
        efit_obs = ellip(fit_overim, pixsize=1./7)
        efit_obsx = ellip(fit_im)

        # now repeat for apodized moments
        Qap_g = apodized_quadrupole(uncvlim, pixsize=1./7)
        Qap_o = apodized_quadrupole(overim, pixsize=1./7)
        Qap_p = apodized_quadrupole(im)
        Qapfit_g = apodized_quadrupole(fit_uncvlim, pixsize=1./7)
        Qapfit_o = apodized_quadrupole(fit_overim, pixsize=1./7)
        Qapfit_p = apodized_quadrupole(fit_im)

        # and record the quadrupole moment residuals
        Qresid_o = Q_o - Qfit_o
        Qresid_p = Q_p - Qfit_p
        Qapresid_o = Qap_o - Qapfit_o
        Qapresid_p = Qap_p - Qapfit_p

        # fit FoD is sum of pixelized xx and yy quadrupole moments
        FoD = Qapresid_p[0] + Qapresid_p[1]
        out[iout] = (efit_model, FoD, fit.chisqr)
        iout +=1

        # output!
        print
        print
        print 'Ring params'
        print '-----------'
        print 'beta  : {:5.2f}'.format(beta)
        print 'gamma : {:5.2f}'.format(gamma)
        print
        print 'Truth image'
        print '-----------'
        print 'Ixx_g  : {:8f}'.format(Q_g[0])
        print 'Iyy_g  : {:8f}'.format(Q_g[1])
        print 'Ixy_g  : {:8f}'.format(Q_g[2])
        print 'Ixx_o  : {:8f}'.format(Q_o[0])
        print 'Iyy_o  : {:8f}'.format(Q_o[1])
        print 'Ixy_o  : {:8f}'.format(Q_o[2])
        print 'e_model: |{:6f}| = {:6f}'.format(e_model, abs(e_model))
        print 'e_gal:   |{:6f}| = {:6f}'.format(e_gal, abs(e_gal))
        print 'e_obs:   |{:6f}| = {:6f}'.format(e_obs, abs(e_obs))
        print 'e_obsx:  |{:6f}| = {:6f}'.format(e_obsx, abs(e_obsx))
        print
        print 'Best fit image'
        print '--------------'
        print 'Ixx_g  : {:8f}'.format(Qfit_g[0])
        print 'Iyy_g  : {:8f}'.format(Qfit_g[1])
        print 'Ixy_g  : {:8f}'.format(Qfit_g[2])
        print 'Ixx_o  : {:8f}'.format(Qfit_o[0])
        print 'Iyy_o  : {:8f}'.format(Qfit_o[1])
        print 'Ixy_o  : {:8f}'.format(Qfit_o[2])
        print 'e_model: |{:6f}| = {:6f}'.format(efit_model, abs(efit_model))
        print 'e_gal:   |{:6f}| = {:6f}'.format(efit_gal, abs(efit_gal))
        print 'e_obs:   |{:6f}| = {:6f}'.format(efit_obs, abs(efit_obs))
        print 'e_obsx:  |{:6f}| = {:6f}'.format(efit_obsx, abs(efit_obsx))
        print
        print 'Truth to best fit moment comparisons'
        print '------------------------------------'
        print 'delta Ixx_o: {:8f}'.format(Qresid_o[0])
        print 'delta Iyy_o: {:8f}'.format(Qresid_o[1])
        print 'delta Ixy_o: {:8f}'.format(Qresid_o[2])
        print 'delta Ixx_p: {:8f}'.format(Qresid_p[0])
        print 'delta Iyy_p: {:8f}'.format(Qresid_p[1])
        print 'delta Ixy_p: {:8f}'.format(Qresid_p[2])
        print 'delta weighted Ixx_o: {:8f}'.format(Qapresid_o[0])
        print 'delta weighted Iyy_o: {:8f}'.format(Qapresid_o[1])
        print 'delta weighted Ixy_o: {:8f}'.format(Qapresid_o[2])
        print 'delta weighted Ixx_p: {:8f}'.format(Qapresid_p[0])
        print 'delta weighted Iyy_p: {:8f}'.format(Qapresid_p[1])
        print 'delta weighted Ixy_p: {:8f}'.format(Qapresid_p[2])

        #display
        f, axarr = plt.subplots(4, 4, figsize=(6,6))
        im /= 49
        fit_im /=49
        ims = [uncvlim, gal_PSF_overim, overim, im,
               fit_uncvlim, star_PSF_overim, fit_overim, fit_im,
               fit_uncvlim - uncvlim, gal_PSF_overim - star_PSF_overim,
               fit_overim - overim, fit_im - im,
               (fit_uncvlim - uncvlim)/uncvlim,
               (star_PSF_overim - gal_PSF_overim)/gal_PSF_overim,
               (fit_overim - overim)/overim,
               (fit_im - im)/im]
        k=0
        norm = 0.00954924
        for i in range(4):
            for j in range(4):
                if j == 1: #PSF
                    img = my_imshow(numpy.log10(abs(ims[k]/ims[k].max())), ax=axarr[i, j],
                                    extent=[0, im.shape[0], 0, im.shape[1]],
                                    vmin=-5, vmax=0)
                elif i==3:
                    img = my_imshow(numpy.log10(abs(ims[k])), ax=axarr[i, j],
                                    extent=[0, im.shape[0], 0, im.shape[1]],
                                    vmin=-5, vmax=0)
                else:
                    img = my_imshow(numpy.log10(abs(ims[k]/norm)), ax=axarr[i, j],
                                    extent=[0, im.shape[0], 0, im.shape[1]],
                                    vmin=-5, vmax=0)
                axarr[i, j].get_xaxis().set_ticks([])
                axarr[i, j].get_yaxis().set_ticks([])
                k += 1
        axarr[0,0].set_title('galaxy')
        axarr[0,1].set_title('PSF')
        axarr[0,2].set_title('convolution')
        axarr[0,3].set_title('pixelized')
        axarr[0,0].set_ylabel('truth')
        axarr[1,0].set_ylabel('best fit')
        axarr[2,0].set_ylabel('abs(residual')
        axarr[3,0].set_ylabel('abs(rel resid)')
        f.subplots_adjust(hspace=0.02, wspace=0.07, bottom=0.11)
        cbar_ax = f.add_axes([0.122, 0.05, 0.77, 0.04])
        plt.colorbar(img, cax=cbar_ax, orientation='horizontal')
        # plt.show()
        pp.savefig()
    pp.close()
    return out

def case_study(n, bd=False):
    plate_scale = 0.2
    seeing500 = (0.7 / plate_scale) * (625./500)**(-0.2)

    s_engine = chroma.ImageEngine.GalSimSEngine(size=31,
                                                gsp=galsim.GSParams(maximum_fft_size=32768))
    galtool = chroma.GalTools.SGalTool(s_engine)
    PSF_model = chroma.PSF_model.GSSeeingPSF
    PSF_ellip = 0.0
    PSF_phi = 0.0
    data_dir = '../../data/'
    filter_file = data_dir+'filters/LSST_r.dat'
    gal_SED_file = data_dir+'SEDs/CWW_E_ext.ascii'
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    # thin out stellar spectrum
    swave = swave[::50]
    sphotons = sphotons[::50]
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = PSF_model(swave, sphotons, moffat_FWHM_500 = seeing500)
    ssize = chroma.relative_second_moment_radius(swave, sphotons)

    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.0)
    gparam.add('y0', value=0.0)
    gparam.add('n', value=n, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)

    gparam = galtool.set_uncvl_r2(gparam, (0.27/0.2)**2) # 0.27 arcsecond second moment radius

    redshift = 1.3
    # thin out galactic spectrum
    gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, redshift)
    gwave = gwave[::50]
    gphotons = gphotons[::50]
    gphotons /= scipy.integrate.simps(gphotons, gwave)
    gal_PSF = PSF_model(gwave, gphotons, moffat_FWHM_500 = seeing500)
    gsize = chroma.relative_second_moment_radius(gwave, gphotons)

    print
    print
    print
    print '###################'
    print 'Starting case study'
    print '###################'
    print
    print 'Running ring test for n={:3.1f}'.format(gparam['n'].value)
    if bd:
        print 'Using bulge+disk model to reconstruct'
    else:
        print 'Using Sersic model to reconstruct'

    gamma0 = 0+0j
    diag = diagnostic_ringtest(gamma0, 3, gparam, star_PSF, gal_PSF, bd=bd)
    print
    print 'input shear: {:8.5f}'.format(gamma0)
    print 'measured ellipticities:'
    for e in diag['ellip']:
        print '{:8.5f}'.format(e)
    gamma0_hat = sum(diag['ellip'])/len(diag)
    print 'average measured ellipticity: {:8.5f}'.format(gamma0_hat)
    c_calib = gamma0_hat

    gamma1 = 0.01 + 0.02j
    diag2 = diagnostic_ringtest(gamma1, 3, gparam, star_PSF, gal_PSF, bd=bd)
    print
    print 'input shear: {:8.5f}'.format(gamma1)
    print 'measured ellipticities:'
    for e in diag2['ellip']:
        print '{:8.5f}'.format(e)
    gamma1_hat = sum(diag2['ellip'])/len(diag2)
    print 'average measured ellipticity: {:8.5f}'.format(gamma1_hat)
    m0 = (gamma1_hat.real - c_calib.real)/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c_calib.imag)/gamma1.imag - 1.0
    m_calib = complex(m0, m1)

    delta_logR2 = numpy.log(gsize/ssize)
    m_analytic = delta_logR2 * (0.66 / 0.27)**2 * complex(1.0, 1.0)
    c_analytic = complex(0,0)

    print
    print
    print '+---------+'
    print '| results |'
    print '+---------+'
    print
    print 'n: {:3.1f}'.format(gparam['n'].value)
    print 'bd: {}'.format(bd)
    print
    print 'delta log(R_*^2): {:9.6f}'.format(delta_logR2)
    print 'analytic c: {:9.6f}'.format(c_analytic)
    print 'analytic m: {:9.6f}'.format(m_analytic)
    print
    print 'ring c: {:9.6f}'.format(c_calib)
    print 'ring m: {:9.6f}'.format(m_calib)
    print 'average chisqr: {:8.5e}'.format((sum(diag['chisqr']) \
                                            + sum(diag2['chisqr'])) / (2 * len(diag)))
    print 'average FoD: {:8.5e}'.format((sum(diag['FoD']) \
                                         + sum(diag2['FoD'])) / (2 * len(diag)))



if __name__ == '__main__':
    case_study(0.5, bd=False)
    # case_study(0.5, bd=True)
    case_study(1.0, bd=False)
    # case_study(1.0, bd=True)
    case_study(4.0, bd=False)
    # case_study(4.0, bd=True)
