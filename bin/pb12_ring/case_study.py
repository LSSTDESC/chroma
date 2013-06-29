import copy

import lmfit
import scipy.integrate
import numpy
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
import galsim
galsim.GSParams.maximum_fft_size = 16384

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
    return Ixx, Iyy, Ixy

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
    return Ixx, Iyy, Ixy

def diagnostic_ringtest(gamma, n_ring, gparam, star_PSF, gal_PSF):
    s_engine = chroma.ImageEngine.GalSimSEngine(size=31,
                                                gsp=galsim.GSParams(maximum_fft_size=32768))
    galtool = chroma.GalTools.SGalTool(s_engine)

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

    betas = numpy.linspace(0.0, 2.0 * numpy.pi, 2 * n_ring, endpoint=False)
    for beta in betas[1:]:
        # generate target image
        ring_gparam = galtool.get_ring_params(gparam, gamma, beta)
        uncvlim = s_engine.get_uncvl_image(ring_gparam, pixsize=1./7)
        overim = s_engine.get_image(ring_gparam, gal_PSF, pixsize=1./7)
        im = s_engine.get_image(ring_gparam, gal_PSF)

        # fit the target image
        def resid(param):
            testim = s_engine.get_image(param, star_PSF)
            return (testim - im).flatten()
        fit = lmfit.minimize(resid, copy.deepcopy(ring_gparam))
        fitparam = fit.params
        fit_uncvlim = s_engine.get_uncvl_image(fitparam, pixsize=1./7)
        fit_overim = s_engine.get_image(fitparam, star_PSF, pixsize=1./7)
        fit_im = s_engine.get_image(fitparam, star_PSF)

        # compute unweighted quadrupole moments
        Q_g = quadrupole(uncvlim, pixsize=1./7)
        Q_o = quadrupole(overim, pixsize=1./7)
        e_model = ring_gparam['gmag'].value * complex(numpy.cos(2.0 * ring_gparam['phi'].value),
                                                      numpy.sin(2.0 * ring_gparam['phi'].value))
        # unweighted ellipticities
        e_gal = ellip(uncvlim, pixsize=1./7)
        e_obs = ellip(overim, pixsize=1./7)
        e_obsx = ellip(im)

        # repeat for observed (PSF-convolved) galaxy
        Qfit_g = quadrupole(fit_uncvlim, pixsize=1./7)
        Qfit_o = quadrupole(fit_overim, pixsize=1./7)
        efit_model = fitparam['gmag'].value * complex(numpy.cos(2.0 * fitparam['phi'].value),
                                                      numpy.sin(2.0 * fitparam['phi'].value))
        # and corresponding ellipticities
        efit_gal = ellip(fit_uncvlim, pixsize=1./7)
        efit_obs = ellip(fit_overim, pixsize=1./7)
        efit_obsx = ellip(fit_im)

        # now repeat for apodized moments
        Qap_g = apodized_quadrupole(uncvlim, pixsize=1./7)
        Qap_o = apodized_quadrupole(overim, pixsize=1./7)
        Qapfit_g = apodized_quadrupole(fit_uncvlim, pixsize=1./7)
        Qapfit_o = apodized_quadrupole(fit_overim, pixsize=1./7)

        #output!
        print
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
        print '---------------'
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
        print 'Noteworthy comparison'
        print '---------------------'
        print 'delta Ixx_g: {:8f}'.format(Q_o[0] - Qfit_o[0])
        print 'delta Iyy_g: {:8f}'.format(Q_o[1] - Qfit_o[1])
        print 'delta Ixy_g: {:8f}'.format(Q_o[2] - Qfit_o[2])
        print 'delta weighted Ixx_g: {:8f}'.format(Qap_o[0] - Qapfit_o[0])
        print 'delta weighted Iyy_g: {:8f}'.format(Qap_o[1] - Qapfit_o[1])
        print 'delta weighted Ixy_g: {:8f}'.format(Qap_o[2] - Qapfit_o[2])

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
        axarr[2,0].set_ylabel('residual')
        axarr[3,0].set_ylabel('percent residual')
        f.subplots_adjust(hspace=0.02, wspace=0.07, bottom=0.11)
        cbar_ax = f.add_axes([0.122, 0.05, 0.77, 0.04])
        plt.colorbar(img, cax=cbar_ax, orientation='horizontal')
        plt.show()

def case_study():
    s_engine = chroma.ImageEngine.GalSimSEngine(size=51,
                                                gsp=galsim.GSParams(maximum_fft_size=32768))
    galtool = chroma.GalTools.SGalTool(s_engine)
    PSF_model = chroma.PSF_model.GSAtmPSF
    PSF_ellip = 0.0
    PSF_phi = 0.0
    data_dir = '../../data/'
    filter_file = data_dir+'filters/LSST_r.dat'
    gal_SED_file = data_dir+'SEDs/CWW_E_ext.ascii'
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    # swave = swave[::50]
    # sphotons = sphotons[::50]
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = PSF_model(swave, sphotons, zenith=numpy.pi*60/180, moffat_FWHM=2.5)

    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.0)
    gparam.add('y0', value=0.0)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2)
    gparam.add('phi', value=0.0)

    gparam = galtool.set_uncvl_r2(gparam, (0.27/0.2)**2) # 0.27 arcsecond second moment radius

    redshift = 1.3
    gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, redshift)
    # gwave = gwave[::50]
    # gphotons = gphotons[::50]
    gphotons /= scipy.integrate.simps(gphotons, gwave)
    gal_PSF = PSF_model(gwave, gphotons, zenith=numpy.pi*60/180, moffat_FWHM=2.5)

   # m, c = measure_shear_calib(gparam, gal_PSF, star_PSF, s_engine)
   # print m
   # print c

    diagnostic_ringtest(0.0 + 0.0j, 3, gparam, star_PSF, gal_PSF)

if __name__ == '__main__':
    case_study()
