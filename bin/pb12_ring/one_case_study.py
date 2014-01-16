import copy

import lmfit
import scipy.integrate
import numpy as np
import galsim
import matplotlib.pyplot as plt
import astropy.io.fits as fits

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

def one_case_study(Sersic_n, zenith, bd, gamma, beta,
                   gal_SED_name, star_SED_name, filter_name, redshift, Rbar=0.0):
    s_engine = chroma.ImageEngine.GalSimSEngine(size=31,
                                                gsp=galsim.GSParams(maximum_fft_size=32768))
    galtool = chroma.GalTools.SGalTool(s_engine)
    PSF_model = chroma.PSF_model.GSAtmPSF
    PSF_ellip = 0.0
    PSF_phi = 0.0
    data_dir = '../../data/'
    filter_file = data_dir+'filters/'+filter_name+'.dat'
    gal_SED_file = data_dir+'SEDs/'+gal_SED_name+'.ascii'
    star_SED_file = data_dir+'SEDs/'+star_SED_name+'.ascii'

    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    # thin out stellar spectrum
    swave = swave[::50]
    sphotons = sphotons[::50]
    sphotons /= scipy.integrate.simps(sphotons, swave)
    star_PSF = PSF_model(swave, sphotons, zenith=zenith)
    star_PSF_overim = s_engine.get_PSF_image(star_PSF, pixsize=1./7)
    smom = chroma.disp_moments(swave, sphotons, zenith=zenith)


    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.0)
    gparam.add('y0', value=0.0)
    gparam.add('n', value=Sersic_n, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)

    gparam = galtool.set_uncvl_r2(gparam, (0.27/0.2)**2) # 0.27 arcsecond second moment radius

    gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, redshift)
    # thin out galactic spectrum
    gwave = gwave[::50]
    gphotons = gphotons[::50]
    gphotons /= scipy.integrate.simps(gphotons, gwave)
    gal_PSF = PSF_model(gwave, gphotons, zenith=zenith)
    gmom = chroma.disp_moments(gwave, gphotons, zenith=zenith)

    bd_engine = chroma.ImageEngine.GalSimBDEngine(size=31,
                                                  gsp=galsim.GSParams(maximum_fft_size=32768))
    rotgparam = galtool.get_ring_params(gparam, gamma, beta)
    # generate target image
    uncvlim = s_engine.get_uncvl_image(rotgparam, pixsize=1./7)
    gal_PSF_overim = s_engine.get_PSF_image(gal_PSF, pixsize=1./7)
    overim = s_engine.get_image(rotgparam, gal_PSF, pixsize=1./7)
    im = s_engine.get_image(rotgparam, gal_PSF)

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
        outparam.add('b_gmag', value=param['gmag'].value)
        outparam.add('b_phi', value=param['phi'].value)
        outparam.add('d_x0', expr='b_x0')
        outparam.add('d_y0', expr='b_y0')
        outparam.add('d_n', value=1.0, vary=False)
        outparam.add('d_r_e', value=param['r_e'].value)
        outparam.add('d_gmag', expr='b_gmag')
        outparam.add('d_phi', expr='b_phi')
        if bd:
            outparam.add('b_flux', value=1.0)
            outparam.add('d_flux', expr='1.0 - b_flux')
        else:
            outparam.add('b_flux', value=1.0, vary=False)
            outparam.add('d_flux', value=0.0, vary=False)
        return outparam
    fit = lmfit.minimize(resid, initparam(rotgparam))
    fitparam = fit.params
    print 'target flux: {}'.format(1.0)
    print 'fitted flux: {}'.format(fitparam['b_flux'].value + fitparam['d_flux'].value)
    # generate fit images
    fit_uncvlim = bd_engine.get_uncvl_image(fitparam, pixsize=1./7)
    fit_overim = bd_engine.get_image(fitparam, star_PSF, star_PSF, pixsize=1./7)
    fit_im = bd_engine.get_image(fitparam, star_PSF, star_PSF)
    #nudge fit image to match centroid of target image
    delta_rbar = (gmom[0] - smom[0]) * 180./np.pi * 3600 / 0.2
    fitparam_nudge = copy.deepcopy(fitparam)
    fitparam_nudge['b_y0'].value = fitparam['b_y0'].value + delta_rbar
    fitparam_nudge['d_y0'].value = fitparam['d_y0'].value + delta_rbar
    nudge_uncvlim = bd_engine.get_uncvl_image(fitparam, pixsize=1./7)

    uncvlim_HDU = fits.PrimaryHDU(uncvlim)
    gal_PSF_overim_HDU = fits.ImageHDU(gal_PSF_overim)
    overim_HDU = fits.ImageHDU(overim)
    im_HDU = fits.ImageHDU(im)
    fit_uncvlim_HDU = fits.ImageHDU(fit_uncvlim)
    star_PSF_overim_HDU = fits.ImageHDU(star_PSF_overim)
    fit_overim_HDU = fits.ImageHDU(fit_overim)
    fit_im_HDU = fits.ImageHDU(fit_im)
    nudge_uncvlim_HDU = fits.ImageHDU(nudge_uncvlim)

    hdulist = fits.HDUList([uncvlim_HDU, gal_PSF_overim_HDU, overim_HDU, im_HDU,
                            fit_uncvlim_HDU, star_PSF_overim_HDU, fit_overim_HDU, fit_im_HDU,
                            nudge_uncvlim_HDU])
    hdulist.writeto('output/one_case.fits', clobber=True)

if __name__ == '__main__':
    one_case_study(Sersic_n=1.0, zenith=60.0*np.pi/180, bd=True,
                   gamma=0.01+0.02j, beta=(30.0*np.pi/180),
                   gal_SED_name='CWW_E_ext', star_SED_name='ukg5v', filter_name='LSST_r',
                   redshift=1.3)
