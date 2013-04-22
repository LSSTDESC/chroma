import os
import sys

import numpy as np
from lmfit import Parameters, Minimizer
import matplotlib.pyplot as plt

import _mypath
import chroma

def fiducial_galaxy():
    gparam = Parameters()
    #bulge
    gparam.add('b_x0', value=0.1)
    gparam.add('b_y0', value=0.3)
    gparam.add('b_n', value=4.0, vary=False)
    gparam.add('b_r_e', value=1.1 * 1.1)
    gparam.add('b_flux', value=0.25)
    gparam.add('b_gmag', value=0.2)
    gparam.add('b_phi', value=0.0)
    #disk
    gparam.add('d_x0', expr='b_x0')
    gparam.add('d_y0', expr='b_y0')
    gparam.add('d_n', value=1.0, vary=False)
    gparam.add('d_r_e', value=1.1)
    gparam.add('d_flux', expr='1.0 - b_flux')
    gparam.add('d_gmag', expr='b_gmag')
    gparam.add('d_phi', expr='b_phi')
    #initialize constrained variables
    dummyfit = Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()
    return gparam

def measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                        PSF_ellip, PSF_phi,
                        im_fac, argv=None):
    wave, photons = chroma.utils.get_photons([bulge_SED_file, disk_SED_file],
                                             filter_file, redshift)
    bulge_photons, disk_photons = photons
    use=None
    if argv is None:
        use = 'voigt'
    elif len(argv) == 1:
        use = 'voigt'
    else:
        use = argv[1]
    PSF_kwargs = {'ellipticity':PSF_ellip, 'phi':PSF_phi}
    if use == 'voigt':
        gal = chroma.voigt12.VoigtBDGal(gparam, wave, bulge_photons, disk_photons,
                                        PSF_kwargs=PSF_kwargs, im_fac=im_fac)
        map(im_fac.load_PSF, [gal.bulge_PSF, gal.disk_PSF, gal.composite_PSF, gal.circ_PSF])
    elif use == 'gsfull':
        gal = chroma.GalSimBDGal(gparam, wave, bulge_photons, disk_photons, PSF_kwargs=PSF_kwargs)
    elif use == 'gs':
        gal = chroma.GalSimBDGalInt(gparam, wave, bulge_photons, disk_photons, PSF_kwargs=PSF_kwargs)
    else:
        print 'error'
        sys.exit()
    gal.set_FWHM_ratio(1.4)

    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gal.gen_target_image,
                                       gal.gen_init_param,
                                       gal.measure_ellip)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gal.gen_target_image,
                                       gal.gen_init_param,
                                       gal.measure_ellip)
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1
    return m, c

def fig4_bulge_sersic_index(argv, im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    filter_file = '../../data/filters/voigt12_350.dat'
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9
    PSF_ellip = 0.05
    PSF_phi = 0.0

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_bulge_sersic_index.dat', 'w')
    for bulge_n in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        gparam = fiducial_galaxy()
        gparam['b_n'].value = bulge_n
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac, argv)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(bulge_n, c, m))
    fil.close()

def fig4_bulge_flux(argv, im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    filter_file = '../../data/filters/voigt12_350.dat'
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9
    PSF_ellip = 0.05
    PSF_phi = 0.0

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_bulge_flux.dat', 'w')
    for bulge_flux in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        gparam = fiducial_galaxy()
        gparam['b_flux'].value = bulge_flux
        gparam['d_flux'].value = 1.0 - bulge_flux
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac, argv)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(bulge_flux, c, m))
    fil.close()

def fig4_gal_ellip(argv, im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    PSF_ellip = 0.05
    PSF_phi = 0.0
    filter_file = '../../data/filters/voigt12_350.dat'
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_gal_ellip.dat', 'w')
    for gal_ellip in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        gparam = fiducial_galaxy()
        gparam['b_gmag'].value = gal_ellip
        gparam['d_gmag'].value = gal_ellip
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac, argv)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(gal_ellip, c, m))
    fil.close()

def fig4_y0(argv, im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    filter_file = '../../data/filters/voigt12_350.dat'
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9
    PSF_ellip = 0.05
    PSF_phi = 0.0

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_y0.dat', 'w')
    for y0 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        gparam = fiducial_galaxy()
        gparam['b_y0'].value = y0
        gparam['d_y0'].value = y0
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac, argv)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(y0, c, m))
    fil.close()

def fig4data(argv):
    im_fac = chroma.voigt12.ImageFactory()
    fig4_bulge_sersic_index(argv, im_fac)
    fig4_bulge_flux(argv, im_fac)
    fig4_gal_ellip(argv, im_fac)
    fig4_y0(argv, im_fac)

def fig4plot(argv):
    #setup plots
    fig = plt.figure(figsize=(10.0, 7.5), dpi=60)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
    ax1 = fig.add_subplot(221)
    ax1.set_yscale('log')
    ax1.set_ylabel('|m|')
    ax1.set_ylim(5.e-5, 1.e-2)
    ax1.set_xlabel('n$_{\mathrm{s, b}}$')
    ax1.set_xlim(1.5, 4.0)

    ax2 = fig.add_subplot(222)
    ax2.set_yscale('log')
    ax2.set_ylabel('|m|')
    ax2.set_ylim(5.e-5, 1.e-2)
    ax2.set_xlabel('B/T')
    ax2.set_xlim(0.0, 1.0)

    ax3 = fig.add_subplot(223)
    ax3.set_yscale('log')
    ax3.set_ylabel('|m|')
    ax3.set_ylim(5.e-5, 1.e-2)
    ax3.set_xlabel('e$_{\mathrm{g}}$')
    ax3.set_xlim(0.1, 0.6)

    ax4 = fig.add_subplot(224)
    ax4.set_yscale('log')
    ax4.set_ylabel('|m|')
    ax4.set_ylim(5.e-5, 1.e-2)
    ax4.set_xlabel('y$_0$')
    ax4.set_xlim(0.0, 0.5)

    ax1.fill_between([1.5, 4.0], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([1.5, 4.0], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([1.5, 4.0], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax2.fill_between([0.0, 1.0], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([0.0, 1.0], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([0.0, 1.0], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax3.fill_between([0.1, 0.6], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax3.fill_between([0.1, 0.6], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax3.fill_between([0.1, 0.6], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax4.fill_between([0.0, 0.5], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax4.fill_between([0.0, 0.5], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax4.fill_between([0.0, 0.5], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    # load bulge sersic index data

    calib = {'bulge_n':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_bulge_sersic_index.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                bulge_n, c1, c2, m1, m2 = line.split(' ')
                calib['bulge_n'].append(float(bulge_n))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['bulge_n'], abs(np.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['bulge_n'], abs(np.array(calib['m1'])), color='red')
    ax1.plot(calib['bulge_n'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['bulge_n'], abs(np.array(calib['m2'])), color='red', ls='--')

    # load bulge flux data

    calib = {'bulge_flux':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_bulge_flux.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                bulge_flux, c1, c2, m1, m2 = line.split(' ')
                calib['bulge_flux'].append(float(bulge_flux))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax2.plot(calib['bulge_flux'], abs(np.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['bulge_flux'], abs(np.array(calib['m1'])), color='red')
    ax2.plot(calib['bulge_flux'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['bulge_flux'], abs(np.array(calib['m2'])), color='red', ls='--')


    # load galaxy ellipticity data

    calib = {'gal_ellip':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_gal_ellip.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                gal_ellip, c1, c2, m1, m2 = line.split(' ')
                calib['gal_ellip'].append(float(gal_ellip))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax3.plot(calib['gal_ellip'], abs(np.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax3.plot(calib['gal_ellip'], abs(np.array(calib['m1'])), color='red')
    ax3.plot(calib['gal_ellip'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax3.plot(calib['gal_ellip'], abs(np.array(calib['m2'])), color='red', ls='--')

    # load y0 data

    calib = {'y0':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_y0.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                y0, c1, c2, m1, m2 = line.split(' ')
                calib['y0'].append(float(y0))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax4.plot(calib['y0'], abs(np.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax4.plot(calib['y0'], abs(np.array(calib['m1'])), color='red')
    ax4.plot(calib['y0'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax4.plot(calib['y0'], abs(np.array(calib['m2'])), color='red', ls='--')

    plt.savefig('output/fig4.pdf')

if __name__ == '__main__':
    fig4data(sys.argv)
    fig4plot(sys.argv)
