import os

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
                        im_fac):
    wave, photons = chroma.utils.get_photons([bulge_SED_file, disk_SED_file],
                                             filter_file, redshift)
    bulge_photons, disk_photons = photons
    gal = chroma.voigt12.bdgal(gparam, wave, bulge_photons, disk_photons,
                               PSF_ellip, PSF_phi, im_fac)
    gal.set_FWHM_ratio(1.4)
    gen_target_image = gal.target_image_fn_generator()
    gen_init_param = gal.init_param_generator()
    measure_ellip = gal.ellip_measurement_generator()

    gamma0 = 0.0 + 0.0j
    gamma0_hat = chroma.utils.ringtest(gamma0, 3,
                                       gen_target_image,
                                       gen_init_param,
                                       measure_ellip)
    c = gamma0_hat.real, gamma0_hat.imag

    gamma1 = 0.01 + 0.02j
    gamma1_hat = chroma.utils.ringtest(gamma1, 3,
                                       gen_target_image,
                                       gen_init_param,
                                       measure_ellip)
    m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
    m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
    m = m0, m1
    return m, c

def fig3_fiducial(im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    PSF_ellip = 0.05
    PSF_phi = 0.0

    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Running on fiducial galaxy parameters'
    print

    if not os.path.isdir('./output/'):
        os.mkdir('output/')
    fil = open('output/fig3_fiducial.dat', 'w')
#    for fw in [350]:
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3_redshift(im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    filter_widths = [150, 250, 350, 450]
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 1.4

    print
    print 'Varying the redshift'
    print

    if not os.path.isdir('./output/'):
        os.mkdir('output/')
    fil = open('output/fig3_redshift.dat', 'w')
    for fw in filter_widths:
        gparam = fiducial_galaxy()
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3_bulge_radius(im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    filter_widths = [150, 250, 350, 450]
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the bulge radius'
    print

    if not os.path.isdir('./output/'):
        os.mkdir('output/')
    fil = open('output/fig3_bulge_radius.dat', 'w')
    for fw in filter_widths:
        gparam = fiducial_galaxy()
        gparam['b_r_e'].value = gparam['b_r_e'].value * 0.4/1.1
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3_disk_spectrum(im_fac=None):
    if im_fac is None:
        im_fac = chroma.voigt12.ImageFactory()
    filter_widths = [150, 250, 350, 450]
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Im_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the disk spectrum'
    print

    if not os.path.isdir('./output/'):
        os.mkdir('output/')
    fil = open('output/fig3_disk_spectrum.dat', 'w')
    for fw in filter_widths:
        gparam = fiducial_galaxy()
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, im_fac)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3data():
    im_fac = chroma.voigt12.ImageFactory()
    fig3_fiducial(im_fac)
    fig3_redshift(im_fac)
    fig3_bulge_radius(im_fac)
    fig3_disk_spectrum(im_fac)

def fig3plot():
    #setup plots
    fig = plt.figure(figsize=(5.5,7), dpi=100)
    fig.subplots_adjust(left=0.18)
    ax1 = fig.add_subplot(211)
    ax1.set_yscale('log')
    ax1.set_ylabel('|m|')
    ax1.set_ylim(5.e-5, 2.e-2)
    ax1.set_xlim(150, 450)
    ax1.fill_between([150,450], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([150,450], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([150,450], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax2 = fig.add_subplot(212)
    ax2.set_yscale('log')
    ax2.set_xlabel('Filter width (nm)')
    ax2.set_ylabel('|c|')
    ax2.set_ylim(1.5e-5, 1.e-3)
    ax2.set_xlim(150, 450)
    ax2.fill_between([150,450], [3.e-4, 3.e-4], [1.5e-5, 1.5e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([150,450], [3.e-4/2, 3.e-4/2], [1.5e-5, 1.5e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([150,450], [3.e-4/5, 3.e-4/5], [1.5e-5, 1.5e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    #load fiducial galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_fiducial.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass
    ax1.plot(calib['width'], abs(np.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m1'])), color='red')
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), color='red')
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), 's', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), color='red')
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), color='red')


    #load varied redshift galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_redshift.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['width'], abs(np.array(calib['m1'])), 's', mfc='None', mec='blue', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m1'])), color='blue', linestyle='--')
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='blue', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), color='blue', linestyle='--')
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), 's', mfc='None', mec='blue', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), color='blue', linestyle='--')
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), 'x', mfc='None', mec='blue', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), color='blue', linestyle='--')


    #load varied bulge radius galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_bulge_radius.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['width'], abs(np.array(calib['m1'])), 's', mfc='None', mec='green', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m1'])), color='green', linestyle='-.')
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='green', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), color='green', linestyle='-.')
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), 's', mfc='None', mec='green', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), color='green', linestyle='-.')
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), 'x', mfc='None', mec='green', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), color='green', linestyle='-.')


    #load varied disk spectrum galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_disk_spectrum.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['width'], abs(np.array(calib['m1'])), 's', mfc='None', mec='black', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m1'])), color='black', linestyle=':')
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), 'x', mfc='None', mec='black', mew=1.3)
    ax1.plot(calib['width'], abs(np.array(calib['m2'])), color='black', linestyle=':')
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), 's', mfc='None', mec='black', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c1'])), color='black', linestyle=':')
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), 'x', mfc='None', mec='black', mew=1.3)
    ax2.plot(calib['width'], abs(np.array(calib['c2'])), color='black', linestyle=':')


    plt.savefig('output/fig3.pdf')

if __name__ == '__main__':
    fig3data()
    fig3plot()
