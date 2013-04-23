# fig3.py
#
# Reproduce Figure 3 from Voigt et al. (2012).
#
# The Voigt paper investigates the effects of the wavelength-dependent Euclid PSF on estimating
# shapes of galaxies with color gradients.  In figure 3, the shear calibration parameters `m` and
# `c`, defined such that the estimated shear `gamma_hat` is related to the true shear `gamma` as
#    gamma_hat = (1 + m) * gamma + c
# are plotted for various bulge+disk model galaxies as a function of the width of the Euclid imaging
# filter.  This python script attempts to reproduce this figure using three different algorithms:
#
# 1) The algorithm described in Voigt+12.
# 2) Using the package `galsim` to produce postage stamp images of galaxies.
# 3) Using `galsim` to generate exact PSFs, but then caching these PSFs as oversampled images to speed
#    up the computation.
# By comparing a handful of both intermediate and final results, it appears that the differences
# between (2) and (3) are at least one order of magnitude smaller than the differences between (1)
# and either (2) or (3), indicating that the approximation introduced in (3) is not severely
# affecting the accuracy of the calculations.  On the other hand, (3) is several orders of magnitude
# faster to execute than (2).

import os
import sys

import numpy
import lmfit
import matplotlib.pyplot as plt

import _mypath
import chroma

def fiducial_galaxy():
    '''Bulge + disk parameters of the fiducial galaxy described in Voigt+12.'''
    gparam = lmfit.Parameters()
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
    dummyfit = lmfit.Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()
    return gparam

def measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                        PSF_ellip, PSF_phi,
                        PSF_model, bd_engine):
    '''Perform two ring tests to solve for shear calibration parameters `m` and `c`.'''
    wave, photons = chroma.utils.get_photons([bulge_SED_file, disk_SED_file],
                                             filter_file, redshift)
    bulge_photons, disk_photons = photons
    use=None
    PSF_kwargs = {'ellipticity':PSF_ellip, 'phi':PSF_phi}

    gal = chroma.BDGal(gparam, wave, bulge_photons, disk_photons,
                       PSF_model=PSF_model, PSF_kwargs=PSF_kwargs,
                       bd_engine=bd_engine)

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

def fig3_fiducial(bd_engine, PSF_model):
    '''Generate `m` and `c` vs. filter width for the fiducial bulge+disk galaxy.'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Running on fiducial galaxy parameters'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_fiducial.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3_redshift(bd_engine, PSF_model):
    '''Generate `m` and `c` for the fiducial galaxy, but change the redshift to 1.4'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 1.4

    print
    print 'Varying the redshift'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_redshift.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3_bulge_radius(bd_engine, PSF_model):
    '''Generate `m` and `c` for the fiducial galaxy, but adjust the bulge radius such that the ratio
    b_r_e/d_r_e = 0.4'''
    filter_widths = [150, 250, 350, 450]
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the bulge radius'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_bulge_radius.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        gparam['b_r_e'].value = gparam['b_r_e'].value * 0.4/1.1
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

def fig3_disk_spectrum(bd_engine, PSF_model):
    filter_widths = [150, 250, 350, 450]
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = '../../data/SEDs/CWW_E_ext.ascii'
    disk_SED_file = '../../data/SEDs/CWW_Im_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the disk spectrum'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_disk_spectrum.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = '../../data/filters/voigt12_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()

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
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='red')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='red')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='red')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='red')


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

    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='blue', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='blue', linestyle='--')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='blue', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='blue', linestyle='--')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='blue', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='blue', linestyle='--')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='blue', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='blue', linestyle='--')


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

    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='green', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='green', linestyle='-.')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='green', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='green', linestyle='-.')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='green', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='green', linestyle='-.')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='green', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='green', linestyle='-.')


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

    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='black', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='black', linestyle=':')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='black', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='black', linestyle=':')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='black', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='black', linestyle=':')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='black', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='black', linestyle=':')


    plt.savefig('output/fig3.pdf')

def fig3data(argv):
    if len(argv) == 1:
        use='gs'
    else:
        use=argv[1]
    if use == 'voigt':
        bd_engine = chroma.imgen.VoigtBDEngine()
        PSF_model = chroma.PSF_model.VoigtEuclidPSF
    elif use == 'gs':
        bd_engine = chroma.imgen.GalSimBDEngine()
        PSF_model = chroma.PSF_model.GSEuclidPSFInt
    elif use == 'gsfull':
        bd_engine = chroma.imgen.GalSimBDEngine()
        PSF_model = chroma.PSF_model.GSEuclidPSF
    else:
        print 'unknown or missing command line option'
        sys.exit()

    fig3_fiducial(bd_engine, PSF_model)
    fig3_redshift(bd_engine, PSF_model)
    fig3_bulge_radius(bd_engine, PSF_model)
    fig3_disk_spectrum(bd_engine, PSF_model)

if __name__ == '__main__':
    fig3data(sys.argv)
    fig3plot()
