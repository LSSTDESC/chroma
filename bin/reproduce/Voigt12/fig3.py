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
# 3) Using `galsim` to generate exact PSFs, but then caching these PSFs as oversampled images to
#    speed up the computation.
#
# By comparing a handful of both intermediate and final results, it appears that the differences
# between (2) and (3) are at least one order of magnitude smaller than the differences between (1)
# and either (2) or (3), indicating that the approximation introduced in (3) is not severely
# affecting the accuracy of the calculations.  On the other hand, (3) is several orders of magnitude
# faster to execute than (2).

import os
import sys

import numpy
import lmfit

import _mypath
import chroma

from fig_utils import *

data_dir = '../../../data/'

def fig3_fiducial(bd_engine, PSF_model):
    '''Generate `m` and `c` vs. filter width for the fiducial bulge+disk galaxy.'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Running on fiducial galaxy parameters'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_fiducial.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
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
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    redshift = 1.4

    print
    print 'Varying the redshift'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_redshift.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
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
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
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
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
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
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Im_ext.ascii'
    redshift = 0.9

    print
    print 'Varying the disk spectrum'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig3_disk_spectrum.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()


def fig3data(argv):
    '''Main routine to reproduce Voigt+12 figure 3.

    Expects argv[1] to take on one of 4 possible values:
    'gs' -- Use galsim with PSF image caching
    'gsfull' -- Use galsim without image caching: more exact, but much slower
    'voigt' -- User Voigt+12 algorithm for creating images
    None -- defaults to 'gs'
    '''
    if len(argv) == 1:
        use='gs'
    else:
        use=argv[1]
    if use == 'voigt':
        bd_engine = chroma.ImageEngine.VoigtBDEngine()
        PSF_model = chroma.PSF_model.VoigtEuclidPSF
    elif use == 'gs':
        bd_engine = chroma.ImageEngine.GalSimBDEngine()
        PSF_model = chroma.PSF_model.GSEuclidPSFInt
    elif use == 'gsfull':
        bd_engine = chroma.ImageEngine.GalSimBDEngine()
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
