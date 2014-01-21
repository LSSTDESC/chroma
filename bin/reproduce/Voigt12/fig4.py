# fig4.py
#
# Reproduce Figure 4 from Voigt et al. (2012).
#
# The Voigt paper investigates the effects of the wavelength-dependent Euclid PSF on estimating
# shapes of galaxies with color gradients.  In figure 4, the multiplicative shear calibration
# parameter `m`, defined such that the estimated shear `gamma_hat` is related to the true shear
# `gamma` as
#    gamma_hat = (1 + m) * gamma + c
# (where `c` is the additive shear calibration parameter, ignored in this figure) are plotted for
# various bulge+disk model galaxies as imaged by the fiducial 350nm wide Euclid imaging filter.
# Parameters varied include:
# upper left panel:  the Sersic index of the bulge
# upper right panel:  the bulge-to-total flux ratio
# lower left panel:  the galaxy ellipticity
# lower right pane:  the y-coordinate of the galaxy centroid (looking at sub pixel shifts).
#
# This python script attempts to reproduce this figure using three different algorithms for
# generating the images used in the ring test:
#
# 1) Using the algorithm described in Voigt+12.
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
import matplotlib.pyplot as plt

import _mypath
import chroma

from fig_utils import *

data_dir = '../../../data/'

def fig4_bulge_sersic_index(bd_engine, PSF_model):
    '''Compute `m` dependence on bulge Sersic index (wrt fiducial galaxy)'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    filter_file = data_dir+'/filters/Euclid_350.dat'
    redshift = 0.9

    print
    print 'Varying bulge Sersic index'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_bulge_sersic_index.dat', 'w')
    for bulge_n in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        gparam = fiducial_galaxy()
        gparam['b_n'].value = bulge_n
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(bulge_n, c, m))
    fil.close()

def fig4_bulge_flux(bd_engine, PSF_model):
    '''Compute `m` dependence on bulge-to-total flux ratio (wrt fiducial galaxy)'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    filter_file = data_dir+'/filters/Euclid_350.dat'
    redshift = 0.9

    print
    print 'Varying bulge flux'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_bulge_flux.dat', 'w')
    for bulge_flux in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        gparam = fiducial_galaxy()
        gparam['b_flux'].value = bulge_flux
        gparam['d_flux'].value = 1.0 - bulge_flux
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(bulge_flux, c, m))
    fil.close()

def fig4_gal_ellip(bd_engine, PSF_model):
    '''Compute `m` dependence on galaxy ellipticity (wrt fiducial galaxy)'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    filter_file = data_dir+'/filters/Euclid_350.dat'
    redshift = 0.9

    print
    print 'Varying galaxy ellipticity'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_gal_ellip.dat', 'w')
    for gal_ellip in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        gparam = fiducial_galaxy()
        gparam['b_gmag'].value = gal_ellip
        gparam['d_gmag'].value = gal_ellip
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(gal_ellip, c, m))
    fil.close()

def fig4_y0(bd_engine, PSF_model):
    '''Compute `m` dependence on y-coordinate of galaxy center (wrt fiducial galaxy)'''
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_SED_file = data_dir+'/SEDs/CWW_E_ext.ascii'
    disk_SED_file = data_dir+'/SEDs/CWW_Sbc_ext.ascii'
    filter_file = data_dir+'/filters/Euclid_350.dat'
    redshift = 0.9

    print
    print 'Varying galaxy centroid y0'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fil = open('output/fig4_y0.dat', 'w')
    for y0 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        gparam = fiducial_galaxy()
        gparam['b_y0'].value = y0
        gparam['d_y0'].value = y0
        m, c = measure_shear_calib(gparam, filter_file, bulge_SED_file, disk_SED_file, redshift,
                                   PSF_ellip, PSF_phi, PSF_model, bd_engine)
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(y0, c, m))
    fil.close()

def fig4data(argv):
    '''Main routine to reproduce Voigt+12 figure 4.

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

    fig4_bulge_sersic_index(bd_engine, PSF_model)
    fig4_bulge_flux(bd_engine, PSF_model)
    fig4_gal_ellip(bd_engine, PSF_model)
    fig4_y0(bd_engine, PSF_model)

if __name__ == '__main__':
    fig4data(sys.argv)
