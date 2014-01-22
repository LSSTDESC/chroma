import os

import numpy as np
import lmfit
import galsim

def fiducial_galaxy():
    '''Bulge + disk parameters of the fiducial galaxy described in Voigt+12.'''
    gparam = lmfit.Parameters()
    # bulge
    gparam.add('b_x0', value=0.1)
    gparam.add('b_y0', value=0.3)
    gparam.add('b_n', value=4.0, vary=False)
    gparam.add('b_hlr', value=1.1 * 1.1)
    gparam.add('b_flux', value=0.25)
    gparam.add('b_gmag', value=0.2)
    gparam.add('b_phi', value=0.0)
    # disk
    gparam.add('d_x0', expr='b_x0')
    gparam.add('d_y0', expr='b_y0')
    gparam.add('d_n', value=1.0, vary=False)
    gparam.add('d_hlr', value=1.1)
    gparam.add('d_flux', expr='1.0 - b_flux')
    gparam.add('d_gmag', expr='b_gmag')
    gparam.add('d_phi', expr='b_phi')
    # initialize constrained variables
    dummyfit = lmfit.Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()
    return gparam

data_dir = '../../../data/'

def fig3_fiducial():
    b_wave, b_flambda = np.genfromtxt(data_dir+'SEDs/CWW_E_ext.ascii').T
    b_SED = galsim.SED(wave=b_wave, flambda=b_flambda)
    d_wave, d_flambda = np.genfromtxt(data_dir+'SEDs/CWW_Sbc_ext.ascii').T
    d_SED = galsim.SED(wave=d_wave, flambda=d_flambda)

    print
    print 'Running on fiducial galaxy parameters'
    print

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    file_ = open('output/fig3_fiducial.dat', 'w')
    for fw in [150, 250, 350, 450]:
        gparam = fiducial_galaxy()
        filter_file = data_dir+'/filters/Euclid_{:03d}.dat'.format(fw)

        # here be algorithm...
        mono_bulge = galsim.Sersic(n=gparam['b_n'].value,
                                   half_light_radius=gparam['b_hlr'].value)
        mono_bulge.applyShear(g=gparam['b_gmag'].value,
                              beta=gparam['b_phi'].value * galsim.radians)
        bulge = galsim.Chromatic(mono_bulge, b_SED)
        mono_disk = galsim.Sersic(n=gparam['d_n'].value,
                                  half_light_radius=gparam['d_hlr'].value)
        mono_disk.applyShear(g=gparam['d_gmag'].value,
                             beta=gparam['d_phi'].value * galsim.radians)
        disk = galsim.Chromatic(mono_disk, d_SED)

        # here should adjust FWHM(gal convolved with PSF) / FWHM(PSF) -> 1.4

        # now generate target image.
        base_PSF = galsim.Gaussian(half_light_radius=0.7)
        base_PSF.applyShear(g=0.05, beta=0.0*galsim.radians)
        PSF = galsim.ChromaticShiftAndDilate(base_PSF, shift_fn = lambda w:(w/520.0)**0.6)
        gal = galsim.Convolve([])

if __name__ == '__main__':
    fig3_fiducial()
