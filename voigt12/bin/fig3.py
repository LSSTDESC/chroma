import _mypath
import numpy as np
import ringtest
from lmfit import Parameters, Minimizer
from VoigtImageFactory import VoigtImageFactory
from bdgal import *


def fiducial_galaxy():
    gparam = Parameters()
    gparam.add('b_x0', value=2.1)
    gparam.add('b_y0', value=3.3)
    gparam.add('b_n', value=4.0, vary=False)
    gparam.add('b_r_e', value=2.7)
    gparam.add('b_flux', value=0.25)
    gparam.add('b_gmag', value=0.4)
    gparam.add('b_phi', value=0.0)
    #disk
    gparam.add('d_x0', expr='b_x0')
    gparam.add('d_y0', expr='b_y0')
    gparam.add('d_n', value=1.0, vary=False)
    gparam.add('d_r_e', value=2.7 * 1.1)
    gparam.add('d_flux', expr='1.0 - b_flux')
    gparam.add('d_gmag', expr='b_gmag')
    gparam.add('d_phi', expr='b_phi')
    dummyfit = Minimizer(lambda x: 0, gparam)
    dummyfit.prepare_fit()
    return gparam

def fig3_fiducial(im_fac=None):
    if im_fac is None:
        im_fac = VoigtImageFactory()
    gparam = fiducial_galaxy()
    filter_widths = [150, 250, 350, 450]
    PSF_ellip = 0.05
    PSF_phi = 0.0
    bulge_sed_file = '../data/SEDs/CWW_E_ext.ascii'
    disk_sed_file = '../data/SEDs/CWW_Sbc_ext.ascii'
    redshift = 0.9

    print
    print 'Running on fiducial galaxy parameters'
    print

    fil = open('fig3_fiducial.dat', 'w')
    for fw in filter_widths:
        filter_file = '../data/filters/voigt12_{:03d}.dat'.format(fw)
        b_PSF, d_PSF, c_PSF, circ_c_PSF = build_PSFs(filter_file,
                                                     gparam['b_flux'].value,
                                                     bulge_sed_file, disk_sed_file,
                                                     redshift, PSF_ellip, PSF_phi)
        map(im_fac.load_PSF, [b_PSF, d_PSF, c_PSF, circ_c_PSF])

        set_fwhm_ratio(gparam, 1.4, circ_c_PSF, im_fac)
        gen_target_image = target_image_fn_generator(gparam, b_PSF, d_PSF, im_fac)
        gen_init_param = init_param_generator(gparam)
        measure_ellip = ellip_measurement_generator(c_PSF, im_fac)

        gamma0 = 0.0 + 0.0j
        gamma0_hat = ringtest.ringtest(gamma0, 3,
                                       gen_target_image,
                                       gen_init_param,
                                       measure_ellip)
        c = gamma0_hat.real, gamma0_hat.imag

        gamma1 = 0.01 + 0.02j
        gamma1_hat = ringtest.ringtest(gamma1, 3,
                                       gen_target_image,
                                       gen_init_param,
                                       measure_ellip)
        m0 = (gamma1_hat.real - c[0])/gamma1.real - 1.0
        m1 = (gamma1_hat.imag - c[1])/gamma1.imag - 1.0
        m = m0, m1
        print 'c:    {:10g}  {:10g}'.format(c[0], c[1])
        print 'm:    {:10g}  {:10g}'.format(m[0], m[1])
        fil.write('{} {} {}\n'.format(fw, c, m))
    fil.close()
