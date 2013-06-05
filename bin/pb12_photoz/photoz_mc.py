# see how bad R and V reconstruction are when using photometric redshifts to estimate SEDs
# step 1: draw galaxy SED uniformly from templates
# step 2: draw galaxy redshift uniformly in redshift from 0 to 3
# step 3: estimate photoz assuming sigma_z = 0.02 * (1+z)

import random

import numpy
import astropy.utils.console

import _mypath
import chroma

def moments(s_wave, s_flux, f_wave, f_throughput, zenith, **kwargs):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    m = chroma.disp_moments(wave, photons, zenith=zenith * numpy.pi / 180.0, **kwargs)
    return m
    # gaussian_sigma = 1.0 / 2.35 # 1 arcsec FWHM -> sigma
    # m2 = chroma.weighted_second_moment(wave, photons, 1.0,
    #                                    zenith=zenith * numpy.pi / 180.0,
    #                                    Rbar=m[0], **kwargs)
    # return m[0], m[1], m2

def photoz_mc(niter, filtername, zenith, sigma_z=0.02):
    SED_dir = '../../data/SEDs/'
    gal_SEDs = ['CWW_E_ext', 'CWW_Im_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                'KIN_Sa_ext', 'KIN_Sb_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']

    filter_dir = '../../data/filters/'
    filter_file = filter_dir + 'LSST_{}.dat'.format(filtername)

    f_data = numpy.genfromtxt(filter_file)
    f_wave = f_data[:,0]
    f_throughput = f_data[:,1]

    s_waves = []
    s_photons = []
    for s in gal_SEDs:
        SED_file = SED_dir + s + '.ascii'
        s_data = numpy.genfromtxt(SED_file)
        s_waves.append(s_data[:,0])
        s_photons.append(s_data[:,1] * s_data[:,0])

    zplot = numpy.empty(0)
    Rbar_plot = numpy.empty(0)
    V_plot = numpy.empty(0)

    with astropy.utils.console.ProgressBar(niter) as bar:
        for i in range(niter):
            iSED = random.randint(0,7)
            SED = gal_SEDs[iSED]
            z_spec = random.uniform(0.0, 3.0)
            z_phot = random.gauss(z_spec, sigma_z * (1.0 + z_spec))

            obs_wave = s_waves[iSED] * (1.0 + z_spec)
            model_wave = s_waves[iSED] * (1.0 + z_phot)
            obs_m = moments(obs_wave, s_photons[iSED], f_wave, f_throughput, zenith)
            model_m = moments(model_wave, s_photons[iSED], f_wave, f_throughput, zenith)
            zplot = numpy.append(zplot, z_spec)
            Rbar_plot = numpy.append(Rbar_plot, obs_m[0] - model_m[0])
            V_plot = numpy.append(V_plot, obs_m[1] - model_m[1])
            bar.update()
    return zplot, Rbar_plot, V_plot

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    z, R, V = photoz_mc(5000, 'r', 30.0)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.hist(R * 3600 * 180 / numpy.pi, 50)
    ax1.set_xlabel('$\Delta \overline{\mathrm{R}}$')
    ax2 = fig.add_subplot(212)
    ax2.hist(V * (3600 * 180 / numpy.pi)**2, 50)
    ax2.set_xlabel('$\Delta \mathrm{V}$')
    plt.show()
