# see how bad R and V reconstruction are when using photometric redshifts to estimate SEDs
# step 1: draw galaxy SED uniformly from templates
# step 2: draw galaxy redshift uniformly in redshift from 0 to 3
# step 3: estimate photoz assuming sigma_z = 0.02 * (1+z)

import random
import os

import numpy
import astropy.utils.console

import _mypath
import chroma

data_dir = '../../../data/'

def moments(s_wave, s_flux, f_wave, f_throughput, zenith, **kwargs):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    m = chroma.disp_moments(wave, photons, zenith=zenith, **kwargs)
    s = chroma.relative_second_moment_radius(wave, photons)

    return m[0], m[1], s

def photoz_mc(niter, filtername, zenith, sigma_z=0.02):
    gal_SEDs = ['CWW_E_ext', 'CWW_Im_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                'KIN_Sa_ext', 'KIN_Sb_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']

    filter_file = data_dir + 'filters/LSST_{}.dat'.format(filtername)

    f_wave, f_throughput = numpy.genfromtxt(filter_file).T

    s_waves = []
    s_photons = []
    for s in gal_SEDs:
        SED_file = data_dir+'SEDs/'+s+'.ascii'
        s_wave, s_photon = numpy.genfromtxt(SED_file).T
        s_waves.append(s_wave)
        s_photons.append(s_photon * s_wave)

    zplot = numpy.empty(0)
    Rbar_plot = numpy.empty(0)
    V_plot = numpy.empty(0)
    S_plot = numpy.empty(0)

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
            S_plot = numpy.append(S_plot, obs_m[2] - model_m[2])
            bar.update()
    return zplot, Rbar_plot, V_plot, S_plot

def plot_photoz_mc(niter, filtername, zenith, sigma_z=0.02):
    import matplotlib.pyplot as plt

    z, R, V, S = photoz_mc(niter, filtername, zenith, sigma_z=sigma_z)
    R *= 3600 * 180 / numpy.pi
    V *= (3600 * 180 / numpy.pi)**2

    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot2grid((2,4), (0,0), colspan=3)
    ax2 = plt.subplot2grid((2,4), (0,3))
    ax3 = plt.subplot2grid((2,4), (1,0), colspan=3)
    ax4 = plt.subplot2grid((2,4), (1,3))
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0.1)

    ax1.scatter(z, R, s=3)
    ax2.hist(R, bins=50, color='Red', orientation='horizontal', range=[-0.006, 0.006])
    ax2xlim=ax2.get_xlim()
    ax1.fill_between([0.0, 3.0], [-6e-3, -6e-3], [6e-3, 6e-3], color='gray', alpha=0.2)
    ax2.fill_between([0.0, ax2xlim[1]], [-6e-3, -6e-3], [6e-3, 6e-3], color='gray', alpha=0.2)

    ax3.scatter(z, V, s=3)
    ax4.hist(V, bins=50, color='Red', orientation='horizontal', range=[-0.0003, 0.0003])
    ax4xlim=ax4.get_xlim()
    ax3.fill_between([0.0, 3.0], [-1e-4, -1e-4], [1e-4, 1e-4], color='gray', alpha=0.2)
    ax4.fill_between([0.0, ax4xlim[1]], [-1e-4, -1e-4], [1e-4, 1e-4], color='gray', alpha=0.2)
    ax4.set_xlim(ax4xlim)

    ax1.set_ylim([-0.006, 0.006])
    ax2.set_ylim([-0.006, 0.006])
    ax3.set_ylim([-0.0003, 0.0003])
    ax4.set_ylim([-0.0003, 0.0003])
    ax1.set_xlim([0.0, 3.0])
    ax3.set_xlim([0.0, 3.0])

    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}$')
    ax3.set_ylabel('$\Delta \mathrm{V}$')
    ax3.set_xlabel('redshift')
    title_text = 'zenith angle = {:02d} degrees, filter = {}'
    title_text = title_text.format(int(round(zenith * 180 / numpy.pi)), filtername)
    ax1.set_title(title_text)

    if not os.path.exists('output/'):
        os.mkdir('output/')
    plt.savefig('output/photoz_mc.{}.z{}.png'.format(filtername,
                                                     int(round(zenith * 180 / numpy.pi))),
                dpi=200)

    f = plt.figure(figsize=(8,3.5))
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((1, 4), (0, 3))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax1.scatter(z, S, s=3)
    ax2.hist(S, bins=50, color='Red', orientation='horizontal', range=[-0.005, 0.005])
    ax1.set_ylim(-0.005, 0.005)
    ax2.set_ylim(-0.005, 0.005)
    ax2xlim = ax2.get_xlim()

    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylabel('$\mathrm{phot} (\Delta r^2/r^2)\,-\,\mathrm{spec} (\Delta r^2/r^2)$')
    ax1.set_xlabel('redshift')
    ax1.set_title('{}-band'.format(filtername))

    ax1.fill_between([0.0, 3.0], [-0.0004, -0.0004], [0.0004, 0.0004],
                     color='gray', alpha=0.2)
    ax2.fill_between([0.0, ax2xlim[1]], [-0.0004, -0.0004], [0.0004, 0.0004],
                     color='gray', alpha=0.2)
    ax2.set_xlim(ax2xlim)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig('output/photoz_mc.S.{}.png'.format(filtername),
                dpi=200)

if __name__ == '__main__':
    plot_photoz_mc(2000, 'r', 30.0 * numpy.pi / 180)
