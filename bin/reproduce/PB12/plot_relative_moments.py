import os

import numpy as np
import astropy.utils.console
import matplotlib.pyplot as plt

import _mypath
import chroma

# def compute_relative_moments(filter_name, zenith, **kwargs):
#     spec_dir = '../../../data/SEDs/'
#     filter_dir = '../../../data/filters/'

#     f_wave, f_throughput = np.genfromtxt(filter_dir + '{}.dat'.format(filter_name)).T
#     bandpass = chroma.Bandpass(f_wave, f_throughput)
#     # KIN galaxy spectra are cut-off above 994nm, so truncate redward wavelengths
#     # Note this makes y-band calculations impossible.
#     # Also truncate out-of-band leaks below 5e-3.
#     bandpass.truncate(rel_throughput=5.e-3, redlim=994)

#     G5v_wave, G5v_flambda = np.genfromtxt(spec_dir + 'ukg5v.ascii').T
#     G5v_SED = chroma.SED(G5v_wave, G5v_flambda)

#     G5v_mom = G5v_SED.DCR_moment_shifts(bandpass, zenith, **kwargs)

#     star_types = ['uko5v',
#                   'ukb5iii',
#                   'uka5v',
#                   'ukf5v',
#                   'ukg5v',
#                   'ukk5v',
#                   'ukm5v']
#     star_diffs = {}
#     for star_type in star_types:
#         star_diffs[star_type] = {}
#         wave, flambda = np.genfromtxt(spec_dir + star_type + '.ascii').T
#         star_SED = chroma.SED(wave, flambda)

#         m = star_SED.DCR_moment_shifts(bandpass, zenith, **kwargs)
#         star_diffs[star_type]['M1'] = (m[0] - G5v_mom[0]) * 180 / np.pi * 3600 # rad -> arcsec
#         # rad^2 -> arcsec^2
#         star_diffs[star_type]['M2'] = (m[1] - G5v_mom[1]) * (180 / np.pi * 3600)**2

#     gal_types= ['CWW_E_ext',
#                 'KIN_Sa_ext',
#                 'KIN_Sb_ext',
#                 'CWW_Sbc_ext',
#                 'CWW_Scd_ext',
#                 'CWW_Im_ext',
#                 'KIN_SB1_ext',
#                 'KIN_SB6_ext']

#     gal_diffs = {}
#     with astropy.utils.console.ProgressBar(100 * len(gal_types)) as bar:
#         for gal_type in gal_types:
#             gal_diffs[gal_type] = {'M1':[], 'M2':[], 'wM2':[]}
#             wave, flambda = np.genfromtxt(spec_dir + gal_type + '.ascii').T
#             gal_SED = chroma.SED(wave, flambda)
#             for z in np.arange(0.0, 1.3, 0.02):
#                 bar.update()
#                 gal_SED.set_redshift(z)
#                 m = gal_SED.DCR_moment_shifts(bandpass, zenith, **kwargs)
#                 # rad -> arcsec, rad^2 -> arcsec^2
#                 gal_diffs[gal_type]['M1'].append((m[0] - G5v_mom[0]) * 180 / np.pi * 3600)
#                 gal_diffs[gal_type]['M2'].append((m[1] - G5v_mom[1]) * (180 / np.pi * 3600)**2)
#     return star_diffs, gal_diffs

def plot_analytic_moments(filter_name, zenith=45.0):
    stars = np.load('../../analytic/stars.npy')
    gals = np.load('../../analytic/galaxies.npy')

    #------------------------------#
    # Differences in first moments #
    #------------------------------#

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = plt.subplot(111)
    ax1.set_xlim(-0.1, 1.3)
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}$ (arcsec)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = {}, filter = {}'.format(zenith, filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    star_types = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    G_idx = stars['star_type'] == 'ukg5v'

    #plot stars
    for star_name, star_type, star_color in zip(star_names, star_types, star_colors):
        star_idx = stars['star_type'] == star_type
        Rbar = stars[star_idx]['Rbar'][filter_name]
        dRbar = (Rbar - stars[G_idx]['Rbar'][filter_name]) * 180.0 / np.pi * 3600.0
        ax1.scatter(0.0, dRbar, c=star_color, marker='*', s=160, label=star_name)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']

    #plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        Rbar = gals[gal_idx]['Rbar'][filter_name]
        dRbar = (Rbar - stars[G_idx]['Rbar'][filter_name]) * 180.0 / np.pi * 3600.0
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, dRbar, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    f.savefig('output/Rbar.{}.z{}.png'.format(filter_name, zenith))

    #-------------------------------#
    # Differences in second moments #
    #-------------------------------#

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = plt.subplot(111)
    ax1.set_xlim(-0.1, 1.3)
    ax1.set_ylabel('$\Delta \mathrm{V}$ (arcsec$^2$)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = {}, filter = {}'.format(zenith, filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    #plot stars
    for star_name, star_type, star_color in zip(star_names, star_types, star_colors):
        star_idx = stars['star_type'] == star_type
        V = stars[star_idx]['V'][filter_name]
        dV = (V - stars[G_idx]['V'][filter_name]) * (180.0 / np.pi * 3600.0)**2
        ax1.scatter(0.0, dV, c=star_color, marker='*', s=160, label=star_name)

    #plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        V = gals[gal_idx]['V'][filter_name]
        dV = (V - stars[G_idx]['V'][filter_name]) * (180.0 / np.pi * 3600.0)**2
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, dV, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    f.savefig('output/V.{}.z{}.png'.format(filter_name, zenith))

if __name__ == '__main__':
    plot_analytic_moments('LSST_g', 45.0)
