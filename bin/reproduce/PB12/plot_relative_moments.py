import os

import numpy as np
import astropy.utils.console
import matplotlib.pyplot as plt

import _mypath
import chroma

def compute_relative_moments(filter_name, zenith, **kwargs):
    spec_dir = '../../../data/SEDs/'
    filter_dir = '../../../data/filters/'

    f_wave, f_throughput = np.genfromtxt(filter_dir + '{}.dat'.format(filter_name)).T
    bandpass = chroma.Bandpass(f_wave, f_throughput)
    # KIN galaxy spectra are cut-off above 994nm, so truncate redward wavelengths
    # Note this makes y-band calculations impossible.
    # Also truncate out-of-band leaks below 5e-3.
    bandpass.truncate(rel_throughput=5.e-3, redlim=994)

    G5v_wave, G5v_flambda = np.genfromtxt(spec_dir + 'ukg5v.ascii').T
    G5v_SED = chroma.SED(G5v_wave, G5v_flambda)

    G5v_mom = G5v_SED.DCR_moment_shifts(bandpass, zenith, **kwargs)

    star_types = ['uko5v',
                  'ukb5iii',
                  'uka5v',
                  'ukf5v',
                  'ukg5v',
                  'ukk5v',
                  'ukm5v']
    star_diffs = {}
    for star_type in star_types:
        star_diffs[star_type] = {}
        wave, flambda = np.genfromtxt(spec_dir + star_type + '.ascii').T
        star_SED = chroma.SED(wave, flambda)

        m = star_SED.DCR_moment_shifts(bandpass, zenith, **kwargs)
        star_diffs[star_type]['M1'] = (m[0] - G5v_mom[0]) * 180 / np.pi * 3600 # rad -> arcsec
        # rad^2 -> arcsec^2
        star_diffs[star_type]['M2'] = (m[1] - G5v_mom[1]) * (180 / np.pi * 3600)**2

    gal_types= ['CWW_E_ext',
                'KIN_Sa_ext',
                'KIN_Sb_ext',
                'CWW_Sbc_ext',
                'CWW_Scd_ext',
                'CWW_Im_ext',
                'KIN_SB1_ext',
                'KIN_SB6_ext']

    gal_diffs = {}
    with astropy.utils.console.ProgressBar(100 * len(gal_types)) as bar:
        for gal_type in gal_types:
            gal_diffs[gal_type] = {'M1':[], 'M2':[], 'wM2':[]}
            wave, flambda = np.genfromtxt(spec_dir + gal_type + '.ascii').T
            gal_SED = chroma.SED(wave, flambda)
            for z in np.arange(0.0, 1.3, 0.02):
                bar.update()
                gal_SED.set_redshift(z)
                m = gal_SED.DCR_moment_shifts(bandpass, zenith, **kwargs)
                # rad -> arcsec, rad^2 -> arcsec^2
                gal_diffs[gal_type]['M1'].append((m[0] - G5v_mom[0]) * 180 / np.pi * 3600)
                gal_diffs[gal_type]['M2'].append((m[1] - G5v_mom[1]) * (180 / np.pi * 3600)**2)
    return star_diffs, gal_diffs

def plot_analytic_moments(filter_name, zenith=45.0):
    a_star_diff, a_gal_diff = compute_relative_moments(filter_name, zenith)
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
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals1 = np.empty(0)
    yvals2 = np.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['M1']
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, label=star_name)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = np.arange(0.0, 1.3, 0.02)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['M1']
        ax1.plot(zs, analytic, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    f.savefig('output/Rbar.{}.z{}.png'.format(filter_name, zenith))

    # V plot, unweighted
    ####################

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = plt.subplot(111)
    ax1.set_xlim(-0.1, 1.3)
    ax1.set_ylabel('$\Delta \mathrm{V}$ (arcsec$^2$)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = {}, filter = {}'.format(zenith, filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['M2']
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, label=star_name)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = np.arange(0.0, 1.3, 0.02)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['M2']
        ax1.plot(zs, analytic, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    f.savefig('output/V.{}.z{}.png'.format(filter_name, zenith))

if __name__ == '__main__':
    plot_analytic_moments('LSST_g', 45.0)
