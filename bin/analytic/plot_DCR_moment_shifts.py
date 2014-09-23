"""Plot differential chromatic refraction data placed in pickle files by analytic_table.py.
"""

import cPickle
import os

import numpy as np
import matplotlib
try:
    matplotlib.use('Agg')
except:
    pass
import matplotlib.pyplot as plt

def plot_DCR_moment_shifts(filter_name):
    print "Plotting {}-band first moment shift".format(filter_name)
    stars = cPickle.load(open('output/stars.pkl'))
    gals = cPickle.load(open('output/galaxies.pkl'))

    #------------------------------#
    # Differences in first moments #
    #------------------------------#

    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(111)
    xlim = (-0.1, 3.0)
    ax1.set_xlim(xlim)
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}$ (arcsec)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = 45 degrees, filter = {}'.format(filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    star_types = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    # Normalize all chromatic shifts to the shift for a G5v star.
    G_idx = stars['star_type'] == 'ukg5v'

    # plot stars
    for star_name, star_type, star_color in zip(star_names, star_types, star_colors):
        star_idx = stars['star_type'] == star_type
        Rbar = stars[star_idx]['Rbar'][filter_name]
        dRbar = (Rbar - stars[G_idx]['Rbar'][filter_name]) * 180.0 / np.pi * 3600.0
        ax1.scatter(0.0, dRbar, c=star_color, marker='*', s=160, label=star_name, zorder=3)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']

    # plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        Rbar = gals[gal_idx]['Rbar'][filter_name]
        dRbar = (Rbar - stars[G_idx]['Rbar'][filter_name]) * 180.0 / np.pi * 3600.0
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, dRbar, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    # plot bars showing LSST/DES requirements
    LSST_Rbar_req = np.sqrt(3e-3)
    DES_Rbar_req = np.sqrt(8e-3)
    ax1.fill_between(xlim, [-DES_Rbar_req]*2, [DES_Rbar_req]*2, color='#DDDDDD', zorder=2)
    ax1.fill_between(xlim, [-LSST_Rbar_req]*2, [LSST_Rbar_req]*2, color='#AAAAAA', zorder=2)

    f.tight_layout()
    f.savefig('output/Rbar.{}.png'.format(filter_name), dpi=220)

    #-------------------------------#
    # Differences in second moments #
    #-------------------------------#
    print "Plotting {}-band second moment shift".format(filter_name)

    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(111)
    ax1.set_xlim(xlim)
    ax1.set_ylabel('$\Delta \mathrm{V}$ (arcsec$^2$)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = 45 degrees, filter = {}'.format(filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    #plot stars
    for star_name, star_type, star_color in zip(star_names, star_types, star_colors):
        star_idx = stars['star_type'] == star_type
        V = stars[star_idx]['V'][filter_name]
        dV = (V - stars[G_idx]['V'][filter_name]) * (180.0 / np.pi * 3600.0)**2
        ax1.scatter(0.0, dV, c=star_color, marker='*', s=160, label=star_name, zorder=3)

    #plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        V = gals[gal_idx]['V'][filter_name]
        dV = (V - stars[G_idx]['V'][filter_name]) * (180.0 / np.pi * 3600.0)**2
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, dV, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    LSST_V_req = 4.8e-4
    DES_V_req = 2.9e-3
    ax1.fill_between(xlim, [-DES_V_req]*2, [DES_V_req]*2, color='#DDDDDD', zorder=2)
    ax1.fill_between(xlim, [-LSST_V_req]*2, [LSST_V_req]*2, color='#AAAAAA', zorder=2)

    f.tight_layout()
    f.savefig('output/V.{}.png'.format(filter_name), dpi=220)

if __name__ == '__main__':
    plot_DCR_moment_shifts('LSST_g')
    plot_DCR_moment_shifts('LSST_r')
    plot_DCR_moment_shifts('LSST_i')
