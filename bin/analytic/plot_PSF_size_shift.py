"""Plot chromatic seeing biases data placed in pickle files by analytic_table.py.
"""

import cPickle
import os

import numpy as np
import matplotlib.pyplot as plt

def plot_PSF_size_shifts(filter_name, alpha):
    if alpha == -0.2:
        alpha_idx = 'S_m02'
    elif alpha == 0.6:
        alpha_idx = 'S_p06'
    elif alpha == 1.0:
        alpha_idx = 'S_p10'
    else:
        raise ValueError("Unknown value of alpha requested")

    stars = cPickle.load(open('output/stars.pkl'))
    gals = cPickle.load(open('output/galaxies.pkl'))

    #------------------------------#
    # Differences in first moments #
    #------------------------------#

    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(111)
    xlim = (-0.1, 3.0)
    ax1.set_xlim(xlim)
    ax1.set_ylabel('$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$')
    ax1.set_xlabel('redshift')
    ax1.set_title('filter = {}'.format(filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    star_types = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    G_idx = stars['star_type'] == 'ukg5v'

    #plot stars
    for star_name, star_type, star_color in zip(star_names, star_types, star_colors):
        star_idx = stars['star_type'] == star_type
        S = stars[star_idx][alpha_idx][filter_name]
        dSbyS = (S - stars[G_idx][alpha_idx][filter_name]) / S
        ax1.scatter(0.0, dSbyS, c=star_color, marker='*', s=160, label=star_name, zorder=3)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']

    #plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        S = gals[gal_idx][alpha_idx][filter_name]
        dSbyS = (S - stars[G_idx][alpha_idx][filter_name]) / S
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, dSbyS, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    if alpha == -0.2:
        alpha_str = 'S_m02'
        ax1.fill_between(xlim, [-0.0025]*2, [0.0025]*2, color='#DDDDDD', zorder=2)
        ax1.fill_between(xlim, [-0.0004]*2, [0.0004]*2, color='#AAAAAA', zorder=2)
    elif alpha == 0.6:
        alpha_str = 'S_p06'
        ax1.fill_between(xlim, [-0.002]*2, [0.002]*2, color='#AAAAAA', zorder=2)
    elif alpha == 1.0:
        alpha_str = 'S_p10'
    f.tight_layout()
    f.savefig('output/{}.{}.png'.format(alpha_str, filter_name), dpi=220)

if __name__ == "__main__":
    plot_PSF_size_shifts('LSST_r', -0.2)
    plot_PSF_size_shifts('LSST_i', -0.2)
    plot_PSF_size_shifts('Euclid_350', 0.6)
