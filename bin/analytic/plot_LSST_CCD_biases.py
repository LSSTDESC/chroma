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

def plot_LSST_CCD_additive_chromatic_bias(filter_name):
    print "Plotting {}-band additive bias".format(filter_name)
    stars = cPickle.load(open('output/stars.pkl'))
    gals = cPickle.load(open('output/galaxies.pkl'))

    Ixx_atmscope = 0.12 # arcsec^2
    Iyy_atmscope = 0.12 # symmetric
    r2gal = 0.3**2 # arcsec^2

    f = plt.figure(figsize=(6,4))
    ax1 = plt.subplot(111)
    xlim = (-0.1, 3.0)
    ax1.set_xlim(xlim)
    ylim = (-0.003, 0.003)
    ax1.set_ylim(ylim)
    ax1.set_ylabel('additive bias')
    ax1.set_xlabel('redshift')
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
        # last factor below is rad^2 -> arcsec^2
        Ixx_CCD = 0.0065 + stars[star_idx]['linear'][filter_name] * (180/np.pi * 3600)**2
        dIxx_CCD = Ixx_CCD - (0.0065 + stars[G_idx]['linear'][filter_name] * (180/np.pi * 3600)**2)
        dIyy_CCD = -dIxx_CCD #in this model
        c1 = (dIxx_CCD - dIyy_CCD) / (2 * r2gal)
        ax1.scatter(0.0, c1, facecolor=star_color, edgecolor='k', marker='*', s=160, 
                    label=star_name, zorder=3)

    ax1.text(0.05, 0.05, r"$\beta_- = 1 \times 10^{-5} \mathrm{arcsec}^2/\,\mathrm{nm}$", 
             transform=ax1.transAxes)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']

    # plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        Ixx_CCD = 0.0065 + gals[gal_idx]['linear'][filter_name] * (180./np.pi * 3600)**2
        dIxx_CCD = Ixx_CCD - (0.0065 + stars[G_idx]['linear'][filter_name] * (180/np.pi * 3600)**2)
        dIyy_CCD = -dIxx_CCD
        c1 = (dIxx_CCD - dIyy_CCD) / (2 * r2gal)
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, c1, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    # plot bars showing LSST/DES requirements
    LSST_c1_req = np.sqrt(1.7e-7)/2
    ax1.fill_between(xlim, [-LSST_c1_req]*2, [LSST_c1_req]*2, color='#AAAAAA', zorder=2)

    f.tight_layout()
    f.savefig('output/CCD_c.{}.png'.format(filter_name), dpi=220)
    f.savefig('output/CCD_c.{}.pdf'.format(filter_name), dpi=220)

def plot_LSST_CCD_multiplicative_chromatic_bias(filter_name):
    print "Plotting {}-band multiplicative bias".format(filter_name)
    stars = cPickle.load(open('output/stars.pkl'))
    gals = cPickle.load(open('output/galaxies.pkl'))

    Ixx_atmscope = 0.12 # arcsec^2
    Iyy_atmscope = 0.12 # symmetric
    r2gal = 0.3**2 # arcsec^2

    f = plt.figure(figsize=(6,4))
    ax1 = plt.subplot(111)
    xlim = (-0.1, 3.0)
    ax1.set_xlim(xlim)
    ylim = (-0.006, 0.006)
    ax1.set_ylim(ylim)
    ax1.set_ylabel('multiplicative bias')
    ax1.set_xlabel('redshift')
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
        # last factor below is rad^2 -> arcsec^2
        Ixx_CCD = 0.0065 + stars[star_idx]['linear'][filter_name] * (180/np.pi * 3600)**2
        dIxx_CCD = Ixx_CCD - (0.0065 + stars[G_idx]['linear'][filter_name] * (180/np.pi * 3600)**2)
        dIyy_CCD = dIxx_CCD #in this model
        m = -(dIxx_CCD + dIyy_CCD) / r2gal
        ax1.scatter(0.0, m, facecolor=star_color, edgecolor='k', marker='*', s=160, 
                    label=star_name, zorder=3)

    ax1.text(0.05, 0.90, r"$\beta_+ = 1 \times 10^{-5} \mathrm{arcsec}^2/\,\mathrm{nm}$", 
             transform=ax1.transAxes)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']

    # plot gals
    for gal_name, gal_type, gal_color in zip(gal_names, gal_types, gal_colors):
        gal_idx = gals['gal_type'] == gal_type
        Ixx_CCD = 0.0065 + gals[gal_idx]['linear'][filter_name] * (180./np.pi * 3600)**2
        dIxx_CCD = Ixx_CCD - (0.0065 + stars[G_idx]['linear'][filter_name] * (180/np.pi * 3600)**2)
        dIyy_CCD = dIxx_CCD
        m = -(dIxx_CCD + dIyy_CCD) / r2gal
        zs = gals[gal_idx]['redshift']
        ax1.plot(zs, m, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    # plot bars showing LSST/DES requirements
    LSST_m_req = 3.e-3
    ax1.fill_between(xlim, [-LSST_m_req]*2, [LSST_m_req]*2, color='#AAAAAA', zorder=2)

    f.tight_layout()
    f.savefig('output/CCD_m.{}.png'.format(filter_name), dpi=220)
    f.savefig('output/CCD_m.{}.pdf'.format(filter_name), dpi=220)

if __name__ == '__main__':
    plot_LSST_CCD_additive_chromatic_bias('LSST_r')
    plot_LSST_CCD_multiplicative_chromatic_bias('LSST_r')
    plot_LSST_CCD_additive_chromatic_bias('LSST_g')
    plot_LSST_CCD_multiplicative_chromatic_bias('LSST_g')
    plot_LSST_CCD_additive_chromatic_bias('LSST_i')
    plot_LSST_CCD_multiplicative_chromatic_bias('LSST_i')
