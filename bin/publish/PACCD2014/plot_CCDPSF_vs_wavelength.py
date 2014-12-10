import cPickle
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        stardata = cPickle.load(open('../../analytic/output/stars.pkl'))
        galdata = cPickle.load(open('../../analytic/output/galaxies.pkl'))
    except:
        raise IOError("Need to run analytic_table.py script in directory $CHROMA/bin/analytic/")

    # LSST r-band power law size figure
    # Want multiplicative bias (vs G5v star) vs. redshift
    rsqr_gal = 0.3**2 # arcsec^2
    Ixx0 = 0.1147 # size of constant part of PSF (telescope, atm) (Moffat beta=5, FWHM=0.64)
    Ixx1 = 0.0065 # size of CCD part at effective wavelength (Gaussian FWHM=0.19)

    f = plt.figure(figsize=(6, 4))
    ax = f.add_subplot(111)
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(-0.006, 0.006)
    ax.set_xlabel("redshift")
    ax.set_ylabel("multiplicative bias")
    ax.set_title(r"$\beta_+ = 1\times\,10^{-5} \mathrm{arcsec}^2/\mathrm{nm}$")
    # Stars first
    G_idx = stardata['star_type'] == 'ukg5v'
    G_Ixx = Ixx0 + Ixx1 + stardata[G_idx]['linear']['LSST_r'] * (180/np.pi * 3600)**2
    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    for star, star_name, star_color in zip(stars, star_names, star_colors):
        star_idx = stardata['star_type'] == star
        Ixx = Ixx0 + Ixx1 + stardata[star_idx]['linear']['LSST_r'] * (180/np.pi * 3600)**2
        dIxx = Ixx - G_Ixx
        m = -2 * dIxx / rsqr_gal
        ax.scatter(0.0, m, c=star_color, marker='*', s=160, label=star_name, edgecolor='black',
                   zorder=3)

    # and galaxies
    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Magenta']
    w0 = galdata['gal_type'] == 'CWW_E_ext'
    zs = galdata[w0]['redshift']
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        gal_idx = galdata['gal_type'] == gal
        Ixx = Ixx0 + Ixx1 + galdata[gal_idx]['linear']['LSST_r'] * (180/np.pi * 3600)**2
        dIxx = Ixx - G_Ixx
        m = -2 * dIxx / rsqr_gal
        ax.plot(zs, m, c=gal_color, label=gal_name)

    ax.legend(prop={"size":9})
    ax.fill_between([-0.1, 3.0], [-0.003, -0.003], [0.003, 0.003],
                     color='grey', edgecolor='None', alpha=0.5)

    f.tight_layout()
    f.savefig('CCDPSF_size_vs_wavelength.pdf')


    # LSST r-band linear ellipticity figure
    # Want additive bias (vs G5v star) vs. redshift
    f = plt.figure(figsize=(6, 4))
    ax = f.add_subplot(111)
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(-0.003, 0.003)
    ax.set_xlabel("redshift")
    ax.set_ylabel("additive bias")
    ax.set_title(r"$\beta_- = 1\times\,10^{-5} \mathrm{arcsec}^2/\mathrm{nm}$")
    # Stars first
    G_idx = stardata['star_type'] == 'ukg5v'
    G_Ixx = Ixx0 + Ixx1 + stardata[G_idx]['linear']['LSST_r'] * (180/np.pi * 3600)**2
    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    for star, star_name, star_color in zip(stars, star_names, star_colors):
        star_idx = stardata['star_type'] == star
        Ixx = Ixx0 + Ixx1 + stardata[star_idx]['linear']['LSST_r'] * (180/np.pi * 3600)**2
        dIxx = Ixx - G_Ixx
        c = dIxx / rsqr_gal
        ax.scatter(0.0, c, c=star_color, marker='*', s=160, label=star_name, edgecolor='black',
                   zorder=3)

    # and galaxies
    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Magenta']
    w0 = galdata['gal_type'] == 'CWW_E_ext'
    zs = galdata[w0]['redshift']
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        gal_idx = galdata['gal_type'] == gal
        Ixx = Ixx0 + Ixx1 + galdata[gal_idx]['linear']['LSST_r'] * (180/np.pi * 3600)**2
        dIxx = Ixx - G_Ixx
        c = dIxx / rsqr_gal
        ax.plot(zs, c, c=gal_color, label=gal_name)

    ax.legend(prop={"size":9})
    ax.fill_between([-0.1, 3.0], [-0.0003, -0.0003], [0.0003, 0.0003],
                     color='grey', edgecolor='None', alpha=0.5)

    f.tight_layout()
    f.savefig('CCDPSF_ellip_vs_wavelength.pdf')
