"""Plot differential chromatic refraction biases as a function of color, fit a trendline, and
also plot the residual.
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

def PSF_size_color_correction(shape_filter, color_filters, alpha):
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

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    star_types = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_pcolors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_pcolors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']

    # Normalize all chromatic shifts to the shift for a G5V star.
    G_idx = stars['star_type'] == 'ukg5v'

    # Populate lists of data for stars to plot and fit with
    star_dSbySs = []
    star_colors = []
    for star_type in star_types:
        star_idx = stars['star_type'] == star_type
        S = stars[star_idx][alpha_idx][shape_filter]
        star_dSbySs.append((S - stars[G_idx][alpha_idx][shape_filter]) / S)
        star_colors.append(stars[star_idx]['mag'][color_filters[0]][0]
                           - stars[star_idx]['mag'][color_filters[1]][0])

    # Fit the stars.  Leave out the troublesome M5v star at the end of the list.
    A_star = np.vstack([star_colors[:-1], np.ones(len(star_colors)-1)]).T # design matrix
    slope_star_S, intercept_star_S = np.linalg.lstsq(A_star, star_dSbySs[:-1])[0]

    # Make similar lists for galaxies, each element is a list this time to account for the
    # varying redshifts.
    gal_dSbySs = []
    gal_colors = []
    for gal_type in gal_types:
        gal_idx = gals['gal_type'] == gal_type
        S = gals[gal_idx][alpha_idx][shape_filter]
        gal_dSbySs.append((S - stars[G_idx][alpha_idx][shape_filter]) / S)
        gal_colors.append(gals[gal_idx]['mag'][color_filters[0]]
                          - gals[gal_idx]['mag'][color_filters[1]])

    # Fit the galaxies.  Treat each redshift as an independent point to be fit.
    gal_all_colors = np.array(gal_colors).flatten()
    gal_all_dSbySs = np.array(gal_dSbySs).flatten()
    wgood = np.isfinite(gal_all_dSbySs)
    A_gal = np.vstack([gal_all_colors[wgood], np.ones(wgood.sum())]).T # design matrix
    slope_gal_S, intercept_gal_S = np.linalg.lstsq(A_gal, gal_all_dSbySs[wgood])[0]

    # Open figure
    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.set_ylabel('$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$')
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlabel('{} - {}'.format(*color_filters))
    ax2.set_ylabel('residual')
    ax1.set_title('filter = {}'.format(shape_filter))

    # Scatter plot stars and residuals
    for dSbyS, color, pcolor in zip(star_dSbySs, star_colors, star_pcolors):
        ax1.scatter(color, dSbyS, c=pcolor, marker='*', s=160, zorder=3)
        ax2.scatter(color, dSbyS - (intercept_star_S + slope_star_S * color),
                    c=pcolor, marker='*', s=160, zorder=3)

    # Line plot galaxies and residuals
    for dSbyS, color, pcolor in zip(gal_dSbySs, gal_colors, gal_pcolors):
        ax1.plot(color, dSbyS, c=pcolor)
        ax2.plot(color, dSbyS - (intercept_gal_S + slope_gal_S * color), c=pcolor)

    # Plot trendlines
    cmin = np.hstack([gal_all_colors, star_colors]).min()
    cmax = np.hstack([gal_all_colors, star_colors]).max()
    color_range = np.array([cmin, cmax])
    ax1.plot(color_range, intercept_gal_S + color_range * slope_gal_S)
    ax1.plot(color_range, intercept_star_S + color_range * slope_star_S)
    f.tight_layout()

    xlim = ax2.get_xlim()
    ax2.set_xlim(xlim)
    ax1.set_xlim(xlim)
    if alpha == -0.2:
        alpha_str = 'S_m02'
        DES_size_req = 4.5e-3
        LSST_size_req = 9.8e-4
        ax1.fill_between(xlim, [-DES_size_req]*2, [DES_size_req]*2, color='#DDDDDD', zorder=2)
        ax1.fill_between(xlim, [-LSST_size_req]*2, [LSST_size_req]*2, color='#AAAAAA', zorder=2)
        ax2.fill_between(xlim, [-DES_size_req]*2, [DES_size_req]*2, color='#DDDDDD', zorder=2)
        ax2.fill_between(xlim, [-LSST_size_req]*2, [LSST_size_req]*2, color='#AAAAAA', zorder=2)
        ax1.set_ylim(-0.02, 0.02)
        ax2.set_ylim(-0.02, 0.02)
    elif alpha == 0.6:
        alpha_str = 'S_p06'
        Euclid_size_req = 0.001
        ax1.fill_between(xlim, [-Euclid_size_req]*2, [Euclid_size_req]*2, color='#AAAAAA', zorder=2)
        ax2.fill_between(xlim, [-Euclid_size_req]*2, [Euclid_size_req]*2, color='#AAAAAA', zorder=2)
    elif alpha == 1.0:
        alpha_str = 'S_p10'

    f.savefig('output/{}_{}_vs_{}-{}.png'.format(alpha_str,
                                                 shape_filter,
                                                 color_filters[0],
                                                 color_filters[1]), dpi=220)

if __name__ == '__main__':
   PSF_size_color_correction('LSST_g', ['LSST_g', 'LSST_r'], alpha=-0.2)
   PSF_size_color_correction('LSST_r', ['LSST_r', 'LSST_i'], alpha=-0.2)
   PSF_size_color_correction('Euclid_350', ['LSST_r', 'LSST_i'], alpha=0.6)
