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

def DCR_color_correction(shape_filter, color_filters):
    stars = cPickle.load(open('output/stars.pkl'))
    gals = cPickle.load(open('output/galaxies.pkl'))

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    star_types = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_pcolors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                 'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_pcolors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']

    # Normalize all chromatic shifts to the shift for a G5v star.
    G_idx = stars['star_type'] == 'ukg5v'

    # Populate lists of data for stars to plot and fit with
    star_dRbars = []
    star_dVs = []
    star_colors = []
    for star_type in star_types:
        star_idx = stars['star_type'] == star_type
        Rbar = stars[star_idx]['Rbar'][shape_filter]
        V = stars[star_idx]['V'][shape_filter]
        star_dRbars.append((Rbar - stars[G_idx]['Rbar'][shape_filter]) * 180.0 / np.pi * 3600.0)
        star_dVs.append((V - stars[G_idx]['V'][shape_filter]) * (180.0 / np.pi * 3600.0)**2)
        star_colors.append(stars[star_idx]['mag'][color_filters[0]][0]
                           - stars[star_idx]['mag'][color_filters[1]][0])

    # Fit the stars.  Leave out the troublesome M5v star at the end of the list.
    A_star = np.vstack([star_colors[:-1], np.ones(len(star_colors)-1)]).T # design matrix
    slope_star_Rbar, intercept_star_Rbar = np.linalg.lstsq(A_star, star_dRbars[:-1])[0]
    slope_star_V, intercept_star_V = np.linalg.lstsq(A_star, star_dVs[:-1])[0]

    # Make similar lists for galaxies, each element is a list this time to account for the
    # varying redshifts.
    gal_dRbars = []
    gal_dVs = []
    gal_colors = []
    for gal_type in gal_types:
        gal_idx = gals['gal_type'] == gal_type
        Rbar = gals[gal_idx]['Rbar'][shape_filter]
        V = gals[gal_idx]['V'][shape_filter]
        gal_dRbars.append((Rbar - gals[G_idx]['Rbar'][shape_filter]) * 180.0 / np.pi * 3600.0)
        gal_dVs.append((V - gals[G_idx]['V'][shape_filter]) * (180.0 / np.pi * 3600.0)**2)
        gal_colors.append(gals[gal_idx]['mag'][color_filters[0]]
                          - gals[gal_idx]['mag'][color_filters[1]])

    # Fit the galaxies.  Treat each redshift as an independent point to be fit.
    gal_all_colors = np.array(gal_colors).flatten()
    gal_all_dRbars = np.array(gal_dRbars).flatten()
    gal_all_dVs = np.array(gal_dVs).flatten()
    A_gal = np.vstack([gal_all_colors, np.ones(len(gal_all_colors))]).T # design matrix
    slope_gal_Rbar, intercept_gal_Rbar = np.linalg.lstsq(A_gal, gal_all_dRbars)[0]
    slope_gal_V, intercept_gal_V = np.linalg.lstsq(A_gal, gal_all_dVs)[0]

    # Open Rbar figure
    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}\,(\mathrm{arcsec})$')
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlabel('{} - {}'.format(*color_filters))
    ax2.set_ylabel('residual')
    ax1.set_title('zenith angle = 45 degrees, filter = {}'.format(shape_filter))

    # Scatter plot stars and residuals
    for dRbar, color, pcolor in zip(star_dRbars, star_colors, star_pcolors):
        ax1.scatter(color, dRbar, c=pcolor, marker='*', s=160, zorder=3)
        ax2.scatter(color, dRbar - (intercept_star_Rbar + slope_star_Rbar * color),
                    c=pcolor, marker='*', s=160, zorder=3)

    # Line plot galaxies and residuals
    for dRbar, color, pcolor in zip(gal_dRbars, gal_colors, gal_pcolors):
        ax1.plot(color, dRbar, c=pcolor)
        ax2.plot(color, dRbar - (intercept_gal_Rbar + slope_gal_Rbar * color), c=pcolor)

    # Plot trendlines
    cmin = np.hstack([gal_all_colors, star_colors]).min()
    cmax = np.hstack([gal_all_colors, star_colors]).max()
    color_range = np.array([cmin, cmax])
    ax1.plot(color_range, intercept_gal_Rbar + color_range * slope_gal_Rbar)
    ax1.plot(color_range, intercept_star_Rbar + color_range * slope_star_Rbar)
    f.tight_layout()

    xlim = ax2.get_xlim()
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.fill_between(xlim, [-0.025]*2, [0.025]*2, color='#DDDDDD', zorder=2)
    ax1.fill_between(xlim, [-0.01]*2, [0.01]*2, color='#AAAAAA', zorder=2)
    ax2.fill_between(xlim, [-0.025]*2, [0.025]*2, color='#DDDDDD', zorder=2)
    ax2.fill_between(xlim, [-0.01]*2, [0.01]*2, color='#AAAAAA', zorder=2)


    f.savefig('output/Rbar_{}_vs_{}-{}.png'.format(shape_filter,
                                                   color_filters[0],
                                                   color_filters[1]), dpi=220)

    # Open V figure
    f = plt.figure(figsize=(8,6))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.set_ylabel('$\Delta \mathrm{V}\,(\mathrm{arcsec}^2)$')
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlabel('{} - {}'.format(*color_filters))
    ax2.set_ylabel('residual')
    ax1.set_title('zenith angle = 45 degrees, filter = {}'.format(shape_filter))

    # Scatter plot stars and residuals
    for dV, color, pcolor in zip(star_dVs, star_colors, star_pcolors):
        ax1.scatter(color, dV, c=pcolor, marker='*', s=160, zorder=3)
        ax2.scatter(color, dV - (intercept_star_V + slope_star_V * color),
                    c=pcolor, marker='*', s=160, zorder=3)

    # Line plot galaxies and residuals
    for dV, color, pcolor in zip(gal_dVs, gal_colors, gal_pcolors):
        ax1.plot(color, dV, c=pcolor)
        ax2.plot(color, dV - (intercept_gal_V + slope_gal_V * color), c=pcolor)

    # Plot trendlines
    ax1.plot(color_range, intercept_gal_V + color_range * slope_gal_V)
    ax1.plot(color_range, intercept_star_V + color_range * slope_star_V)
    f.tight_layout()

    xlim = ax2.get_xlim()
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.fill_between(xlim, [-0.0006]*2, [0.0006]*2, color='#DDDDDD', zorder=2)
    ax1.fill_between(xlim, [-0.0001]*2, [0.0001]*2, color='#AAAAAA', zorder=2)
    ax2.fill_between(xlim, [-0.0006]*2, [0.0006]*2, color='#DDDDDD', zorder=2)
    ax2.fill_between(xlim, [-0.0001]*2, [0.0001]*2, color='#AAAAAA', zorder=2)

    f.savefig('output/V_{}_vs_{}-{}.png'.format(shape_filter,
                                                color_filters[0],
                                                color_filters[1]), dpi=220)

if __name__ == '__main__':
   DCR_color_correction('LSST_r', ['LSST_r', 'LSST_i'])
