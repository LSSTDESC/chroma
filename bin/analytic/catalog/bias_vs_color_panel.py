"""Make a panel figure for the paper of chromatic biases (RbarSqr, V, S_m02) vs r-i, both before
and after SVR correction.
"""

import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

band = 'LSST_r'
band_dict = {'LSST_r':"r",
             'LSST_i':"i"}
fontsize=12

# Some annotation arrow properties
arrowdict = dict(facecolor='black', shrink=0.1, width=1.5, headwidth=4, frac=0.2)

# hardcode some requirements, order is [DES, LSST]
r2sqr_gal = np.r_[0.4, 0.3]**2
r2sqr_PSF = np.r_[0.8, 0.7]**2

mean_m_req = np.r_[0.008, 0.003]
mean_DeltaRbarSqr_req = mean_m_req / 2.0
mean_DeltaV_req = r2sqr_gal * mean_m_req
mean_dS_m02_req = mean_m_req * r2sqr_gal / r2sqr_PSF

def set_range(x):
    """ Return a plotting range 30% larger than the interval containing 99% of the data.
    """
    xs = sorted(x)
    n = len(xs)
    low = xs[int(0.005*n)]
    high = xs[int(0.995*n)]
    span = high-low
    return [low - 0.3*span, high + 0.3*span]

def plot_panel(ax, galdata, colordata, cdata, cbands, ylabel, ylim):
    clim = [0.0, 2.2]

    rorder = np.random.permutation(len(colordata))

    im = ax.scatter(colordata[rorder], galdata[rorder], c=cdata[rorder],
                    vmin=clim[0], vmax=clim[1], zorder=4, s=3)
    im.set_rasterized(True)


    ax.set_xlabel(r"${} - {}$".format(band_dict[cbands[0]], band_dict[cbands[1]]),
                  fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_ylim(ylim)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)

    return im

def bias_vs_color_panel(gals, stars, band, cbands, outfile):
    f, axarr = plt.subplots(3, 2, figsize=(9, 11))

    colordata = gals['mag'][cbands[0]] - gals['mag'][cbands[1]]
    cdata = gals['redshift']

    # RbarSqr
    ylabel = r"$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)"
    stardata = stars['Rbar'][band] * 180/np.pi * 3600
    galdata = gals['Rbar'][band] * 180/np.pi * 3600
    norm = np.mean(stardata)
    stardata -= norm
    galdata -= norm
    stardata **= 2
    galdata **= 2
    ylim = set_range(galdata)
    ylim[0] = 0.0
    plot_panel(axarr[0, 0], galdata, colordata, cdata, cbands, ylabel, ylim)
    ax = axarr[0, 0]
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
    ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[0]]*2, color='#999999')
    ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[1]]*2, color='#777777')
    ax.axhline(-mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)

    # Corrected RbarSqr
    stardata = (stars['Rbar'][band] - stars['photo_Rbar'][band]) * 180/np.pi * 3600
    galdata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
    # d((DR)^2) = 2 DR d(DR)
    stardata = 2 * (stars['Rbar'][band] * 180/np.pi * 3600 - norm) * stardata
    galdata = 2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * galdata
    ylabel = r"$\delta(\left(\Delta \overline{\mathrm{R}}\right)^2)$ (arcsec$^2$)"
    plot_panel(axarr[0, 1], galdata, colordata, cdata, cbands, ylabel, ylim)
    ax = axarr[0, 1]
    ax.set_xlim(xlim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
    ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[0]]*2, color='#999999')
    ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[1]]*2, color='#777777')
    ax.axhline(-mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)

    # V
    ylabel = r"$\Delta \mathrm{V}}$ (arcsec$^2$)"
    stardata = stars['V'][band] * (180/np.pi * 3600)**2
    galdata = gals['V'][band] * (180/np.pi * 3600)**2
    norm = np.mean(stardata)
    stardata -= norm
    galdata -= norm
    ylim = set_range(np.concatenate([stardata, galdata]))
    plot_panel(axarr[1, 0], galdata, colordata, cdata, cbands, ylabel, ylim)
    ax = axarr[1, 0]
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
    ax.fill_between(xlim, [-mean_DeltaV_req[0]]*2, [mean_DeltaV_req[0]]*2, color='#999999')
    ax.fill_between(xlim, [-mean_DeltaV_req[1]]*2, [mean_DeltaV_req[1]]*2, color='#777777')
    ax.axhline(-mean_DeltaV_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-mean_DeltaV_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaV_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaV_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.annotate("LSST requirement",
                xy=(-0.35, -mean_DeltaV_req[1]),
                xytext=(-0.25, -mean_DeltaV_req[1]-0.0003),
                arrowprops=arrowdict,
                zorder=10)
    ax.annotate("DES requirement",
                xy=(-0.35, -mean_DeltaV_req[0]),
                xytext=(-0.25, -mean_DeltaV_req[0]-0.0003),
                arrowprops=arrowdict,
                zorder=10)

    # Corrected V
    stardata = (stars['V'][band] - stars['photo_'+'V'][band]) * (180/np.pi * 3600)**2
    galdata = (gals['V'][band] - gals['photo_'+'V'][band]) * (180/np.pi * 3600)**2
    ylabel = "$\delta(\Delta \mathrm{V})$ (arcsec$^2$)"
    plot_panel(axarr[1, 1], galdata, colordata, cdata, cbands, ylabel, ylim)
    ax = axarr[1, 1]
    ax.set_xlim(xlim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
    ax.fill_between(xlim, [-mean_DeltaV_req[0]]*2, [mean_DeltaV_req[0]]*2, color='#999999')
    ax.fill_between(xlim, [-mean_DeltaV_req[1]]*2, [mean_DeltaV_req[1]]*2, color='#777777')
    ax.axhline(-mean_DeltaV_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-mean_DeltaV_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaV_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(mean_DeltaV_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.annotate("LSST requirement",
                xy=(-0.35, -mean_DeltaV_req[1]),
                xytext=(-0.25, -mean_DeltaV_req[1]-0.0005),
                arrowprops=arrowdict,
                zorder=10)
    ax.annotate("DES requirement",
                xy=(-0.35, -mean_DeltaV_req[0]),
                xytext=(-0.25, -mean_DeltaV_req[0]-0.0005),
                arrowprops=arrowdict,
                zorder=10)

    # S
    ylabel = r"$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$"
    stardata = stars['S_m02'][band]
    galdata = gals['S_m02'][band]
    starmean = np.mean(stardata)
    stardata = (stardata - starmean)/starmean
    galdata = (galdata - starmean)/starmean
    ylim = set_range(np.concatenate([stardata, galdata]))
    plot_panel(axarr[2, 0], galdata, colordata, cdata, cbands, ylabel, ylim)
    ax = axarr[2, 0]
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
    ax.fill_between(xlim, [-mean_dS_m02_req[0]]*2, [mean_dS_m02_req[0]]*2, color='#999999')
    ax.fill_between(xlim, [-mean_dS_m02_req[1]]*2, [mean_dS_m02_req[1]]*2, color='#777777')
    ax.axhline(-mean_dS_m02_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-mean_dS_m02_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(mean_dS_m02_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(mean_dS_m02_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.annotate("LSST requirement",
                xy=(-0.35, -mean_dS_m02_req[1]),
                xytext=(-0.25, -mean_dS_m02_req[1]-0.003),
                arrowprops=arrowdict,
                zorder=10)
    ax.annotate("DES requirement",
                xy=(-0.35, -mean_dS_m02_req[0]),
                xytext=(-0.25, -mean_dS_m02_req[0]-0.003),
                arrowprops=arrowdict,
                zorder=10)

    # Corrected S
    stardata = (stars['S_m02'][band] - stars['photo_'+'S_m02'][band]) / stars['photo_'+'S_m02'][band]
    galdata = (gals['S_m02'][band] - gals['photo_'+'S_m02'][band]) / gals['photo_'+'S_m02'][band]
    ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"
    im = plot_panel(axarr[2, 1], galdata, colordata, cdata, cbands, ylabel, ylim)
    xlim = ax.get_xlim()
    ax = axarr[2, 1]
    ax.set_xlim(xlim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
    ax.fill_between(xlim, [-mean_dS_m02_req[0]]*2, [mean_dS_m02_req[0]]*2, color='#999999')
    ax.fill_between(xlim, [-mean_dS_m02_req[1]]*2, [mean_dS_m02_req[1]]*2, color='#777777')
    ax.axhline(-mean_dS_m02_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-mean_dS_m02_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(mean_dS_m02_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(mean_dS_m02_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.annotate("LSST requirement",
                xy=(-0.35, -mean_dS_m02_req[1]),
                xytext=(-0.25, -mean_dS_m02_req[1]-0.003),
                arrowprops=arrowdict,
                zorder=10)
    ax.annotate("DES requirement",
                xy=(-0.35, -mean_dS_m02_req[0]),
                xytext=(-0.25, -mean_dS_m02_req[0]-0.003),
                arrowprops=arrowdict,
                zorder=10)

    # colorbar
    colorbar_axes_range = [0.86, 0.77, 0.017, 0.19]
    cbar_ax = f.add_axes(colorbar_axes_range)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("redshift", fontsize=fontsize)

    f.tight_layout()
    f.savefig(outfile, dpi=400)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = "output/corrected_galaxy_data.pkl",
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = "output/corrected_star_data.pkl",
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('--band', default='LSST_r', nargs='?',
                        help="band of chromatic bias to plot (Default: 'LSST_r')")
    parser.add_argument('--color', default=['LSST_r', 'LSST_i'], nargs=2,
                        help="color to use for symbol color (Default: ['LSST_r', 'LSST_i'])")
    parser.add_argument('--outfile', default="output/bias_vs_color_panel.pdf", nargs='?',
                        help="output filename (Default: 'output/bias_vs_color_panel.pdf')")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))
    bias_vs_color_panel(gals, stars, args.band, args.color, args.outfile)
