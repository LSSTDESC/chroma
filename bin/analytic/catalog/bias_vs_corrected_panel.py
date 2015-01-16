"""Make a panel figure for the paper of chromatic biases (RbarSqr, V, S_m02) vs their residuals
after ETR correction.
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

def plot_panel(ax, xdata, ydata, cdata, xlabel, ylabel, xlim, ylim, clim, **kwargs):
    rorder = np.random.permutation(len(cdata))

    im = ax.scatter(xdata[rorder], ydata[rorder], c=cdata[rorder],
                    vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    ax.plot([-1e8,1e8], [-1e8,1e8], c='k')

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=45 )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=45 )
    # ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=45)
        # for label in ax.get_xticklabels():
    #     label.set_fontsize(fontsize)
    # for label in ax.get_yticklabels():
    #     label.set_fontsize(fontsize)

    return im

def bias_vs_corrected_panel(gals, stars, outfile, cbands=None):
    f, axarr = plt.subplots(3, 2, figsize=(9, 11))

    if cbands is None:
        cdata = gals['redshift']
        clim = [0., 2.0]
    else:
        cdata = gals['mag'][cbands[0]] - gals['mag'][cbands[1]]

    for col, band in zip([0,1], ['LSST_r','LSST_i']):
        # RbarSqr
        ylabel = r"$\delta(\left(\Delta \overline{\mathrm{R}}\right)^2)$ (arcsec$^2$)"
        xlabel = r"$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)"
        stardata = stars['Rbar'][band] * 180/np.pi * 3600
        galdata = gals['Rbar'][band] * 180/np.pi * 3600
        norm = np.mean(stardata)
        galdata -= norm
        galdata **= 2

        cgaldata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
        # d((DR)^2) = 2 DR d(DR)
        cgaldata = 2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * cgaldata
        ydata = cgaldata

        xlim = set_range(galdata)
        xlim[0] = 1.e-7
        ylim = set_range(galdata)
        ylim[0] = 1.e-7

        ax = axarr[0, col]
        ax.set_xscale('log')
        ax.set_yscale('log')
        plot_panel(ax, galdata, cgaldata, cdata, xlabel, ylabel, xlim, ylim, clim, s=3)
        # ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
        # ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[0]]*2, color='#999999')
        # ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[1]]*2, color='#777777')
        # ax.axhline(-mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
        # ax.axhline(-mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
        # ax.axhline(mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
        # ax.axhline(mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)

        # V
        ylabel = "$\delta(\Delta \mathrm{V})$ (arcsec$^2$)"
        xlabel = r"$\Delta \mathrm{V}}$ (arcsec$^2$)"
        stardata = stars['V'][band] * (180/np.pi * 3600)**2
        galdata = gals['V'][band] * (180/np.pi * 3600)**2
        norm = np.mean(stardata)
        galdata -= norm

        cgaldata = (gals['V'][band] - gals['photo_'+'V'][band]) * (180/np.pi * 3600)**2

        xlim = set_range(galdata)
        ylim = set_range(galdata)

        ax = axarr[1, col]
        plot_panel(ax, galdata, cgaldata, cdata, xlabel, ylabel, xlim, ylim, clim, s=3)
        # ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB')
        # ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[0]]*2, color='#999999')
        # ax.fill_between(xlim, [0.0]*2, [mean_DeltaRbarSqr_req[1]]*2, color='#777777')
        # ax.axhline(-mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
        # ax.axhline(-mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)
        # ax.axhline(mean_DeltaRbarSqr_req[0], c='k', alpha=0.1, zorder=10, lw=0.5)
        # ax.axhline(mean_DeltaRbarSqr_req[1], c='k', alpha=0.3, zorder=10, lw=0.5)


        # # S
        xlabel = r"$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$"
        ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"
        stardata = stars['S_m02'][band]
        galdata = gals['S_m02'][band]
        starmean = np.mean(stardata)
        galdata = (galdata - starmean)/starmean

        cgaldata = (gals['S_m02'][band] - gals['photo_'+'S_m02'][band]) / gals['photo_'+'S_m02'][band]

        xlim = set_range(galdata)
        ylim = set_range(galdata)

        ax = axarr[2, col]
        im = plot_panel(ax, galdata, cgaldata, cdata, xlabel, ylabel, xlim, ylim, clim, s=3)

    # colorbar
    # colorbar_axes_range = [0.86, 0.77, 0.017, 0.19]
    # cbar_ax = f.add_axes(colorbar_axes_range)
    # cbar = plt.colorbar(im, cax=cbar_ax)
    # cbar_ax.set_ylabel("redshift", fontsize=fontsize)

    f.tight_layout()
    f.savefig(outfile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = "output/corrected_galaxy_data.pkl",
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = "output/corrected_star_data.pkl",
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('--color', default=None, nargs=2,
                        help="color to use for symbol color (Default: None)")
    parser.add_argument('--outfile', default="output/bias_vs_corrected_panel.pdf", nargs='?',
                        help="output filename (Default: 'output/bias_vs_corrected_panel.pdf')")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))
    bias_vs_corrected_panel(gals, stars, args.outfile, cbands=args.color)
