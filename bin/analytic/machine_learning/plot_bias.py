"""Read pkl files created by process_*_catalog.py and *_ML.py and make plots of chromatic biases
as functions of redshift, both before and after photometric corrections are estimated.  Run
`python plot_bias.py --help` for a list of command line options.
"""

import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

# Specify the locations of three different plot elements.
hist_axes_range = [0.13, 0.12, 0.1, 0.8]
scatter_axes_range = [0.23, 0.12, 0.73, 0.8]
colorbar_axes_range = [0.84, 0.15, 0.025, 0.35]
data_dir = '../../../data/'
star_table = '../../analytic/output/stars.pkl'

def hist_with_peak(x, bins=None, range=None, ax=None, orientation='vertical',
                   histtype=None, **kwargs):
    """Plot a histogram normalized to unit peak.
    """
    if ax is None:
        ax = plt.gca()
    hist, bin_edges = np.histogram(x, bins=bins, range=range)
    hist_n = hist * 1.0/hist.max()
    width = bin_edges[1] - bin_edges[0]
    x = np.ravel(zip(bin_edges[:-1], bin_edges[:-1]+width))
    y = np.ravel(zip(hist_n, hist_n))
    if histtype == 'step':
        if orientation == 'vertical':
            plt.plot(x, y, **kwargs)
        elif orientation == 'horizontal':
            plt.plot(y, x, **kwargs)
        else:
            raise ValueError
    elif histtype == 'stepfilled':
        if orientation == 'vertical':
            plt.fill(x, y, **kwargs)
        elif orientation == 'horizontal':
            plt.fill(y, x, **kwargs)
        else:
            raise ValueError
    else:
        raise ValueError

def set_range(x):
    """ Return a plotting range 30% larger than the interval containing 99% of the data.
    """
    xs = sorted(x)
    n = len(xs)
    low = xs[int(0.005*n)]
    high = xs[int(0.995*n)]
    span = high-low
    return [low - 0.3*span, high + 0.3*span]


def plot_bias(gals, stars, bias, band, cbands, outfile, corrected=False, **kwargs):
    """Produce a plot of a chromatic bias, which is one of:
       `Rbar`  - centroid shift due to differential chromatic refraction
       `V`     - second moment shift due to differential chromatic refraction
       `S_m02` - difference in r^2 due to FWHM \propto \lambda^{-0.2}, as per chromatic seeing
       `S_p06` - difference in r^2 due to FWHM \propto \lambda^{+0.6}, as per Euclid
       `S_p10` - difference in r^2 due to FWHM \propto \lambda^{+1.0}, as per diffraction limit
    Either plot the uncorrected shift relative to the shift of a G5v star, or plot the residual shift
    from the photometric estimate of the shift derived from machine learning.

    @param gals       galaxy recarray produced by either process_gal_catalog.py or gal_ML.py
    @param stars      star recarray produced by either process_star_catalog.py or star_ML.py
    @param bias       A string selecting one of the above biases.
    @param band       A string with the filter band for which to plot the chromatic bias
    @param cband      A tuple of two strings containing two bands to form a color.  Plot symbols will
                      be colored according to this color.
    @param corrected  Flag to plot photometric residuals instead of shift relative to G5v star
    @param kwargs     Other arguments to pass to the scatter plot
    """
    # Normalize to the centroid shift of a G5v star.
    table = cPickle.load(open(star_table))
    bias0 = table[table['star_type'] == 'ukg5v'][bias][band][0]

    f = plt.figure(figsize=(8, 6))
    # scatter plot
    ax = f.add_axes(scatter_axes_range)
    xlim = (-0.1, 3)
    ax.set_xlim(xlim)

    # fill in some data based on which chromatic bias is requested.
    if bias == 'Rbar':
        title = '$\Delta \overline{\mathrm{R}}$ (arcsec)'
        ax.fill_between(xlim, [-0.025]*2, [0.025]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-0.01]*2, [0.01]*2, color='#777777', zorder=2)
        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = (stars[bias][band] - bias0) * 180/np.pi * 3600
        galdata = (gals[bias][band] - bias0) * 180/np.pi * 3600
        ylim = set_range(np.concatenate([stardata, galdata]))
        # then replace with corrected measurements if requested
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) * 180/np.pi * 3600
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) * 180/np.pi * 3600
    elif bias == 'V':
        title = '$\Delta \mathrm{V}}$ (arcsec$^2$)'
        ax.fill_between(xlim, [-0.0006]*2, [0.0006]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-0.0001]*2, [0.0001]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) * (180/np.pi * 3600)**2
        galdata = (gals[bias][band] - bias0) * (180/np.pi * 3600)**2
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) * (180/np.pi * 3600)**2
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) * (180/np.pi * 3600)**2
    elif bias == 'S_m02':
        title = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'
        ax.fill_between(xlim, [-0.0025]*2, [0.0025]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-0.0004]*2, [0.0004]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) / bias0
        galdata = (gals[bias][band] - bias0) / bias0
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
    elif bias == 'S_p06':
        title = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'
        ax.fill_between(xlim, [-0.002]*2, [0.002]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) / bias0
        galdata = (gals[bias][band] - bias0) / bias0
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
    elif bias == 'S_p10':
        title = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'
        stardata = (stars[bias][band] - bias0) / bias0
        galdata = (gals[bias][band] - bias0) / bias0
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
    else:
        raise ValueError("Unknown chromatic bias in plot_bias")

    # colors for plotting symbols
    c = gals['magCalc'][cbands[0]] - gals['magCalc'][cbands[1]]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])
    im = ax.scatter(gals.redshift, galdata, c=c, vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)
    ax.set_xlabel('redshift', fontsize=12)
    if bias in ['Rbar', 'V']:
        ax.set_title('zenith angle = 45 degrees, filter = {}'.format(band), fontsize=12)
    else: # size bias is indep of zenith angle, so don't print it.
        ax.set_title('filter = {}'.format(band), fontsize=12)
    ax.yaxis.set_ticklabels([])
    ax.set_ylim(ylim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#AAAAAA', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)

    # star histogram
    hist_ax = f.add_axes(hist_axes_range)
    hist_with_peak(stardata, bins=200, range=ylim, orientation='horizontal',
                   histtype='stepfilled', color='blue')
    hist_ax.xaxis.set_ticklabels([])
    hist_ax.set_ylim(ylim)
    xlim = hist_ax.get_xlim()
    hist_ax.set_xlim(xlim)
    hist_ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='white', zorder=1)
    hist_ax.set_ylabel(title, fontsize=12)
    # gal histogram
    hist_with_peak(galdata, bins=200, range=ylim, orientation='horizontal',
                   histtype='step', color='red')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.08,
                 'stars', fontsize=12, color='blue')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.16,
                 'gals', fontsize=12, color='red')
    for label in hist_ax.get_yticklabels():
        label.set_fontsize(12)

    # colorbar
    cbar_ax = f.add_axes(colorbar_axes_range)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('{} - {}'.format(cbands[0].replace('LSST_',''), cbands[1].replace('LSST_','')),
                       fontsize=12)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(12)

    f.savefig(outfile, dpi=300)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = 'output/corrected_galaxy_data.pkl',
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = 'output/corrected_star_data.pkl',
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('--corrected', action='store_true',
                        help="plot learning residuals instead of G5v residuals.")
    parser.add_argument('bias', default="Rbar", nargs='?',
                        help="""which chromatic bias to plot (Default: 'Rbar')
                             Other possibilities include: 'V', 'S_m02', 'S_p06', 'S_p10'""")
    parser.add_argument('--band', default="LSST_r", nargs='?',
                        help="band of chromatic bias to plot (Default: 'LSST_r')")
    parser.add_argument('--color', default=['LSST_r', 'LSST_i'], nargs=2,
                        help="color to use for symbol color (Default: ['LSST_r', 'LSST_i'])")
    parser.add_argument('--outfile', default="output/chromatic_bias.png", nargs='?',
                        help="output filename (Default: 'output/chromatic_bias.png')")
    parser.add_argument('--nominal_plots', action='store_true',
                        help="Plot some nominal useful LSST and Euclid figures")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))

    if not args.nominal_plots:
        plot_bias(gals, stars, args.bias, args.band, args.color,
                  outfile=args.outfile, corrected=args.corrected, s=2)
    else:
        # LSST r-band
        plot_bias(gals, stars, 'Rbar', 'LSST_r', ('LSST_r', 'LSST_i'),
                  outfile='output/dRbar.LSST_r.png', s=2)
        plot_bias(gals, stars, 'Rbar', 'LSST_r', ('LSST_r', 'LSST_i'),
                  outfile='output/dRbar.corrected.LSST_r.png', corrected=True, s=2)

        plot_bias(gals, stars, 'V', 'LSST_r', ('LSST_r', 'LSST_i'),
                  outfile='output/dV.LSST_r.png', s=2)
        plot_bias(gals, stars, 'V', 'LSST_r', ('LSST_r', 'LSST_i'),
                  outfile='output/dV.corrected.LSST_r.png', corrected=True, s=2)

        plot_bias(gals, stars, 'S_m02', 'LSST_r', ('LSST_r', 'LSST_i'),
                  outfile='output/dS_m02.LSST_r.png', s=2)
        plot_bias(gals, stars, 'S_m02', 'LSST_r', ('LSST_r', 'LSST_i'),
                  outfile='output/dS_m02.corrected.LSST_r.png', corrected=True, s=2)

        # LSST i-band
        plot_bias(gals, stars, 'Rbar', 'LSST_i', ('LSST_r', 'LSST_i'),
                  outfile='output/dRbar.LSST_i.png', s=2)
        plot_bias(gals, stars, 'Rbar', 'LSST_i', ('LSST_r', 'LSST_i'),
                  outfile='output/dRbar.corrected.LSST_i.png', corrected=True, s=2)

        plot_bias(gals, stars, 'V', 'LSST_i', ('LSST_r', 'LSST_i'),
                  outfile='output/dV.LSST_i.png', s=2)
        plot_bias(gals, stars, 'V', 'LSST_i', ('LSST_r', 'LSST_i'),
                  outfile='output/dV.corrected.LSST_i.png', corrected=True, s=2)

        plot_bias(gals, stars, 'S_m02', 'LSST_i', ('LSST_r', 'LSST_i'),
                  outfile='output/dS_m02.LSST_i.png', s=2)
        plot_bias(gals, stars, 'S_m02', 'LSST_i', ('LSST_r', 'LSST_i'),
                  outfile='output/dS_m02.corrected.LSST_i.png', corrected=True, s=2)

        # Euclid 350nm band
        plot_bias(gals, stars, 'S_p06', 'Euclid_350', ('LSST_r', 'LSST_i'),
                  outfile='output/dS_p06.Euclid_350.png', s=2)
        plot_bias(gals, stars, 'S_p06', 'Euclid_350', ('LSST_r', 'LSST_i'),
                  outfile='output/dS_p06.corrected.Euclid_350.png', corrected=True, s=2)
