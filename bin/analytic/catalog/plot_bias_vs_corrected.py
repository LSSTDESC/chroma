"""Read pkl files created by process_*_catalog.py and *_ML.py and make plots of chromatic biases
as functions of color, both before and after photometric corrections are estimated.  Run
`python plot_bias_vs_color.py --help` for a list of command line options.
"""

import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fontsize = 16

def set_range(x):
    """ Return a plotting range 30% larger than the interval containing 99% of the data.
    """
    xs = sorted(x)
    n = len(xs)
    low = xs[int(0.005*n)]
    high = xs[int(0.995*n)]
    span = high-low
    return [low - 0.3*span, high + 0.3*span]


def plot_bias_vs_corrected(gals, stars, bias, band, outfile, cbands=None, **kwargs):
    """Produce a plot of a chromatic bias, which is one of:
       `RbarSqr` - Square of centroid shift due to differential chromatic refraction
       `LnRbarSqr` - Use log plot for square of centroid shift due to differential chromatic
                     refraction
       `V`     - second moment shift due to differential chromatic refraction
       `S_m02` - difference in r^2 due to FWHM \propto \lambda^{-0.2}, as per chromatic seeing
       `S_p06` - difference in r^2 due to FWHM \propto \lambda^{+0.6}, as per Euclid
       `S_p10` - difference in r^2 due to FWHM \propto \lambda^{+1.0}, as per diffraction limit
    Either plot the uncorrected shift relative to the shift of a G5v star, or plot the residual shift
    from the photometric estimate of the shift derived from machine learning.

    @param gals       galaxy recarray produced by gal_ML.py
    @param stars      star recarray produced by star_ML.py
    @param bias       A string selecting one of the above biases.
    @param band       A string with the filter band for which to plot the chromatic bias
    @param cband      A tuple of two strings containing two bands to form a color.  If present, plot
                      symbols will be colored according to this color.  If absent, plot symbols will
                      be colored by redshift.
    @param kwargs     Other arguments to pass to the scatter plot
    """

    f = plt.figure(figsize=(8,6))
    # scatter plot
    ax = f.add_subplot(111)

    # plot symbol color
    if cbands is not None:
        cdata = gals['mag'][cbands[0]] - gals['mag'][cbands[1]]
    else:
        cdata = gals['redshift']

    # fill in some data based on which chromatic bias is requested.
    if bias == 'RbarSqr':
        xlabel = r"$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)"
        ylabel = r"$\delta(\left(\Delta \overline{\mathrm{R}}\right)^2)$ (arcsec$^2$)"

        stardata = stars['Rbar'][band] * 180/np.pi * 3600
        galdata = gals['Rbar'][band] * 180/np.pi * 3600
        norm = np.mean(stardata)
        galdata -= norm
        galdata **= 2
        xdata = galdata

        cgaldata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
        # d((DR)^2) = 2 DR d(DR)
        cgaldata = 2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * cgaldata
        ydata = cgaldata
        xlim = set_range(xdata)
        xlim[0] = 0.0
        ylim = set_range(ydata)
        ylim[0] = 0.0
    elif bias == 'LnRbarSqr':
        xlabel = r"$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)"
        ylabel = r"$|\delta((\Delta \overline{\mathrm{R}})^2)|$ (arcsec$^2$)"
        ax.set_xscale('log')
        ax.set_yscale('log')

        stardata = stars['Rbar'][band] * 180/np.pi * 3600
        galdata = gals['Rbar'][band] * 180/np.pi * 3600
        norm = np.mean(stardata)
        galdata -= norm
        galdata **= 2
        xdata = galdata
        cgaldata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
        # d((DR)^2) = 2 DR d(DR)
        cgaldata = np.abs(2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * cgaldata)
        ydata = cgaldata
        # # make sure to plot at least the entire LSST region
        # if ylim[1] < mean_DeltaRbarSqr_req[1]*10:
        #     ylim[1] = mean_DeltaRbarSqr_req[1]*10
        # ylim[0] = 1.e-7
        # then replace with corrected measurements if requested
        xlim = set_range(xdata)
        xlim[0] = 1.e-7
        ylim = set_range(ydata)
        ylim[0] = 1.e-7
    elif bias == 'V':
        xlabel = r"$\Delta \mathrm{V}}$ (arcsec$^2$)"
        ylabel = "$\delta(\Delta \mathrm{V})$ (arcsec$^2$)"

        stardata = stars[bias][band] * (180/np.pi * 3600)**2
        galdata = gals[bias][band] * (180/np.pi * 3600)**2
        norm = np.mean(stardata)
        galdata -= norm
        xdata = galdata

        cgaldata = (gals[bias][band] - gals['photo_'+bias][band]) * (180/np.pi * 3600)**2
        ydata = cgaldata

        xlim = set_range(xdata)
        ylim = set_range(ydata)
    elif bias == 'S_m02':
        xlabel = r"$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$"
        ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"

        stardata = stars[bias][band]
        galdata = gals[bias][band]
        starmean = np.mean(stardata)
        galdata = (galdata - starmean)/starmean
        xdata = galdata

        cgaldata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
        ydata = cgaldata

        xlim = set_range(xdata)
        ylim = set_range(ydata)
    # elif bias == 'S_p06':
    #     ylabel = "$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$"

    #     # get *uncorrected* bias measurements in order to set ylimits, even if
    #     # corrected measurements are requested for plot.
    #     stardata = stars[bias][band]
    #     galdata = gals[bias][band]
    #     starmean = np.mean(stardata)
    #     stardata = (stardata - starmean)/starmean
    #     galdata = (galdata - starmean)/starmean
    #     ylim = set_range(np.concatenate([stardata, galdata]))
    #     if corrected:
    #         stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
    #         galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
    #         ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"
    # elif bias == 'S_p10':
    #     ylabel = "$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$"

    #     # get *uncorrected* bias measurements in order to set ylimits, even if
    #     # corrected measurements are requested for plot.
    #     stardata = stars[bias][band]
    #     galdata = gals[bias][band]
    #     starmean = np.mean(stardata)
    #     stardata = (stardata - starmean)/starmean
    #     galdata = (galdata - starmean)/starmean
    #     ylim = set_range(np.concatenate([stardata, galdata]))
    #     if corrected:
    #         stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
    #         galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
    #         ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"
    else:
        raise ValueError("Unknown chromatic bias in plot_bias_vs_color")

    band_dict = {'LSST_r':"r",
                 'LSST_i':"i"}

    ax.scatter(xdata, ydata, c=cdata, zorder=4, **kwargs)
    ax.plot([-1e8, 1e8], [-1e8, 1e8], c='k')
    ax.fill_between(xlim, 2*[ylim[0]], 2*[ylim[1]], facecolor="#AAAAAA")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # # label which band we're dealing with
    # ax.text(0.83, 0.93, band.replace('LSST_','')+' band', transform=ax.transAxes,
    #         fontsize=fontsize)

    f.tight_layout()
    f.savefig(outfile, dpi=220)

if __name__ == '__main__':
    s=3
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = "output/corrected_galaxy_data.pkl",
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = "output/corrected_star_data.pkl",
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('bias', default='RbarSqr', nargs='?',
                        help="""which chromatic bias to plot (Default: 'RbarSqr')
                             Other possibilities include: 'V', 'S_m02', 'S_p06', 'S_p10'""")
    parser.add_argument('--band', default='LSST_r', nargs='?',
                        help="band of chromatic bias to plot (Default: 'LSST_r')")
    parser.add_argument('--color', default=None, nargs=2,
                        help="color to use for symbol color (Default: None)")
    parser.add_argument('--outfile', default="output/bias_vs_corrected.png", nargs='?',
                        help="output filename (Default: 'output/bias_vs_corrected.png')")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))

    plot_bias_vs_corrected(gals, stars, args.bias, args.band, args.outfile,
                           cbands=args.color, s=s)
