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


def plot_bias_vs_color(gals, stars, bias, band, cbands, outfile, corrected=False, **kwargs):
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

    @param gals       galaxy recarray produced by either process_gal_catalog.py or gal_ML.py
    @param stars      star recarray produced by either process_star_catalog.py or star_ML.py
    @param bias       A string selecting one of the above biases.
    @param band       A string with the filter band for which to plot the chromatic bias
    @param cband      A tuple of two strings containing two bands to form a color.  Plot symbols will
                      be colored according to this color.
    @param corrected  Flag to plot photometric residuals instead of shift relative to G5v star
    @param kwargs     Other arguments to pass to the scatter plot
    """

    f = plt.figure(figsize=(8, 6))
    # scatter plot
    ax = f.add_subplot(111)

    # x-axis: colors
    colordata = gals['mag'][cbands[0]] - gals['mag'][cbands[1]]

    # fill in some data based on which chromatic bias is requested.
    if bias == 'RbarSqr':
        ylabel = r"$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)"
        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = stars['Rbar'][band] * 180/np.pi * 3600
        galdata = gals['Rbar'][band] * 180/np.pi * 3600
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        stardata **= 2
        galdata **= 2
        ylim = set_range(np.concatenate([stardata, galdata]))
        ylim[0] = 0.0
        # then replace with corrected measurements if requested
        if corrected:
            stardata = (stars['Rbar'][band] - stars['photo_Rbar'][band]) * 180/np.pi * 3600
            galdata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
            # d((DR)^2) = 2 DR d(DR)
            stardata = 2 * (stars['Rbar'][band] * 180/np.pi * 3600 - norm) * stardata
            galdata = 2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * galdata
            ylabel = r'$\delta(\left(\Delta \overline{\mathrm{R}}\right)^2)$ (arcsec$^2$)'
    elif bias == 'LnRbarSqr':
        ylabel = r'$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)'
        ax.set_yscale('log')

        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = stars['Rbar'][band] * 180/np.pi * 3600
        galdata = gals['Rbar'][band] * 180/np.pi * 3600
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        stardata **= 2
        galdata **= 2
        ylim = set_range(np.concatenate([stardata, galdata]))
        # make sure to plot at least the entire LSST region
        if ylim[1] < mean_DeltaRbarSqr_req[1]*10:
            ylim[1] = mean_DeltaRbarSqr_req[1]*10
        ylim[0] = 1.e-7
        # then replace with corrected measurements if requested
        if corrected:
            stardata = (stars['Rbar'][band] - stars['photo_Rbar'][band]) * 180/np.pi * 3600
            galdata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
            # d((DR)^2) = 2 DR d(DR)
            stardata = np.abs(2 * (stars['Rbar'][band] * 180/np.pi * 3600 - norm) * stardata)
            galdata = np.abs(2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * galdata)
            ylabel = r'$|\delta((\Delta \overline{\mathrm{R}})^2)|$ (arcsec$^2$)'
    elif bias == 'V':
        ylabel = '$\Delta \mathrm{V}}$ (arcsec$^2$)'

        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = stars[bias][band] * (180/np.pi * 3600)**2
        galdata = gals[bias][band] * (180/np.pi * 3600)**2
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        ylim = set_range(np.concatenate([stardata, galdata]))
        # make sure to plot at least the entire LSST region
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) * (180/np.pi * 3600)**2
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) * (180/np.pi * 3600)**2
            ylabel = '$\delta(\Delta \mathrm{V})$ (arcsec$^2$)'
    elif bias == 'S_m02':
        ylabel = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'

        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = stars[bias][band]
        galdata = gals[bias][band]
        starmean = np.mean(stardata)
        stardata = (stardata - starmean)/starmean
        galdata = (galdata - starmean)/starmean
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
            ylabel = '$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$'
    elif bias == 'S_p06':
        ylabel = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'

        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = stars[bias][band]
        galdata = gals[bias][band]
        starmean = np.mean(stardata)
        stardata = (stardata - starmean)/starmean
        galdata = (galdata - starmean)/starmean
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
            ylabel = '$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$'
    elif bias == 'S_p10':
        ylabel = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'

        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = stars[bias][band]
        galdata = gals[bias][band]
        starmean = np.mean(stardata)
        stardata = (stardata - starmean)/starmean
        galdata = (galdata - starmean)/starmean
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
            ylabel = '$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$'
    else:
        raise ValueError("Unknown chromatic bias in plot_bias_vs_color")

    band_dict = {'LSST_r':"r",
                 'LSST_i':"i"}

    im = ax.scatter(colordata, galdata, zorder=4, **kwargs)
    ax.set_xlabel(r"${} - {}$".format(band_dict[cbands[0]], band_dict[cbands[1]]),
                  fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_ylim(ylim)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)

    # label which band we're dealing with
    ax.text(0.83, 0.93, band.replace('LSST_','')+' band', transform=ax.transAxes,
            fontsize=fontsize)

    f.tight_layout()
    f.savefig(outfile, dpi=220)

if __name__ == '__main__':
    s=3
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
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))

    plot_bias_vs_color(gals, stars, args.bias, args.band, args.color,
                       outfile=args.outfile, corrected=args.corrected, s=s)
