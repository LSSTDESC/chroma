"""Read pkl files created by process_*_catalog.py and *_ML.py and make plots of chromatic biases
as functions of redshift, both before and after photometric corrections are estimated.  Run
`python plot_bias.py --help` for a list of command line options.
"""

import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

# Specify the locations of three different plot elements.
hist_axes_range = [0.17, 0.12, 0.09, 0.7]
scatter_axes_range = [0.26, 0.12, 0.70, 0.7]
rms_axes_range = [0.26, 0.82, 0.70, 0.1]
colorbar_axes_range = [0.81, 0.15, 0.025, 0.35]
data_dir = '../../../data/'
star_table = '../../analytic/output/stars.pkl'
fontsize = 16

# hardcode some requirements, order is [DES, LSST]
m = np.r_[0.008, 0.003]
c = np.sqrt(2 * m * 4e-4) # 4e-4 is integrated shear power
r2gal = np.r_[0.51, 0.36]**2
r2psf = np.r_[0.8, 0.7]**2
epsf = 0.05 # is this a good assumption for LSST/DES?

dV_mean = m * r2gal
dV_rms = c * 2 * r2gal
dRbar_mean = np.sqrt(dV_mean)
dRbar_rms = np.sqrt(dV_rms)
dS_m02_mean = m * r2gal / r2psf
dS_m02_rms = m * r2gal / r2psf / epsf

m_Euclid = 0.001
r2gal_Euclid = 0.23**2
r2psf_Euclid = 0.2**2
epsf_Euclid = 0.1 # is this a good assumption for Euclid?
dS_p06_mean = m_Euclid * r2gal_Euclid / r2psf_Euclid
dS_p06_rms = m_Euclid * r2gal_Euclid / r2psf_Euclid / epsf_Euclid
dS_p10_mean = m_Euclid * r2gal_Euclid / r2psf_Euclid
dS_p10_rms = m_Euclid * r2gal_Euclid / r2psf_Euclid / epsf_Euclid

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
    xlim = (-0.1, 2.5)
    ax.set_xlim(xlim)

    # create rms axis so can draw requirement bars
    rms_ax = f.add_axes(rms_axes_range)
    # fill in some data based on which chromatic bias is requested.
    if bias == 'Rbar':
        ylabel = '$\Delta \overline{\mathrm{R}}$ (arcsec)'
        ax.fill_between(xlim, [-dRbar_mean[0]]*2, [dRbar_mean[0]]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-dRbar_mean[1]]*2, [dRbar_mean[1]]*2, color='#777777', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dRbar_rms[0]]*2, color='#999999', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dRbar_rms[1]]*2, color='#777777', zorder=2)
        # get *uncorrected* bias measurements in order to set ylimits, even if
        # corrected measurements are requested for plot.
        stardata = (stars[bias][band] - bias0) * 180/np.pi * 3600
        galdata = (gals[bias][band] - bias0) * 180/np.pi * 3600
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        ylim = set_range(np.concatenate([stardata, galdata]))
        # make sure to plot at least the entire LSST region
        if ylim[0] > -dRbar_mean[1]*1.2:
            ylim[0] = -dRbar_mean[1]*1.2
        if ylim[1] < dRbar_mean[1]*1.2:
            ylim[1] = dRbar_mean[1]*1.2
        # then replace with corrected measurements if requested
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) * 180/np.pi * 3600
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) * 180/np.pi * 3600
            norm = np.mean(stardata)
            stardata -= norm
            galdata -= norm
            ylabel = '$\delta(\Delta \overline{\mathrm{R}})$ (arcsec)'
    elif bias == 'V':
        ylabel = '$\Delta \mathrm{V}}$ (arcsec$^2$)'
        ax.fill_between(xlim, [-dV_mean[0]]*2, [dV_mean[0]]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-dV_mean[1]]*2, [dV_mean[1]]*2, color='#777777', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dV_rms[0]]*2, color='#999999', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dV_rms[1]]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) * (180/np.pi * 3600)**2
        galdata = (gals[bias][band] - bias0) * (180/np.pi * 3600)**2
        rms_bands = dV_rms
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        ylim = set_range(np.concatenate([stardata, galdata]))
        # make sure to plot at least the entire LSST region
        if ylim[0] > -dV_mean[1]*1.2:
            ylim[0] = -dV_mean[1]*1.2
        if ylim[1] < dV_mean[1]*1.2:
            ylim[1] = dV_mean[1]*1.2
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) * (180/np.pi * 3600)**2
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) * (180/np.pi * 3600)**2
            norm = np.mean(stardata)
            stardata -= norm
            galdata -= norm
            ylabel = '$\delta(\Delta \mathrm{V})$ (arcsec$^2$)'
    elif bias == 'S_m02':
        ylabel = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'
        ax.fill_between(xlim, [-dS_m02_mean[0]]*2, [dS_m02_mean[0]]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-dS_m02_mean[1]]*2, [dS_m02_mean[1]]*2, color='#777777', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dS_m02_rms[0]]*2, color='#999999', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dS_m02_rms[1]]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) / bias0
        galdata = (gals[bias][band] - bias0) / bias0
        rms_bands = dS_m02_rms
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        ylim = set_range(np.concatenate([stardata, galdata]))
        # make sure to plot at least the entire LSST region
        if ylim[0] > -dS_m02_mean[1]*1.2:
            ylim[0] = -dS_m02_mean[1]*1.2
        if ylim[1] < dS_m02_mean[1]*1.2:
            ylim[1] = dS_m02_mean[1]*1.2
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
            norm = np.mean(stardata)
            stardata -= norm
            galdata -= norm
            ylabel = '$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$'
    elif bias == 'S_p06':
        ylabel = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'
        ax.fill_between(xlim, [-dS_p06_mean]*2, [dS_p06_mean]*2, color='#777777', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dS_p06_rms]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) / bias0
        galdata = (gals[bias][band] - bias0) / bias0
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        ylim = set_range(np.concatenate([stardata, galdata]))
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
            norm = np.mean(stardata)
            stardata -= norm
            galdata -= norm
            ylabel = '$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$'
    elif bias == 'S_p10':
        ylabel = '$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$'
        ax.fill_between(xlim, [-dS_p10_mean]*2, [dS_p10_mean]*2, color='#777777', zorder=2)
        rms_ax.fill_between(xlim, [0]*2, [dS_p10_rms]*2, color='#777777', zorder=2)
        stardata = (stars[bias][band] - bias0) / bias0
        galdata = (gals[bias][band] - bias0) / bias0
        ylim = set_range(np.concatenate([stardata, galdata]))
        norm = np.mean(stardata)
        stardata -= norm
        galdata -= norm
        if corrected:
            stardata = (stars[bias][band] - stars['photo_'+bias][band]) / stars['photo_'+bias][band]
            galdata = (gals[bias][band] - gals['photo_'+bias][band]) / gals['photo_'+bias][band]
            norm = np.mean(stardata)
            stardata -= norm
            galdata -= norm
            ylabel = '$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$'
    else:
        raise ValueError("Unknown chromatic bias in plot_bias")

    # colors for plotting symbols
    c = gals['magCalc'][cbands[0]] - gals['magCalc'][cbands[1]]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])
    im = ax.scatter(gals.redshift, galdata, c=c, vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)
    ax.set_xlabel('redshift', fontsize=fontsize)

    # running mean and RMS:
    nbins = int(len(galdata)**0.4)
    xbins = np.linspace(0.0, np.max(gals.redshift), nbins+1)
    means = [np.mean(galdata[(gals.redshift > xbins[i]) & (gals.redshift < xbins[i+1])]) for i in range(nbins)]
    rmses = [np.std(galdata[(gals.redshift > xbins[i]) & (gals.redshift < xbins[i+1])]) for i in range(nbins)]
    zs = 0.5*(xbins[1:] + xbins[:-1])
    ax.plot(zs, means, color='red', linestyle='-', linewidth=2, zorder=10)

    # if bias in ['Rbar', 'V']:
    #     ax.set_title('zenith angle = 45 degrees, filter = {}'.format(band), fontsize=fontsize)
    # else: # size bias is indep of zenith angle, so don't print it.
    #     ax.set_title('filter = {}'.format(band), fontsize=fontsize)
    ax.yaxis.set_ticklabels([])
    ax.set_ylim(ylim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#AAAAAA', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)

    # star histogram
    hist_ax = f.add_axes(hist_axes_range)
    hist_with_peak(stardata, bins=200, range=ylim, orientation='horizontal',
                   histtype='stepfilled', color='blue')
    hist_ax.xaxis.set_ticklabels([])
    hist_ax.set_ylim(ylim)
    xlim = hist_ax.get_xlim()
    hist_ax.set_xlim(xlim)
    hist_ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='white', zorder=1)
    hist_ax.set_ylabel(ylabel, fontsize=fontsize)
    # gal histogram
    hist_with_peak(galdata, bins=200, range=ylim, orientation='horizontal',
                   histtype='step', color='red')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.08,
                 'stars', fontsize=fontsize, color='blue')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.16,
                 'gals', fontsize=fontsize, color='red')
    for label in hist_ax.get_yticklabels():
        label.set_fontsize(fontsize)

    # colorbar
    cbar_ax = f.add_axes(colorbar_axes_range)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('{} - {}'.format(cbands[0].replace('LSST_',''), cbands[1].replace('LSST_','')),
                       fontsize=fontsize)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(fontsize)

    # rms axis
    rms_ax.set_xlim(ax.get_xlim())
    rms_ax.xaxis.set_ticklabels([])
    rms_ax.set_ylabel('RMS')
    rms_ax.yaxis.set_ticklabels([])
    rms_ax.plot(zs, rmses, color='blue', linewidth=2)

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
    parser.add_argument('--nominal_plots', action='store_true',
                        help="Plot some nominal useful LSST and Euclid figures")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))

    if not args.nominal_plots:
        plot_bias(gals, stars, args.bias, args.band, args.color,
                  outfile=args.outfile, corrected=args.corrected, s=s)
    else:
        if args.corrected:
            # LSST r-band
            plot_bias(gals, stars, 'Rbar', 'LSST_r', ('LSST_r', 'LSST_i'),
                      outfile='output/dRbar_corrected_LSST_r.png', corrected=True, s=s)
            plot_bias(gals, stars, 'V', 'LSST_r', ('LSST_r', 'LSST_i'),
                      outfile='output/dV_corrected_LSST_r.png', corrected=True, s=s)
            plot_bias(gals, stars, 'S_m02', 'LSST_r', ('LSST_r', 'LSST_i'),
                      outfile='output/dS_m02_corrected_LSST_r.png', corrected=True, s=s)
            # LSST i-band
            plot_bias(gals, stars, 'Rbar', 'LSST_i', ('LSST_r', 'LSST_i'),
                      outfile='output/dRbar_corrected_LSST_i.png', corrected=True, s=s)
            plot_bias(gals, stars, 'V', 'LSST_i', ('LSST_r', 'LSST_i'),
                      outfile='output/dV_corrected_LSST_i.png', corrected=True, s=s)
            plot_bias(gals, stars, 'S_m02', 'LSST_i', ('LSST_r', 'LSST_i'),
                      outfile='output/dS_m02_corrected_LSST_i.png', corrected=True, s=s)
            # Euclid 350nm band
            plot_bias(gals, stars, 'S_p06', 'Euclid_350', ('LSST_r', 'LSST_i'),
                      outfile='output/dS_p06_corrected_Euclid_350.png', corrected=True, s=s)
        else:
            # LSST r-band
            plot_bias(gals, stars, 'Rbar', 'LSST_r', ('LSST_r', 'LSST_i'),
                      outfile='output/dRbar_LSST_r.png', s=s)
            plot_bias(gals, stars, 'V', 'LSST_r', ('LSST_r', 'LSST_i'),
                      outfile='output/dV_LSST_r.png', s=s)
            plot_bias(gals, stars, 'S_m02', 'LSST_r', ('LSST_r', 'LSST_i'),
                      outfile='output/dS_m02_LSST_r.png', s=s)
            # LSST i-band
            plot_bias(gals, stars, 'Rbar', 'LSST_i', ('LSST_r', 'LSST_i'),
                      outfile='output/dRbar_LSST_i.png', s=s)
            plot_bias(gals, stars, 'V', 'LSST_i', ('LSST_r', 'LSST_i'),
                      outfile='output/dV_LSST_i.png', s=s)
            plot_bias(gals, stars, 'S_m02', 'LSST_i', ('LSST_r', 'LSST_i'),
                      outfile='output/dS_m02_LSST_i.png', s=s)
            # Euclid 350nm band
            plot_bias(gals, stars, 'S_p06', 'Euclid_350', ('LSST_r', 'LSST_i'),
                      outfile='output/dS_p06_Euclid_350.png', s=s)