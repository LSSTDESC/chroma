"""Read pkl files created by process_*_catalog.py and *_ML.py and make plots of chromatic biases
as functions of redshift, both before and after photometric corrections are estimated.  Run
`python plot_bias.py --help` for a list of command line options.
"""

import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fontsize = 8

# Some annotation arrow properties
arrowdict = dict(facecolor='black', shrink=0.1, width=1.5, headwidth=4, frac=0.2)

# hardcode some requirements, order is [DES, LSST]
r2sqr_gal = np.r_[0.4, 0.3]**2
r2sqr_PSF = np.r_[0.8, 0.7]**2

mean_m_req = np.r_[0.008, 0.003]
# C variance sufficient
var_c_sufficient = 1.8e-7 * np.r_[(5000/18000.)**(-0.5) * (12./30)**(-0.5) * (0.68/0.82)**(-0.6),
                                  1.0]

mean_DeltaRbarSqr_req = mean_m_req / 2.0
var_DeltaRbarSqr_sufficient = var_c_sufficient / 1.0**2

mean_DeltaV_req = r2sqr_gal * mean_m_req
var_DeltaV_sufficient = var_c_sufficient * 4 * r2sqr_gal**2 * 0.5
# last factor of 0.5 needed to account for possible rotation of DeltaV from being purely real.
# see appendix of Meyers+Burchat15.

mean_dS_m02_req = mean_m_req * r2sqr_gal / r2sqr_PSF
epsf = 0.05
var_dS_m02_sufficient = var_c_sufficient / (epsf / 2.0 * r2sqr_PSF / r2sqr_gal)**2 * 0.5
# last factor of 0.5 needed to account for possible rotation of DeltaV from being purely real.
# see appendix of Meyers+Burchat15.

m_Euclid = 0.001
r2gal_Euclid = 0.23**2 # where did I get this from?
r2psf_Euclid = 0.2**2
mean_dS_p06_req = m_Euclid * r2gal_Euclid / r2psf_Euclid
mean_dS_p10_req = m_Euclid * r2gal_Euclid / r2psf_Euclid

print
print
print "DES reqs"
print "<m>: {}".format(mean_m_req[0])
print "<dRbarSqr>: {}".format(mean_DeltaRbarSqr_req[0])
print "<dV>: {}".format(mean_DeltaV_req[0])
print "<dS>: {}".format(mean_dS_m02_req[0])
print "var(c): {}".format(var_c_sufficient[0])
print "var(dRbarSqr): {}".format(var_DeltaRbarSqr_sufficient[0])
print "var(dV): {}".format(var_DeltaV_sufficient[0])
print "var(dS): {}".format(var_dS_m02_sufficient[0])
print
print "LSST reqs"
print "<m>: {}".format(mean_m_req[1])
print "<dRbarSqr>: {}".format(mean_DeltaRbarSqr_req[1])
print "<dV>: {}".format(mean_DeltaV_req[1])
print "<dS>: {}".format(mean_dS_m02_req[1])
print "var(c): {}".format(var_c_sufficient[1])
print "var(dRbarSqr): {}".format(var_DeltaRbarSqr_sufficient[1])
print "var(dV): {}".format(var_DeltaV_sufficient[1])
print "var(dS): {}".format(var_dS_m02_sufficient[1])
print

def hist_with_peak(x, bins=None, range=None, ax=None, orientation='vertical',
                   histtype=None, log=False, **kwargs):
    """Plot a histogram normalized to unit peak.
    """
    if ax is None:
        ax = plt.gca()
    if log:
        x = np.log(x)
        range = [np.log(r) for r in range]
    hist, bin_edges = np.histogram(x, bins=bins, range=range)
    if log:
        bin_edges = [np.exp(b) for b in bin_edges]
        width = bin_edges
    else:
        width = bin_edges[1] - bin_edges[0]
    hist_n = hist * 1.0/hist.max()
    x = np.ravel(zip(bin_edges[:-1], bin_edges[1:]))
    y = np.ravel(zip(hist_n, hist_n))
    x = np.concatenate([[x[0]],x])
    y = np.concatenate([[0],y])
    if histtype == 'step':
        if orientation == 'vertical':
            ax.plot(x, y, **kwargs)
        elif orientation == 'horizontal':
            ax.plot(y, x, **kwargs)
        else:
            raise ValueError
    elif histtype == 'stepfilled':
        if orientation == 'vertical':
            ax.fill(x, y, **kwargs)
        elif orientation == 'horizontal':
            ax.fill(y, x, **kwargs)
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

def panel_getaxes(fig, grid):
    inner_grid = gridspec.GridSpecFromSubplotSpec(100, 100, subplot_spec=grid,
                                                  wspace=0.0, hspace=0.0)

    var_ax = plt.Subplot(fig, inner_grid[:19, 11:])
    fig.add_subplot(var_ax)

    scatter_ax = plt.Subplot(fig, inner_grid[19:, 11:])
    fig.add_subplot(scatter_ax)

    hist_ax = plt.Subplot(fig, inner_grid[19:, :11])
    fig.add_subplot(hist_ax)

    cbar_ax = plt.Subplot(fig, inner_grid[55:95, 80:83])
    fig.add_subplot(cbar_ax)

    return var_ax, scatter_ax, hist_ax, cbar_ax

def setup_scatter_panel(ax, xlim, ylim, log=False):
    # Setup scatter plot limits and text properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if log:
        ax.set_yscale('log')
    ax.set_xlabel("redshift", fontsize=fontsize)
    # clean up tick labels
    ax.yaxis.set_ticklabels([])
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)

def setup_variance_panel(ax, xlim, ylim):
    # Setup variance plot limits and text properties
    # variance axis
    ax.set_xlim(xlim)
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("$\sqrt{\mathrm{Var}}$", fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB', zorder=1)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
    ax.locator_params(axis='y', nbins=4, prune='lower')

def setup_histogram_panel(ax, xlim, ylim, ylabel, log=False):
    ax.set_ylim(ylim)
    hist_xlim = [0.0, 1.0]
    ax.set_xlim(hist_xlim)
    if log:
        ax.set_yscale('log')
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel(ylabel, fontsize=fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)

def fill_requirements(xlim, yreq, ax):
    ax.fill_between(xlim, [-yreq[0]]*2, [yreq[0]]*2, color='#999999', zorder=2)
    ax.fill_between(xlim, [-yreq[1]]*2, [yreq[1]]*2, color='#777777', zorder=2)
    ax.axhline(-yreq[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(-yreq[1], c='k', alpha=0.3, zorder=10, lw=0.5)
    ax.axhline(yreq[0], c='k', alpha=0.1, zorder=10, lw=0.5)
    ax.axhline(yreq[1], c='k', alpha=0.3, zorder=10, lw=0.5)

def RbarSqr_process(stars, gals, band, log=False, corrected=False):
    # get *uncorrected* bias measurements in order to set ylimits, even if
    # corrected measurements are requested for plot.
    stardata = stars['Rbar'][band] * 180/np.pi * 3600
    galdata = gals['Rbar'][band] * 180/np.pi * 3600
    norm = np.mean(stardata) # normalize by the mean stellar data
    stardata -= norm
    galdata -= norm
    stardata **= 2
    galdata **= 2
    ylim = set_range(np.concatenate([stardata, galdata]))
    # make sure to plot at least the entire LSST region
    if log:
        if ylim[1] < mean_DeltaRbarSqr_req[1]*10:
            ylim[1] = mean_DeltaRbarSqr_req[1]*10
        ylim[0] = 1.e-7
    else:
        if ylim[1] < mean_DeltaRbarSqr_req[1]*1.2:
            ylim[1] = mean_DeltaRbarSqr_req[1]*1.2
        ylim[0] = 0.0
    # same concept for variance axis: set range based on uncorrected data
    nbins = int(len(galdata)**0.4)
    xbins = np.linspace(0.0, np.max(gals.redshift), nbins+1)
    zs = 0.5*(xbins[1:] + xbins[:-1])
    sqrt_vars = [np.sqrt(np.var(galdata[(gals.redshift > xbins[i])
                                        & (gals.redshift < xbins[i+1])]))
                 for i in range(nbins)]
    var_ylim = [0, np.max(sqrt_vars)*1.2]
    if var_ylim[1] < np.sqrt(var_DeltaRbarSqr_sufficient[1])*1.2:
        var_ylim[1] = np.sqrt(var_DeltaRbarSqr_sufficient[1])*1.2
    # then replace with corrected measurements if requested
    if corrected:
        stardata = (stars['Rbar'][band] - stars['photo_Rbar'][band]) * 180/np.pi * 3600
        ungaldata = galdata
        galdata = (gals['Rbar'][band] - gals['photo_Rbar'][band]) * 180/np.pi * 3600
        # d((DR)^2) = 2 DR d(DR)
        stardata = np.abs(2 * (stars['Rbar'][band] * 180/np.pi * 3600 - norm) * stardata)
        galdata = np.abs(2 * (gals['Rbar'][band] * 180/np.pi * 3600 - norm) * galdata)
        # running variance
        sqrt_vars = [np.sqrt(np.var(galdata[(gals.redshift > xbins[i])
                                            & (gals.redshift < xbins[i+1])]))
                     for i in range(nbins)]
    # running mean
    means = [np.mean(galdata[(gals.redshift > xbins[i])
                             & (gals.redshift < xbins[i+1])])
             for i in range(nbins)]
    return stardata, galdata, ylim, var_ylim, zs, means, sqrt_vars

def RbarSqr_panel(gals, stars, band, cbands, fig, grid, log=False, corrected=False, **kwargs):
    xlim = (0.0, 2.5) # redshift range

    # Process the data
    process = RbarSqr_process(stars, gals, band, log=log, corrected=corrected)
    stardata, galdata, ylim, var_ylim, zs, means, sqrt_vars = process

    if corrected:
        ylabel = r"$|\delta((\Delta \overline{\mathrm{R}})^2)|$ (arcsec$^2$)"
    else:
        ylabel = r"$\left(\Delta \overline{\mathrm{R}}\right)^2$ (arcsec$^2$)"

    # Get axes
    var_ax, scatter_ax, hist_ax, cbar_ax = panel_getaxes(fig, grid)

    # get colors and color range to store for later
    c=gals['magCalc'][cbands[0]] - gals['magCalc'][cbands[1]]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])

    # Scatter plot
    setup_scatter_panel(scatter_ax, xlim, ylim, log=log)
    rand_order = np.random.shuffle(np.arange(len(gals.redshift)))
    im = scatter_ax.scatter(gals.redshift[rand_order], galdata[rand_order], c=c[rand_order],
                            vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)

    scatter_ax.plot(zs, means, color='red', linestyle='-', linewidth=2, zorder=10)
    fill_requirements(xlim, mean_DeltaRbarSqr_req, scatter_ax)

    # annotate scatter plot
    for i, text in enumerate(["DES", "LSST"]):
        if log:
            xytext = (0.18, mean_DeltaRbarSqr_req[i]/2.1)
        else:
            xytext = (0.18, mean_DeltaRbarSqr_req[i]-0.0001)
        scatter_ax.annotate(text+" requirement",
                            xy=(0.1, mean_DeltaRbarSqr_req[i]),
                            xytext=xytext,
                            arrowprops=arrowdict,
                            zorder=10,
                            fontsize=fontsize)
    scatter_ax.text(0.83, 0.93, band.replace('LSST_','')+' band', transform=scatter_ax.transAxes,
                    fontsize=fontsize)

    # Variance plot
    setup_variance_panel(var_ax, xlim, var_ylim)
    fill_requirements(xlim, np.sqrt(var_DeltaRbarSqr_sufficient), var_ax)
    var_ax.plot(zs, sqrt_vars, color='blue', linewidth=2)

    # Histogram plot
    hist_xlim = [0.0, 1.0]
    setup_histogram_panel(hist_ax, hist_xlim, ylim, ylabel, log=log)
    # plot this histograms
    hist_with_peak(stardata, bins=200, ax=hist_ax, range=ylim, orientation='horizontal',
                   histtype='stepfilled', log=log, color='blue')
    hist_with_peak(galdata, bins=200, ax=hist_ax, range=ylim, orientation='horizontal',
                   histtype='step', log=log, color='red')

    # annotate histogram plot
    hist_ax.text(0.1, 0.93,
                 "stars", fontsize=fontsize, color='blue', transform=hist_ax.transAxes)
    hist_ax.text(0.1, 0.88,
                 "gals", fontsize=fontsize, color='red', transform=hist_ax.transAxes)

    # colorbar
    cbar = plt.colorbar(im, cax=cbar_ax)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(fontsize)
    cbar_ax.set_ylabel("{} - {}".format(cbands[0].replace('LSST_',''),
                                        cbands[1].replace('LSST_','')),
                       fontsize=fontsize)

def V_process(stars, gals, band, corrected=False):
    # get *uncorrected* bias measurements in order to set ylimits, even if
    # corrected measurements are requested for plot.
    stardata = stars['V'][band] * (180/np.pi * 3600)**2
    galdata = gals['V'][band] * (180/np.pi * 3600)**2
    norm = np.mean(stardata)
    stardata -= norm
    galdata -= norm
    ylim = set_range(np.concatenate([stardata, galdata]))
    # make sure to plot at least the entire LSST region
    if ylim[0] > -mean_DeltaV_req[1]*1.2:
        ylim[0] = -mean_DeltaV_req[1]*1.2
    if ylim[1] < mean_DeltaV_req[1]*1.2:
        ylim[1] = mean_DeltaV_req[1]*1.2
    # same concept for variance axis: set range based on uncorrected data
    nbins = int(len(galdata)**0.4)
    xbins = np.linspace(0.0, np.max(gals.redshift), nbins+1)
    zs = 0.5*(xbins[1:] + xbins[:-1])
    sqrt_vars = [np.sqrt(np.var(galdata[(gals.redshift > xbins[i])
                                        & (gals.redshift < xbins[i+1])]))
                 for i in range(nbins)]
    var_ylim = [0, np.max(sqrt_vars)*1.2]
    if var_ylim[1] < np.sqrt(var_DeltaV_sufficient[1])*1.2:
        var_ylim[1] = np.sqrt(var_DeltaV_sufficient[1])*1.2
    # then replace with corrected measurements if requested
    if corrected:
        stardata = (stars['V'][band] - stars['photo_V'][band]) * (180/np.pi * 3600)**2
        ungaldata = galdata
        galdata = (gals['V'][band] - gals['photo_V'][band]) * (180/np.pi * 3600)**2
        # running variance
        sqrt_vars = [np.sqrt(np.var(galdata[(gals.redshift > xbins[i])
                                            & (gals.redshift < xbins[i+1])]))
                     for i in range(nbins)]
    # running mean
    means = [np.mean(galdata[(gals.redshift > xbins[i])
                             & (gals.redshift < xbins[i+1])])
             for i in range(nbins)]
    return stardata, galdata, ylim, var_ylim, zs, means, sqrt_vars

def V_panel(gals, stars, band, cbands, fig, grid, corrected=False, **kwargs):
    xlim = (0.0, 2.5) # redshift range

    # Process the data
    process = V_process(stars, gals, band, corrected=corrected)
    stardata, galdata, ylim, var_ylim, zs, means, sqrt_vars = process

    if corrected:
        ylabel = r"$\delta(\Delta V)$ (arcsec$^2$)"
    else:
        ylabel = r"$\Delta V$ (arcsec$^2$)"

    # Get axes
    var_ax, scatter_ax, hist_ax, cbar_ax = panel_getaxes(fig, grid)

    # get colors and color range to store for later
    c=gals['magCalc'][cbands[0]] - gals['magCalc'][cbands[1]]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])

    # Scatter plot
    setup_scatter_panel(scatter_ax, xlim, ylim)
    rand_order = np.random.shuffle(np.arange(len(gals.redshift)))
    im = scatter_ax.scatter(gals.redshift[rand_order], galdata[rand_order], c=c[rand_order],
                            vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)

    scatter_ax.plot(zs, means, color='red', linestyle='-', linewidth=2, zorder=10)
    fill_requirements(xlim, mean_DeltaV_req, scatter_ax)

    # annotate scatter plot
    if band == 'LSST_i':
        scatter_ax.annotate("LSST requirement",
                            xy=(0.1, mean_DeltaV_req[1]),
                            xytext=(0.18, mean_DeltaV_req[1]-5.e-5),
                            arrowprops=arrowdict,
                            zorder=10,
                            fontsize=fontsize)
    else:
        scatter_ax.annotate("LSST requirement",
                            xy=(0.1, mean_DeltaV_req[1]),
                            xytext=(0.18, mean_DeltaV_req[1]+2.e-4),
                            arrowprops=arrowdict,
                            zorder=10,
                            fontsize=fontsize)
        scatter_ax.annotate("DES requirement",
                            xy=(0.1, mean_DeltaV_req[0]),
                            xytext=(0.18, mean_DeltaV_req[0]-2.e-4),
                            arrowprops=arrowdict,
                            zorder=10,
                            fontsize=fontsize)
    scatter_ax.text(0.83, 0.93, band.replace('LSST_','')+' band', transform=scatter_ax.transAxes,
                    fontsize=fontsize)

    # Variance plot
    setup_variance_panel(var_ax, xlim, var_ylim)
    fill_requirements(xlim, np.sqrt(var_DeltaV_sufficient), var_ax)
    var_ax.plot(zs, sqrt_vars, color='blue', linewidth=2)

    # Histogram plot
    hist_xlim = [0.0, 1.0]
    setup_histogram_panel(hist_ax, hist_xlim, ylim, ylabel)
    # plot this histograms
    hist_with_peak(stardata, bins=200, ax=hist_ax, range=ylim, orientation='horizontal',
                   histtype='stepfilled', color='blue')
    hist_with_peak(galdata, bins=200, ax=hist_ax, range=ylim, orientation='horizontal',
                   histtype='step', color='red')

    # annotate histogram plot
    hist_ax.text(0.1, 0.93,
                 "stars", fontsize=fontsize, color='blue', transform=hist_ax.transAxes)
    hist_ax.text(0.1, 0.88,
                 "gals", fontsize=fontsize, color='red', transform=hist_ax.transAxes)

    # colorbar
    cbar = plt.colorbar(im, cax=cbar_ax)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(fontsize)
    cbar_ax.set_ylabel("{} - {}".format(cbands[0].replace('LSST_',''),
                                        cbands[1].replace('LSST_','')),
                       fontsize=fontsize)


def S_m02_process(stars, gals, band, corrected=False):
    # get *uncorrected* bias measurements in order to set ylimits, even if
    # corrected measurements are requested for plot.
    stardata = stars['S_m02'][band]
    galdata = gals['S_m02'][band]
    starmean = np.mean(stardata)
    stardata = (stardata - starmean)/starmean
    galdata = (galdata - starmean)/starmean
    ylim = set_range(np.concatenate([stardata, galdata]))
    # make sure to plot at least the entire LSST region
    if ylim[0] > -mean_dS_m02_req[1]*1.2:
        ylim[0] = -mean_dS_m02_req[1]*1.2
    if ylim[1] < mean_dS_m02_req[1]*1.2:
        ylim[1] = mean_dS_m02_req[1]*1.2
    # same concept for variance axis: set range based on uncorrected data
    nbins = int(len(galdata)**0.4)
    xbins = np.linspace(0.0, np.max(gals.redshift), nbins+1)
    zs = 0.5*(xbins[1:] + xbins[:-1])
    sqrt_vars = [np.sqrt(np.var(galdata[(gals.redshift > xbins[i])
                                        & (gals.redshift < xbins[i+1])]))
                 for i in range(nbins)]
    var_ylim = [0, np.max(sqrt_vars)*1.2]
    if var_ylim[1] < np.sqrt(var_dS_m02_sufficient[1])*1.2:
        var_ylim[1] = np.sqrt(var_dS_m02_sufficient[1])*1.2
    # then replace with corrected measurements if requested
    if corrected:
        stardata = (stars['S_m02'][band] - stars['photo_S_m02'][band]) / stars['photo_S_m02'][band]
        ungaldata = galdata
        galdata = (gals['S_m02'][band] - gals['photo_S_m02'][band]) / gals['photo_S_m02'][band]
        ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"
        # running variance
        sqrt_vars = [np.sqrt(np.var(galdata[(gals.redshift > xbins[i])
                                            & (gals.redshift < xbins[i+1])]))
                     for i in range(nbins)]
    # running mean
    means = [np.mean(galdata[(gals.redshift > xbins[i])
                             & (gals.redshift < xbins[i+1])])
             for i in range(nbins)]
    return stardata, galdata, ylim, var_ylim, zs, means, sqrt_vars

def S_m02_panel(gals, stars, band, cbands, fig, grid, corrected=False, **kwargs):
    xlim = (0.0, 2.5) # redshift range

    # Process the data
    process = S_m02_process(stars, gals, band, corrected=corrected)
    stardata, galdata, ylim, var_ylim, zs, means, sqrt_vars = process

    if corrected:
        ylabel = "$\delta(\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF})$"
    else:
        ylabel = "$\Delta r^2_\mathrm{PSF}/r^2_\mathrm{PSF}$"

    # Get axes
    var_ax, scatter_ax, hist_ax, cbar_ax = panel_getaxes(fig, grid)

    # get colors and color range to store for later
    c=gals['magCalc'][cbands[0]] - gals['magCalc'][cbands[1]]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])

    # Scatter plot
    setup_scatter_panel(scatter_ax, xlim, ylim)
    rand_order = np.random.shuffle(np.arange(len(gals.redshift)))
    im = scatter_ax.scatter(gals.redshift[rand_order], galdata[rand_order], c=c[rand_order],
                            vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)

    scatter_ax.plot(zs, means, color='red', linestyle='-', linewidth=2, zorder=10)
    fill_requirements(xlim, mean_dS_m02_req, scatter_ax)

    # annotate scatter plot
    for i, text in enumerate(["DES", "LSST"]):
        scatter_ax.annotate(text+" requirement",
                            xy=(0.1, mean_dS_m02_req[i]),
                            xytext=(0.18, mean_dS_m02_req[i]+2.e-3),
                            arrowprops=arrowdict,
                            zorder=10,
                            fontsize=fontsize)
    scatter_ax.text(0.83, 0.93, band.replace('LSST_','')+' band', transform=scatter_ax.transAxes,
                    fontsize=fontsize)

    # Variance plot
    setup_variance_panel(var_ax, xlim, var_ylim)
    fill_requirements(xlim, np.sqrt(var_dS_m02_sufficient), var_ax)
    var_ax.plot(zs, sqrt_vars, color='blue', linewidth=2)

    # Histogram plot
    hist_xlim = [0.0, 1.0]
    setup_histogram_panel(hist_ax, hist_xlim, ylim, ylabel)
    # plot this histograms
    hist_with_peak(stardata, bins=200, ax=hist_ax, range=ylim, orientation='horizontal',
                   histtype='stepfilled', color='blue')
    hist_with_peak(galdata, bins=200, ax=hist_ax, range=ylim, orientation='horizontal',
                   histtype='step', color='red')

    # annotate histogram plot
    hist_ax.text(0.1, 0.93,
                 "stars", fontsize=fontsize, color='blue', transform=hist_ax.transAxes)
    hist_ax.text(0.1, 0.88,
                 "gals", fontsize=fontsize, color='red', transform=hist_ax.transAxes)

    # colorbar
    cbar = plt.colorbar(im, cax=cbar_ax)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(fontsize)
    cbar_ax.set_ylabel("{} - {}".format(cbands[0].replace('LSST_',''),
                                        cbands[1].replace('LSST_','')),
                       fontsize=fontsize)


def plot_bias_panel(args, **kwargs):

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))

    fig = plt.figure(figsize=(9, 10))
    outer_grid = gridspec.GridSpec(len(args.bias), len(args.band),
                                   left=0.1, right=0.95,
                                   top = 0.93, bottom=0.07,
                                   wspace=0.3, hspace=0.2)

    for iband, band in enumerate(args.band):
        for ibias, bias in enumerate(args.bias):
            if bias == 'LnRbarSqr':
                RbarSqr_panel(gals, stars, band, args.color, fig, outer_grid[ibias, iband],
                              log=True, corrected=args.corrected, **kwargs)
            if bias == 'RbarSqr':
                RbarSqr_panel(gals, stars, band, args.color, fig, outer_grid[ibias, iband],
                              log=False, corrected=args.corrected, **kwargs)
            if bias == 'V':
                V_panel(gals, stars, band, args.color, fig, outer_grid[ibias, iband],
                        corrected=args.corrected, **kwargs)
            if bias == 'S_m02':
                S_m02_panel(gals, stars, band, args.color, fig, outer_grid[ibias, iband],
                            corrected=args.corrected, **kwargs)
    plt.savefig(args.outfile, dpi=220)


if __name__ == '__main__':
    s=3
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = "output/corrected_galaxy_data.pkl",
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = "output/corrected_star_data.pkl",
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('--corrected', action='store_true',
                        help="plot learning residuals instead of G5v residuals.")
    parser.add_argument('--bias', default = ['LnRbarSqr', 'V', 'S_m02'], nargs='*',
                        help="which biases (and their order) to include")
    parser.add_argument('--band', default = ['LSST_r', 'LSST_i'], nargs='*',
                        help="which band (and their order) to include")
    parser.add_argument('--color', default=['LSST_r', 'LSST_i'], nargs=2,
                        help="color to use for symbol color (Default: ['LSST_r', 'LSST_i'])")
    parser.add_argument('--outfile', default="output/bias_panel.png",
                        help="output filename (Default: 'output/bias_panel.png')")
    args = parser.parse_args()

    plot_bias_panel(args, s=s)
