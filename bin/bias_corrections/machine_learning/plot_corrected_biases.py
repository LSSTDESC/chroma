import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import _mypath
import chroma

hist_axes_range = [0.15, 0.12, 0.1, 0.8]
scatter_axes_range = [0.25, 0.12, 0.55, 0.8]
colorbar_axes_range = [0.85, 0.12, 0.04, 0.8]
data_dir = '../../../data/'

def hist_with_peak(x, bins=None, range=None, ax=None, orientation='vertical',
                   histtype=None, **kwargs):
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
    xs = sorted(x)
    n = len(xs)
    low = xs[int(0.005*n)]
    high = xs[int(0.995*n)]
    span = high-low
    return [low - 0.3*span, high + 0.3*span]

def R_vs_redshift(gals, stars, band, cband1, cband2, yrange=None, prefix='', **kwargs):
    filter_file = data_dir+'filters/{}.dat'.format(band)
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    smom = chroma.disp_moments(swave, sphotons, zenith=np.pi/6)
    R0 = smom[0] * 180/np.pi * 3600

    f = plt.figure(figsize=(8, 5))

    # scatter plot
    ax = f.add_axes(scatter_axes_range)
    x = gals.redshift
    y = (gals['photo_R'][band] - gals['R'][band]) * np.tan(np.pi/6) * 180/np.pi * 3600
    y_uncorrected = gals['R'][band] * np.tan(np.pi/6) * 180/np.pi * 3600 - R0
    c = gals['magCalc'][cband1] - gals['magCalc'][cband2]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])
    im = ax.scatter(x, y, c=c, vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)
    ax.set_xlabel('redshift', fontsize=12)
    ax.set_title('zenith angle = 30, filter = {}'.format(band), fontsize=12)
    ax.yaxis.set_ticklabels([])
    xlim = (-0.1, 3)
    ax.set_xlim(xlim)
    if yrange is None:
        ylim = set_range(y_uncorrected)
    else:
        ylim = yrange
    ax.set_ylim(ylim)
    ax.fill_between([0.0, 3.0], [ylim[0]]*2, [ylim[1]]*2, color='#AAAAAA', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
    # requirements
    ax.fill_between(xlim, [-0.025]*2, [0.025]*2, color='#999999', zorder=2)
    ax.fill_between(xlim, [-0.01]*2, [0.01]*2, color='#888888', zorder=3)

    # star histogram
    hist_ax = f.add_axes(hist_axes_range)
    hist_with_peak((stars['photo_R'][band] - stars['R'][band])
                   * np.tan(np.pi/6) * 180/np.pi * 3600, bins=200,
                   range=ylim, orientation='horizontal', histtype='stepfilled', color='blue')
    hist_ax.xaxis.set_ticklabels([])
    hist_ax.set_ylim(ylim)
    xlim = hist_ax.get_xlim()
    hist_ax.set_xlim(xlim)
    hist_ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='white', zorder=1)
    hist_ax.set_ylabel('$\Delta \overline{\mathrm{R}}$ (arcsec)', fontsize=12)
    hist_with_peak((gals['photo_R'][band] - gals['R'][band])
                   * np.tan(np.pi/6) * 180/np.pi * 3600, bins=200,
                   range=ylim, orientation='horizontal', histtype='step', color='red')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.08,
                 'stars', fontsize=12, color='blue')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.16,
                 'gals', fontsize=12, color='red')
    for label in hist_ax.get_yticklabels():
        label.set_fontsize(12)

    # colorbar
    cbar_ax = f.add_axes(colorbar_axes_range)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('{} - {}'.format(cband1.replace('LSST_',''), cband2.replace('LSST_','')),
                       fontsize=12)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(12)

    f.savefig('output/{}dR_corrected_{}.png'.format(prefix, band), dpi=300)

def V_vs_redshift(gals, stars, band, cband1, cband2, yrange=None, prefix='', **kwargs):
    filter_file = data_dir+'filters/{}.dat'.format(band)
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    smom = chroma.disp_moments(swave, sphotons, zenith=np.pi/6)
    V0 = smom[1] * (180/np.pi * 3600)**2

    f = plt.figure(figsize=(8, 5))

    # scatter plot
    ax = f.add_axes(scatter_axes_range)
    x = gals.redshift
    y = (gals['photo_V'][band] - gals['V'][band]) * (np.tan(np.pi/6) * 180/np.pi * 3600)**2
    y_uncorrected = gals['V'][band] * (np.tan(np.pi/6) * 180/np.pi * 3600)**2 - V0
    c = gals['magCalc'][cband1] - gals['magCalc'][cband2]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])
    im = ax.scatter(x, y, c=c, vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)
    ax.set_xlabel('redshift', fontsize=12)
    ax.set_title('zenith angle = 30, filter = {}'.format(band), fontsize=12)
    ax.yaxis.set_ticklabels([])
    xlim = (-0.1, 3)
    ax.set_xlim(xlim)
    if yrange is None:
        ylim = set_range(y_uncorrected)
    else:
        ylim = yrange
    ax.set_ylim(ylim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#AAAAAA', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
    # requirements
    ax.fill_between(xlim, [-0.0006]*2, [0.0006]*2, color='#999999', zorder=2)
    ax.fill_between(xlim, [-0.0001]*2, [0.0001]*2, color='#888888', zorder=3)

    # star histogram
    hist_ax = f.add_axes(hist_axes_range)
    hist_with_peak((stars['photo_V'][band] - stars['V'][band])
                   * (np.tan(np.pi/6) * 180/np.pi * 3600)**2, bins=200,
                   range=ylim, orientation='horizontal', histtype='stepfilled', color='blue')
    hist_ax.xaxis.set_ticklabels([])
    hist_ax.set_ylim(ylim)
    xlim = hist_ax.get_xlim()
    hist_ax.set_xlim(xlim)
    hist_ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='white', zorder=1)
    hist_ax.set_ylabel('$\Delta \mathrm{V}$ (arcsec$^2$)', fontsize=12)
    hist_with_peak((gals['photo_V'][band] - gals['V'][band])
                   * (np.tan(np.pi/6) * 180/np.pi * 3600)**2, bins=200,
                   range=ylim, orientation='horizontal', histtype='step', color='red')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.08,
                 'stars', fontsize=12, color='blue')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.16,
                 'gals', fontsize=12, color='red')
    # hist_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for label in hist_ax.get_yticklabels():
        label.set_fontsize(12)

    # colorbar
    cbar_ax = f.add_axes(colorbar_axes_range)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('{} - {}'.format(cband1.replace('LSST_',''), cband2.replace('LSST_','')),
                       fontsize=12)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(12)

    f.savefig('output/{}dV_corrected_{}.png'.format(prefix, band), dpi=300)

def S_vs_redshift(gals, stars, band, cband1, cband2, alpha=-0.2, yrange=None, prefix='', **kwargs):
    filter_file = data_dir+'filters/{}.dat'.format(band)
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    S0 = chroma.relative_second_moment_radius(swave, sphotons, alpha=alpha)

    f = plt.figure(figsize=(8, 5))

    # scatter plot
    ax = f.add_axes(scatter_axes_range)
    x = gals.redshift
    if alpha == -0.2:
        powidx = 'S_m02'
        c_powidx = 'photo_S_m02'
    elif alpha == 0.6:
        powidx = 'S_p06'
        c_powidx = 'photo_S_p06'
    elif alpha == 1.0:
        powidx = 'S_p10'
        c_powidx = 'photo_S_p10'
    else:
        raise ValueError
    y = np.log(gals[c_powidx][band] / gals[powidx][band])
    y_uncorrected = np.log(gals[powidx][band] / S0 )
    c = gals['magCalc'][cband1] - gals['magCalc'][cband2]
    clim = set_range(c)
    clim[1] += 0.1 * (clim[1]-clim[0])
    im = ax.scatter(x, y, c=c, vmin=clim[0], vmax=clim[1], zorder=4, **kwargs)
    ax.set_xlabel('redshift', fontsize=12)
    ax.set_title('filter = {}'.format(band), fontsize=12)
    ax.yaxis.set_ticklabels([])
    xlim = (-0.1, 3)
    ax.set_xlim(xlim)
    if yrange is None:
        ylim = set_range(y_uncorrected)
    else:
        ylim = yrange
    ax.set_ylim(ylim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#AAAAAA', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
    # requirements
    if alpha == -0.2:
        ax.fill_between(xlim, [-0.0025]*2, [0.0025]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-0.0004]*2, [0.0004]*2, color='#888888', zorder=3)
    elif alpha == 0.6:
        ax.fill_between(xlim, [-0.002]*2, [0.002]*2, color='#999999', zorder=2)
    elif alpha == 1.0:
        ax.fill_between(xlim, [-0.002]*2, [0.002]*2, color='#999999', zorder=2)
    else:
        raise ValueError

    # star histogram
    hist_ax = f.add_axes(hist_axes_range)
    hist_with_peak(np.log(stars[c_powidx][band] / stars[powidx][band]), bins=200,
                   range=ylim, orientation='horizontal', histtype='stepfilled', color='blue')
    hist_ax.xaxis.set_ticklabels([])
    hist_ax.set_ylim(ylim)
    xlim = hist_ax.get_xlim()
    hist_ax.set_xlim(xlim)
    hist_ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='white', zorder=1)
    hist_ax.set_ylabel('$\Delta r^2_\mathrm{psf} / r^2_\mathrm{psf}$', fontsize=12)
    hist_with_peak(np.log(gals[c_powidx][band] / gals[powidx][band]), bins=200,
                   range=ylim, orientation='horizontal', histtype='step', color='red')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.08,
                 'stars', fontsize=12, color='blue')
    hist_ax.text(xlim[0] + (xlim[1]-xlim[0])*0.2, ylim[1] - (ylim[1]-ylim[0])*0.16,
                 'gals', fontsize=12, color='red')
    for label in hist_ax.get_yticklabels():
        label.set_fontsize(12)

    # colorbar
    cbar_ax = f.add_axes(colorbar_axes_range)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('{} - {}'.format(cband1.replace('LSST_',''), cband2.replace('LSST_','')),
                       fontsize=12)
    for label in cbar_ax.get_yticklabels():
        label.set_fontsize(12)

    f.savefig('output/{}dS_corrected_{}_{}.png'.format(prefix, band, powidx), dpi=300)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = 'corrected_galaxy_data.pkl',
                        help="input galaxy file. Default is 'corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = 'corrected_star_data.pkl',
                        help="input star file. Default is 'corrected_star_data.pkl'")
    parser.add_argument('--prefix', default = '',
                        help="output png file prefix. Default is ''.")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))
    R_vs_redshift(gals, stars, 'LSST_r', 'LSST_r', 'LSST_i', s=2, yrange=[-0.030, 0.015],
                  prefix=args.prefix)
    V_vs_redshift(gals, stars, 'LSST_r', 'LSST_r', 'LSST_i', s=2, yrange=[-0.0008, 0.0005],
                  prefix=args.prefix)
    S_vs_redshift(gals, stars, 'LSST_r', 'LSST_r', 'LSST_i', s=2, yrange=[-0.02, 0.01],
                  prefix=args.prefix)
    R_vs_redshift(gals, stars, 'LSST_i', 'LSST_r', 'LSST_i', s=2, yrange=[-0.030, 0.015],
                  prefix=args.prefix)
    V_vs_redshift(gals, stars, 'LSST_i', 'LSST_r', 'LSST_i', s=2, yrange=[-0.0008, 0.0005],
                  prefix=args.prefix)
    S_vs_redshift(gals, stars, 'LSST_i', 'LSST_r', 'LSST_i', s=2, yrange=[-0.02, 0.01],
                  prefix=args.prefix)
    S_vs_redshift(gals, stars, 'Euclid_350', 'LSST_r', 'LSST_i', alpha=0.6, s=2,
                  prefix=args.prefix)
