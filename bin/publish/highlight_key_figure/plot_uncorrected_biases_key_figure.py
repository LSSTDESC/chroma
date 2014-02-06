import cPickle
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import _mypath
import chroma

hist_axes_range = [0.13, 0.12, 0.1, 0.8]
scatter_axes_range = [0.23, 0.12, 0.65, 0.8]
colorbar_axes_range = [0.76, 0.15, 0.025, 0.35]
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

def S_vs_redshift(gals, stars, band, cband1, cband2, alpha=-0.2, yrange=None, **kwargs):
    filter_file = data_dir+'filters/{}.dat'.format(band)
    star_SED_file = data_dir+'SEDs/ukg5v.ascii'
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    S0 = chroma.relative_second_moment_radius(swave, sphotons, alpha=alpha)

    f = plt.figure(figsize=(8, 6))

    # scatter plot
    ax = f.add_axes(scatter_axes_range)
    x = gals.redshift
    if alpha == -0.2:
        powidx = 'S_m02'
    elif alpha == 0.6:
        powidx = 'S_p06'
    elif alpha == 1.0:
        powidx = 'S_p10'
    else:
        raise ValueError
    y = np.log(gals[powidx][band] / S0 )
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
        ylim = set_range(y)
    else:
        ylim = yrange
    ax.set_ylim(ylim)
    ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='#BBBBBB', zorder=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
    # requirements
    if alpha == -0.2:
        ax.fill_between(xlim, [-0.0025]*2, [0.0025]*2, color='#999999', zorder=2)
        ax.fill_between(xlim, [-0.0004]*2, [0.0004]*2, color='#777777', zorder=3)
    elif alpha == 0.6:
        ax.fill_between(xlim, [-0.002]*2, [0.002]*2, color='#999999', zorder=2)
    elif alpha == 1.0:
        ax.fill_between(xlim, [-0.002]*2, [0.002]*2, color='#999999', zorder=2)
    else:
        raise ValueError
    ax.annotate('DES requirement (|m|<0.008)', xy=(0.55, 0.0025), xytext=(0.35, 0.004),
                fontsize=10, arrowprops={'arrowstyle':'->', 'color':'black'},
                zorder=10)
    ax.annotate('LSST requirement (|m|<0.0015)', xy=(2.1, -0.0004), xytext=(1.65, -0.004),
                fontsize=10, arrowprops={'arrowstyle':'->', 'color':'black'},
                zorder=10)
    #second y-axis
    ax2 = ax.twinx()
    ax2.set_ylim([y * 3.22 for y in ax.get_ylim()])
    ax2.set_ylabel('multiplicative shear bias', fontsize=12)

    # star histogram
    hist_ax = f.add_axes(hist_axes_range)
    hist_with_peak(np.log(stars[powidx][band] / S0), bins=200,
                   range=ylim, orientation='horizontal', histtype='stepfilled', color='blue')
    hist_ax.xaxis.set_ticklabels([])
    hist_ax.set_ylim(ylim)
    xlim = hist_ax.get_xlim()
    hist_ax.set_xlim(xlim)
    hist_ax.fill_between(xlim, [ylim[0]]*2, [ylim[1]]*2, color='white', zorder=1)
    hist_ax.set_ylabel('$\Delta r^2_\mathrm{PSF} / r^2_\mathrm{PSF}$', fontsize=12)
    hist_with_peak(np.log(gals[powidx][band] / S0), bins=200,
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

    f.savefig('output/dS_{}_{}.png'.format(band, powidx), dpi=200)

if __name__ == '__main__':
    pkl_dir = '../../bias_corrections/machine_learning/'
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = pkl_dir+'galaxy_data_noemission.pkl',
                        help="input galaxy file.")
    parser.add_argument('--starfile', default = pkl_dir+'corrected_star_data.pkl',
                        help="input star file.")
    args = parser.parse_args()

    gals = cPickle.load(open(args.galfile))
    stars = cPickle.load(open(args.starfile))
    S_vs_redshift(gals, stars, 'LSST_r', 'LSST_r', 'LSST_i', s=2, yrange=[-0.02, 0.01])
