import sys
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.scale as scale
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import astropy.io.fits as fits

import _mypath
import chroma

def symlogtransform(vmin, vmax, linthresh, linscale):
    """Assumes that minimum of domain is negative, maximum of domain is positive,
    and therefore that vmin < 0, vmax > 0.
    """
    lvmin = np.log10(-vmin)
    lvmax = np.log10(vmax)
    llinthresh = np.log10(linthresh)
    decades = ((lvmin - llinthresh) + (lvmax - llinthresh) + linscale)
    def transform(x):
        wlogneg = (x < -linthresh)
        wlogpos = (x > linthresh)
        wlin = (x >= -linthresh) & (x <= linthresh)
        y = np.zeros_like(x, dtype=np.float)
        y[wlogneg] = (-np.log10(-x[wlogneg]) - (-lvmin)) / decades
        y[wlogpos] = (np.log10(x[wlogpos]) - lvmax + decades) / decades
        y[wlin] = (x[wlin] / linthresh / linscale / 2
                   + (lvmin - llinthresh) + linscale / 2.0) / decades
        return y
    return transform

def my_imshow(im, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%8e @ [%4i, %4i]' % (im[y, x], x, y)
        except IndexError:
            return ''
    img = ax.imshow(im, **kwargs)
    ax.format_coord=format_coord
    return img

def symlog10imshow(im, vmin, vmax, linthresh, linscale, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%8e @ [%4i, %4i]' % (im[y, x], x, y)
        except IndexError:
            return ''

    t = symlogtransform(vmin, vmax, linthresh, linscale)
    transformed_img = t(im)
    img = ax.imshow(transformed_img, vmin=0, vmax=1, **kwargs)
    ax.format_coord = format_coord
    return img

def symlogTicksAndLabels(vmin, vmax, linthresh, linscale):
    """Only works for linscale=1
    """
    lvmin = np.log10(-vmin)
    lvmax = np.log10(vmax)
    llinthresh = np.log10(linthresh)
    decades = ((lvmin - llinthresh) + (lvmax - llinthresh) + linscale)
    ticks = np.arange(0, 1.00001, 1.0/decades)
    labels = ['-1e{}'.format(i) for i in range(int(lvmin), int(llinthresh), -1)]
    labels += ['-1e{}'.format(int(llinthresh))]
    labels += ['+1e{}'.format(int(llinthresh))]
    labels += ['+1e{}'.format(i) for i in range(int(llinthresh)+1, int(lvmax)+1)]
    return ticks, labels


def plot_ring_diag(args):
    cmap_resid = chroma.bipolar.bipolar(n=1./3)
    cmap_flux = cm.Blues_r

    # load data
    hdulist = fits.open(args.infile)
    gal_PSF_overim = np.abs(hdulist[0].data[28:96,28:96].T)
    star_PSF_overim = np.abs(hdulist[1].data[28:96,28:96].T)

    nring = (len(hdulist)-2)/6
    for i in range(nring):
    # for i in range(1):
        im = np.abs(hdulist[2+6*i].data[7:24,7:24].T)
        overim = np.abs(hdulist[3+6*i].data[28:96,28:96].T)
        uncvlim = np.abs(hdulist[4+6*i].data[28:96,28:96].T)
        fit_im = np.abs(hdulist[5+6*i].data[7:24,7:24].T)
        fit_overim = np.abs(hdulist[6+6*i].data[28:96,28:96].T)
        fit_uncvlim = np.abs(hdulist[7+6*i].data[28:96,28:96].T)
        nudge_uncvlim = fit_uncvlim
        g1 = hdulist[2+6*i].header['GAMMA1']
        g2 = hdulist[2+6*i].header['GAMMA2']
        beta = hdulist[2+6*i].header['BETA']

        # correct some zeros issues
        uncvlim = np.abs(uncvlim)
        fit_uncvlim = np.abs(fit_uncvlim)

        # plot
        f = plt.figure(figsize=(7, 4.75))

        # first row = truth
        vmin = -4
        vmax = 0
        levels = [-0.5, -1.5, -2.5, -3.5]
        uncvl_levels = [-1.5, -2.5, -3.5, -4.5]

        ax = f.add_subplot(341)
        image = np.log10(uncvlim/uncvlim.max())
        flux_img = my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax, extent=[0,17*4,0,17*4])
        ax.contour(image, levels=uncvl_levels, colors='k', linestyles='solid', origin='image')
        ax.xaxis.grid(zorder=10, linestyle='solid')
        ax.yaxis.grid(zorder=10, linestyle='solid')
        ax.get_xaxis().set_ticks(np.arange(4, 68, 12))
        ax.get_yaxis().set_ticks(np.arange(4, 68, 12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel('truth', size=18)
        title = ax.set_title('model', size=18)

        ax = f.add_subplot(342)
        image = np.log10(gal_PSF_overim/gal_PSF_overim.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax, extent=[0,17*4,0,17*4])
        ax.contour(image, levels=levels, colors='k', linestyles='solid', origin='image')
        ax.xaxis.grid(zorder=10, linestyle='solid')
        ax.yaxis.grid(zorder=10, linestyle='solid')
        ax.get_xaxis().set_ticks(np.arange(4, 68, 12))
        ax.get_yaxis().set_ticks(np.arange(4, 68, 12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('PSF', size=18)

        ax = f.add_subplot(343)
        image = np.log10(overim/overim.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax, extent=[0,17*4,0,17*4])
        ax.contour(image, levels=levels, colors='k', linestyles='solid', origin='image')
        ax.xaxis.grid(zorder=10, linestyle='solid')
        ax.yaxis.grid(zorder=10, linestyle='solid')
        ax.get_xaxis().set_ticks(np.arange(4, 68, 12))
        ax.get_yaxis().set_ticks(np.arange(4, 68, 12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('image', size=18)

        ax = f.add_subplot(344)
        image = np.log10(im/im.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax)
        ax.contour(image, levels=levels, colors='k', linestyles='solid')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title('pixelized', size=18)

        # second row = truth
        vmin = -4
        vmax = 0

        ax = f.add_subplot(345)
        image = np.log10(fit_uncvlim/uncvlim.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax, extent=[0,17*4,0,17*4])
        ax.contour(image, levels=uncvl_levels, colors='k', linestyles='solid', origin='image')
        ax.xaxis.grid(zorder=10, linestyle='solid')
        ax.yaxis.grid(zorder=10, linestyle='solid')
        ax.get_xaxis().set_ticks(np.arange(4, 68, 12))
        ax.get_yaxis().set_ticks(np.arange(4, 68, 12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel('fit', size=18)

        ax = f.add_subplot(346)
        image = np.log10(star_PSF_overim/gal_PSF_overim.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax, extent=[0,17*4,0,17*4])
        ax.contour(image, levels=levels, colors='k', linestyles='solid', origin='image')
        ax.xaxis.grid(zorder=10, linestyle='solid')
        ax.yaxis.grid(zorder=10, linestyle='solid')
        ax.get_xaxis().set_ticks(np.arange(4, 68, 12))
        ax.get_yaxis().set_ticks(np.arange(4, 68, 12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = f.add_subplot(347)
        image = np.log10(fit_overim/overim.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax, extent=[0,17*4,0,17*4])
        ax.contour(image, levels=levels, colors='k', linestyles='solid', origin='image')
        ax.xaxis.grid(zorder=10, linestyle='solid')
        ax.yaxis.grid(zorder=10, linestyle='solid')
        ax.get_xaxis().set_ticks(np.arange(4, 68, 12))
        ax.get_yaxis().set_ticks(np.arange(4, 68, 12))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = f.add_subplot(348)
        image = np.log10(fit_im/im.max())
        my_imshow(image, ax=ax, cmap=cmap_flux, vmin=vmin, vmax=vmax)
        ax.contour(image, levels=levels, colors='k', linestyles='solid')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # third row = residual
        linthresh = 1e-6
        vmin = -1e-2
        vmax = 1e-2
        linscale = 2

        ax = f.add_subplot(3,4,9)
        symlog10imshow((uncvlim - nudge_uncvlim)/uncvlim.max(), vmin, vmax, linthresh, linscale,
                       ax=ax, cmap=cmap_resid)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_ylabel('truth - fit', size=18)

        ax = f.add_subplot(3,4,11)
        symlog10imshow((overim - fit_overim)/overim.max(), vmin, vmax, linthresh, linscale,
                       ax=ax, cmap=cmap_resid)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax = f.add_subplot(3,4,12)
        resid_img = symlog10imshow((im - fit_im)/im.max(), vmin, vmax, linthresh, linscale,
                                   ax=ax, cmap=cmap_resid)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        f.subplots_adjust(hspace = 0.1, wspace=0.0, right=0.8, bottom=0.175, left=0.075)

        #manually adjust ticks and labels
        ticks = [-4, -3, -2, -1, 0]
        labels = ['1e-4', '1e-3', '1e-2', '1e-1', '1']
        cbar_flux_ax = f.add_axes([0.84, 0.43, 0.04, 0.47])
        cbar = plt.colorbar(flux_img, cax=cbar_flux_ax, cmap=cmap_flux, ticks=ticks)
        cbar.ax.set_yticklabels(labels, fontsize=18)

        # ticks, labels = symlogTicksAndLabels(vmin, vmax, linthresh, linscale)
        # manual override
        labels = ['-1e-2', '-1e-4', '-1e-6', '+1e-6', '+1e-4', '+1e-2']
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        cbar_resid_ax = f.add_axes([0.09, 0.11, 0.697, 0.04])
        cbar_resid = plt.colorbar(resid_img, cax=cbar_resid_ax, cmap=cmap_resid, ticks=ticks,
                                  orientation='horizontal')
        cbar_resid.ax.set_xticklabels(labels, fontsize=18)

        # cbar = plt.colorbar(img, cax=cbar_ax, orientation='horizontal', cmap=cmap_resid,
        #                     ticks=ticks)

        plt.savefig(args.outprefix+'-g1-{}-g2-{}-beta{}.png'.format(g1, g2, beta),
                    dpi=300)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('infile', help="Input diagnostic fits filename.")
    parser.add_argument('outprefix', nargs='?', default="output/ring_diag",
                        help="Output PNG filename prefix. (Default: output/ring_diag)")
    args = parser.parse_args()
    plot_ring_diag(args)
