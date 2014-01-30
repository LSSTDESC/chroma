import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mpl as mpl
import matplotlib.scale as scale
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import astropy.io.fits as fits


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
    lvmin = np.log10(-vmin)
    lvmax = np.log10(vmax)
    llinthresh = np.log10(linthresh)
    decades = ((lvmin - llinthresh) + (lvmax - llinthresh) + linscale)
    ticks = np.arange(0, 1.00001, 1.0/decades)
    labels = ['-1e{}'.format(i) for i in range(int(lvmin), int(llinthresh), -1)]
    labels += ['-1e{}'.format(int(llinthresh))]
    labels += ['1e{}'.format(int(llinthresh))]
    labels += ['1e{}'.format(i) for i in range(int(llinthresh)+1, int(lvmax)+1)]
    return ticks, labels


def plot_two_cmap(filename):
    cmap_resid = cm.PiYG
    cmap_flux = cm.spectral

    # load data
    hdulist = fits.open(filename)
    uncvlim = hdulist[0].data
    gal_PSF_overim = hdulist[1].data
    overim = hdulist[2].data
    im = hdulist[3].data
    fit_uncvlim = hdulist[4].data
    star_PSF_overim = hdulist[5].data
    fit_overim = hdulist[6].data
    fit_im = hdulist[7].data
    nudge_uncvlim = hdulist[8].data

    im /= 49
    fit_im /= 49
    norm = 0.00954924

    # plot
    f = plt.figure(figsize=(6, 5))

    linthresh = 1e-4
    vmin = -1e0
    vmax = 1e0

    # first row = truth
    ax = f.add_subplot(441)
    #symlog10imshow(uncvlim/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_flux)
    flux_img = my_imshow(np.log10(uncvlim/norm), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylabel('truth')

    ax = f.add_subplot(442)
    # symlog10imshow(gal_PSF_overim/gal_PSF_overim.max(), vmin, vmax, linthresh, 1, ax=ax,
    #                cmap=cmap_flux)
    my_imshow(np.log10(gal_PSF_overim/gal_PSF_overim.max()), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = f.add_subplot(443)
    # symlog10imshow(overim/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_flux)
    my_imshow(np.log10(overim/norm), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = f.add_subplot(444)
    # symlog10imshow(im/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_flux)
    my_imshow(np.log10(im/norm), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # second row = truth
    ax = f.add_subplot(445)
    # symlog10imshow(fit_uncvlim/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_flux)
    my_imshow(np.log10(fit_uncvlim/norm), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylabel('fit')

    ax = f.add_subplot(446)
    # symlog10imshow(star_PSF_overim/star_PSF_overim.max(), vmin, vmax, linthresh, 1, ax=ax,
    #                cmap=cmap_flux)
    my_imshow(np.log10(star_PSF_overim/star_PSF_overim.max()), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = f.add_subplot(447)
    # symlog10imshow(fit_overim/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_flux)
    my_imshow(np.log10(fit_overim/norm), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = f.add_subplot(448)
    #symlog10imshow(fit_im/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_flux)
    my_imshow(np.log10(fit_im/norm), ax=ax, cmap=cmap_flux, vmin=-5, vmax=0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # third row = residual
    ax = f.add_subplot(4,4,11)
    symlog10imshow((overim - fit_overim)/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_resid)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = f.add_subplot(4,4,12)
    symlog10imshow((im - fit_im)/norm, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_resid)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # third row = residual
    ax = f.add_subplot(4,4,15)
    symlog10imshow((overim - fit_overim)/overim, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_resid)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylabel('(truth - fit) / truth')

    ax = f.add_subplot(4,4,16)
    resid_img = symlog10imshow((im - fit_im)/im, vmin, vmax, linthresh, 1, ax=ax, cmap=cmap_resid)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    f.subplots_adjust(hspace = 0.1, wspace=0.1, right=0.8)

    cbar_flux_ax = f.add_axes([0.85, 0.51, 0.04, 0.35])
    cbar = plt.colorbar(flux_img, cax=cbar_flux_ax, cmap=cmap_flux)

    ticks, labels = symlogTicksAndLabels(vmin, vmax, linthresh, 1)
    cbar_resid_ax = f.add_axes([0.85, 0.1, 0.04, 0.35])
    cbar_resid = plt.colorbar(resid_img, cax=cbar_resid_ax, cmap=cmap_resid, ticks=ticks)
    cbar_resid.ax.set_yticklabels(labels)

    # cbar = plt.colorbar(img, cax=cbar_ax, orientation='horizontal', cmap=cmap_resid,
    #                     ticks=ticks)

    plt.show()


if __name__ == '__main__':
    plot_two_cmap('output/one_case.fits')
