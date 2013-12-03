import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits

def my_imshow(my_img,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return '%5e @ [%4i, %4i]' % (my_img[y, x], x, y)
        except IndexError:
            return ''
    img = ax.imshow(my_img,**kwargs)
    ax.format_coord = format_coord
    return img

def plot_one_case_study(filename, cmap='jet'):

    hdulist = fits.open(filename)
    uncvlim = hdulist[0].data
    gal_PSF_overim = hdulist[1].data
    overim = hdulist[2].data
    im = hdulist[3].data
    fit_uncvlim = hdulist[4].data
    star_PSF_overim = hdulist[5].data
    fit_overim = hdulist[6].data
    fit_im = hdulist[7].data

    f, axarr = plt.subplots(4, 4, figsize=(6,6))
    im /= 49
    fit_im /=49
    ims = [uncvlim, gal_PSF_overim, overim, im,
           fit_uncvlim, star_PSF_overim, fit_overim, fit_im,
           fit_uncvlim - uncvlim, gal_PSF_overim - star_PSF_overim,
           fit_overim - overim, fit_im - im,
           (fit_uncvlim - uncvlim)/uncvlim,
           (star_PSF_overim - gal_PSF_overim)/gal_PSF_overim,
           (fit_overim - overim)/overim,
           (fit_im - im)/im]
    k=0
    norm = 0.00954924
    for i in range(4):
        for j in range(4):
            if j == 1: #PSF
                img = my_imshow(np.log10(abs(ims[k]/ims[k].max())), ax=axarr[i, j],
                                extent=[0, im.shape[0], 0, im.shape[1]],
                                vmin=-5, vmax=0, cmap=cmap)
            elif i==3: #frac residual
                img = my_imshow(np.log10(abs(ims[k])), ax=axarr[i, j],
                                extent=[0, im.shape[0], 0, im.shape[1]],
                                vmin=-5, vmax=0, cmap=cmap)
            else: #non-PSF image/resid
                img = my_imshow(np.log10(abs(ims[k]/norm)), ax=axarr[i, j],
                                extent=[0, im.shape[0], 0, im.shape[1]],
                                vmin=-5, vmax=0, cmap=cmap)
            axarr[i, j].get_xaxis().set_ticks([])
            axarr[i, j].get_yaxis().set_ticks([])
            k += 1
    axarr[0,0].set_title('galaxy')
    axarr[0,1].set_title('PSF')
    axarr[0,2].set_title('convolution')
    axarr[0,3].set_title('pixelized')
    axarr[0,0].set_ylabel('truth')
    axarr[1,0].set_ylabel('best fit')
    axarr[2,0].set_ylabel('abs(residual')
    axarr[3,0].set_ylabel('abs(rel resid)')
    f.subplots_adjust(hspace=0.02, wspace=0.07, bottom=0.11)
    cbar_ax = f.add_axes([0.122, 0.05, 0.77, 0.04])
    plt.colorbar(img, cax=cbar_ax, orientation='horizontal', cmap=cmap)
    plt.show()


if __name__ == '__main__':
    plot_one_case_study('one_case.fits')
