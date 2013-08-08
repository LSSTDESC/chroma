from pylab import *
import galsim

size=15
over=7

# create PSF components and compositions
psf1 = galsim.Gaussian(flux=1.0, fwhm=2.5)
psf2 = galsim.Gaussian(flux=0.5, fwhm=1.5)
psf2.applyShift(0.0, 2.0) # shift 5 pixels up
psf = galsim.Add([psf1, psf2])
psf_coarse = galsim.Convolve([psf, galsim.Pixel(1.)])
psf_fine = galsim.Convolve([psf, galsim.Pixel(1./7)])

# create some images to hold PSFs for visualization
im1 = galsim.ImageD(size, size)               # no pix, coarse
im2 = galsim.ImageD(size * over, size * over) # no pix, fine
im3 = galsim.ImageD(size, size)               # pix, coarse
im4 = galsim.ImageD(size * over, size * over) # pix, fine

# draw the PSFs
psf.draw(image=im1, dx=1.)
psf.draw(image=im2, dx=1./over)
psf_coarse.draw(image=im3, dx=1.)
psf_fine.draw(image=im4, dx=1./over)

# display
f=figure()
ax1=f.add_subplot(221)
ax1.imshow(im1.array)
ax1.set_title('no pix, coarse')
ax2=f.add_subplot(222)
ax2.imshow(im2.array)
ax2.set_title('no pix, fine')
ax3=f.add_subplot(223)
ax3.imshow(im3.array)
ax3.set_title('pix, coarse')
ax4=f.add_subplot(224)
ax4.imshow(im4.array)
ax4.set_title('pix, fine')
show()
