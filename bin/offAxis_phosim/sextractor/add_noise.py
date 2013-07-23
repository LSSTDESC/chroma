import pyfits
from numpy.random import normal

if __name__ == "__main__":
    import sys
    fn = sys.argv[1]
    hdulist = pyfits.open(fn)
    image = hdulist[0].data
    noisy_image = image + normal(25.0, 5.0, image.shape)
    output_hdu = pyfits.PrimaryHDU(noisy_image, hdulist[0].header)
    outfn = fn.replace('.fits', '_noisy.fits').replace('.gz', '')
    output_hdu.writeto(outfn)
