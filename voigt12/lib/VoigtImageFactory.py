import numpy as np
from scipy.signal import fftconvolve

class VoigtImageFactory(object):
    ''' Class to create 15x15 pixel postage stamp images of galaxies as described in Voigt+12.

    Procedure is:
    1. Use a grid 17x17, which will be clipped to 15x15 at the end.
    2. Oversample each pixel in this grid with 7x7 subpixels.
    3. Further oversample each of the central 5x5 subpixels with 25x25 subsubpixels.
    4. Evaluate Sersic profile(s) at the center of each subpixel or subsubpixel.
    5. Average together subsubpixels into subpixels.
    6. Create an image of the PSF on the same 17^2 x 7^2 oversampled grid.
    7. Convolve the oversampled galaxy image by the oversampled PSF image.
    8. Bin down to 17x17 by taking the means of subpixels
    9. Clip 17x17 pixel image to 15x15 pixels.
    This class generalizes the above procedure to arbitrary oversampling sizes, regions and paddings.
    '''
    def __init__(self, size=15, pad=1, oversample_factor=7, HD_size=5, HD_factor=25):
        '''Initialize the image factory

        Arguments
        ---------
        size -- output postage stamp image is (size x size) pixels (default 15)
        pad -- pad the postage stamp image internally by pad pixels (on each side) before
               clipping down for final output. (default 1, produces 17x17 padded image with
               defalt size setting.)
        oversample_factor -- number of linear subpixels per pixel. (default 7)
        HD_size -- linear size of the region to overoversample in subpixels (default 5 further
                   oversamples 5x5 central subpixels)
        HD_factor -- number of linear subsubpixels per subpixel. (default 25)
        '''
        self.size = size
        self.pad = pad
        self.oversample_factor = oversample_factor
        self.HD_size = HD_size
        self.HD_factor = HD_factor
        self.PSF_image_dict = {} # cache PSF images
        self.padded_size = self.size + 2 * self.pad
        self.padded_oversize = self.padded_size * self.oversample_factor
        self.ysub, self.xsub = self._get_subpix_centers()

    @staticmethod
    def _rebin(a, shape):
        '''Bin down image a to have final size given by shape.

        I think I stole this from stackoverflow somewhere...
        '''
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    def get_PSF_image(self, PSF):
        '''Get oversampled image of PSF from dictionary cache if possible, else generate.'''
        try:
            key = PSF.key # use key if PSF defines one (such as Voigt12PSF)
        except AttributeError:
            key = id(PSF) # otherwise, id is guaranteed to not cause a collision at least...)
        if key not in self.PSF_image_dict.keys():
            self.load_PSF(PSF)
        return self.PSF_image_dict[key]

    def load_PSF(self, PSF):
        '''Evaluate PSF at subpixel centers and store in PSF dictionary cache.'''
        try:
            key = PSF.key
        except AttributeError:
            key = id(PSF)
        self.PSF_image_dict[key] = PSF(self.ysub, self.xsub)/(self.oversample_factor**2.0)

    def _get_subpix_centers(self):
        '''Calculate the coordinates of the centers of each subpixel.

        Note that the coordinate axis used assumes that the center of the central pixel is (0,0)
        '''
        pix_start = -(self.padded_size - 1.0) / 2.0 # center of the first full pixel
        pix_end = (self.padded_size - 1.0) / 2.0 # center of the last full pixel
        subpix_start = pix_start - (self.oversample_factor - 1.0) / (2.0 * self.oversample_factor)
        subpix_end = pix_end + (self.oversample_factor - 1.0) / (2.0 * self.oversample_factor)
        subpix_centers = np.linspace(subpix_start, subpix_end, self.padded_oversize)
        # note tricksy x, y convention for np.meshgrid
        xsub, ysub = np.meshgrid(subpix_centers, subpix_centers)
        return ysub, xsub

    def _get_subsubpix_centers(self, HD_center_coord):
        '''Calculate the coordinates of the centers of each subsubpixel in the high-def region.

        Arguments
        ---------
        HD_center_coord -- the coordinate (w.r.t. center of central full pixel) of the
                           high-definition region.
        '''
        reg_size = 1.0 * self.HD_size / self.oversample_factor
        subsubpix_size = 1.0 / (self.oversample_factor * self.HD_factor)
        xlow = HD_center_coord[1] - reg_size / 2.0 + 0.5 * subsubpix_size
        ylow = HD_center_coord[0] - reg_size / 2.0 + 0.5 * subsubpix_size
        xhigh = HD_center_coord[1] + reg_size / 2.0 - 0.5 * subsubpix_size
        yhigh = HD_center_coord[0] + reg_size / 2.0 - 0.5 * subsubpix_size
        xsubsub = np.linspace(xlow, xhigh, self.HD_size * self.HD_factor)
        ysubsub = np.linspace(ylow, yhigh, self.HD_size * self.HD_factor)
        # note tricksy x, y convention for np.meshgrid
        xsubsub, ysubsub = np.meshgrid(xsubsub, ysubsub)
        return ysubsub, xsubsub

    def _get_HD_subpix_region(self, HD_center):
        '''Calculate the region boundaries of the high-def region, in units of subpixels.

        Arguments
        ---------
        HD_center -- the indices of the brightest subpixel and hence center of the high-definition
                     region.
        '''
        HD_xlow = HD_center[1] - (self.HD_size-1.0)/2.0
        HD_xhigh = HD_center[1] + (self.HD_size-1.0)/2.0 + 1
        HD_ylow = HD_center[0] - (self.HD_size-1.0)/2.0
        HD_yhigh = HD_center[0] + (self.HD_size-1.0)/2.0 + 1
        return HD_ylow, HD_yhigh, HD_xlow, HD_xhigh

    def get_overimage(self, obj_list):
        '''Produce the PSF-convolved oversampled padded image given a list of galaxy profile
        functions and PSF functions.

        Arguments
        ---------
        obj_list -- list of tuples, each tuple consists of:
                    (gal, PSF).
                    gal -- a callable (arguments y, x) returning the surface brightness of the
                           galaxy profile.
                    PSF -- a callable (arguments y, x) returning the intensity of the PSF.

        Different PSFs can be used for different galaxy components, permitting treatment of
        wavelength-dependent PSFs and galaxies with color gradients.  If the PSF is `None`,
        then the convolution for that component is skipped.
        '''
        oversampled_image = np.zeros((self.padded_oversize, self.padded_oversize),
                                     dtype=np.float64)
        for gal, PSF in obj_list:
            galim = gal(self.ysub, self.xsub)
            # do the high-def resampling if needed
            if self.HD_size > 0:
                w=np.where(galim == galim.max()) # center high-def region on brightest subpixel
                HD_center = np.array([w[0][0], w[1][0]])
                HD_reg = self._get_HD_subpix_region(HD_center)
                # proceed only if HD region is not too close to the edges
                if min(HD_reg) >= 0 and max(HD_reg) < self.padded_oversize:
                    HD_center_coord = self.ysub[w][0], self.xsub[w][0]
                    ysubsub, xsubsub = self._get_subsubpix_centers(HD_center_coord)
                    galHD = gal(ysubsub, xsubsub)
                    galim[HD_reg[0]:HD_reg[1], HD_reg[2]:HD_reg[3]] = \
                      self._rebin(galHD, (self.HD_size, self.HD_size))
            if PSF is not None:
                PSFim = self.get_PSF_image(PSF)
                galim = fftconvolve(galim, PSFim, mode='same')
            oversampled_image += galim
        return oversampled_image

    def get_padded_image(self, obj_list):
        '''Downsample the image returned by `get_overimage`, but don't remove the padding yet.'''
        oversampled_image = self.get_overimage(obj_list)
        padded_im = self._rebin(oversampled_image, (self.padded_size, self.padded_size))
        return padded_im

    def get_image(self, obj_list):
        '''Remove the padding from the image returned by `get_padded_image`'''
        padded_im = self.get_padded_image(obj_list)
        im = padded_im[self.pad:-self.pad, self.pad:-self.pad]
        return im

if __name__ == '__main__':
    # compare our image generation to that produced by GalSim.
    # The GalSim comparison image and yaml file for producing that image are in the
    # data directory.
    from sersic import Sersic
    import pyfits
    image_factory = VoigtImageFactory()
    image_factory_extreme = VoigtImageFactory(pad=3, oversample_factor=25, HD_size=75, HD_factor=35)

    gmag = 0.2
    phi = 30 * np.pi/180.0

    bulge_flux = 0.25
    bulge_rad = 2.0
    bulge_n = 4.0

    disk_flux = 0.75
    disk_rad = 3.0
    disk_n = 1.0

    y0=0.3
    x0=0.1

    bulge = Sersic(y0, x0, bulge_n, r_e=bulge_rad, flux=bulge_flux, phi=phi, gmag=gmag)
    disk = Sersic(y0, x0, disk_n, r_e=disk_rad, flux=disk_flux, phi=phi, gmag=gmag)

    FWHM = 1.0 * np.sqrt(8.0 * np.log(2.0))
    psf = Sersic(0.0, 0.0, 0.5, FWHM=FWHM, flux=1.0, phi=0.0, gmag=0.0)

    image = image_factory.get_image([(disk, psf),(bulge, psf)])
    gimage = pyfits.getdata('../data/galsim_test/galsim_test.fits')

    print 'Comparing two images each with total flux=1.0.  The differences should be small'
    print 'RMS difference with default settings: {}'.format(np.std(image-gimage))

    image_extreme = image_factory_extreme.get_image([(disk, psf),(bulge, psf)])
    print 'RMS difference with extreme settings: {}'.format(np.std(image_extreme-gimage))
