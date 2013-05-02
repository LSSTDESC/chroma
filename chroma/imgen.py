import numpy
import scipy.signal
import galsim

import chroma

class GalSimEngine(object):
    '''Class to use `galsim` to create postage stamp images of multi-component galaxies.

    Idea is to be able to hot-swap this class with the VoigtEngine below which reconstructs the
    postage stamp generation used by Voigt+12.
    '''
    def __init__(self, size=15, oversample_factor=7):
        '''Initialize the image engine

        Arguments
        ---------
        size -- output postage stamp image is `size` x `size` pixels (default 15)
        oversample_factor -- For FWHM calculations, oversample the image by this factor (default 7)
        '''
        self.size=size
        self.oversample_factor=oversample_factor
        self.oversize=size*oversample_factor

    def PSF_FWHM(self, PSF):
        '''Estimate the FWHM of a PSF given by `PSF` (must be galsim.SBProfile object)
        Measures along a single row, so assumes that PSF is circularly symmetric.
        '''
        PSF_image = galsim.ImageD(self.oversize, self.oversize)
        PSF.draw(image=PSF_image, dx=1.0/self.oversample_factor)
        return chroma.utils.FWHM(PSF_image.array, scale=self.oversample_factor)

    def _get_gal(self, obj_list, pixsize):
        '''Create galsim.SBProfile object from list of galaxy profiles, associated PSFs, and a pixel
        scale.
        '''
        pixel = galsim.Pixel(pixsize)
        cvls = [galsim.Convolve(gal, PSF, pixel) for gal, PSF in obj_list]
        gal = galsim.Add(cvls)
        return gal

    def galcvl_FWHM(self, obj_list):
        '''Estimate FWHM of galaxy convolved with PSF.
        '''
        return chroma.utils.FWHM(self.get_image(obj_list, pixsize=1.0/self.oversample_factor),
                                 scale=self.oversample_factor)

    def get_image(self, obj_list, pixsize=1.0):
        '''Create postage stamp image of galaxy.
        '''
        gal = self._get_gal(obj_list, pixsize)
        get_image = galsim.ImageD(int(round(self.size/pixsize)), int(round(self.size/pixsize)))
        gal.draw(image=get_image, dx=pixsize)
        return get_image.array

class GalSimBDEngine(GalSimEngine):
    def gparam_to_galsim(self, gparam):
        bulge = galsim.Sersic(n=gparam['b_n'].value, half_light_radius=gparam['b_r_e'].value)
        bulge.applyShift(gparam['b_x0'].value, gparam['b_y0'].value)
        bulge.applyShear(g=gparam['b_gmag'].value, beta=gparam['b_phi'].value * galsim.radians)
        bulge.setFlux(gparam['b_flux'].value)

        disk = galsim.Sersic(n=gparam['d_n'].value, half_light_radius=gparam['d_r_e'].value)
        disk.applyShift(gparam['d_x0'].value, gparam['d_y0'].value)
        disk.applyShear(g=gparam['d_gmag'].value, beta=gparam['d_phi'].value * galsim.radians)
        disk.setFlux(gparam['d_flux'].value)
        return bulge, disk

    def bdcvl_FWHM(self, gparam, bulge_PSF, disk_PSF):
        bulge, disk = self.gparam_to_galsim(gparam)
        return self.galcvl_FWHM([(bulge, bulge_PSF), (disk, disk_PSF)])

    def bd_image(self, gparam, bulge_PSF, disk_PSF):
        '''Use galsim to make a galaxy image from params in gparam and using the bulge and disk
        PSFs `bulge_PSF` and `disk_PSF`.

        Arguments
        ---------
        gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
                  `b_` prefix for bulge, `d_` prefix for disk.
                  Suffixes are all init arguments for the Sersic object.

        Note that you can specify the composite PSF `c_PSF` for both bulge and disk PSF when using
        during ringtest fits.
        '''
        bulge, disk = self.gparam_to_galsim(gparam)
        return self.get_image([(bulge, bulge_PSF), (disk, disk_PSF)])

class VoigtEngine(object):
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
    def __init__(self, size=15, oversample_factor=7, pad=1, HD_size=5, HD_factor=25):
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
        if key not in self.PSF_image_dict:
            self.PSF_image_dict[key] = PSF(self.ysub, self.xsub)/(self.oversample_factor**2.0)

    def PSF_FWHM(self, PSF):
        '''Estimate the FWHM of a PSF given by `PSF` (must be galsim.SBProfile object)
        Measures along a single row, so assumes that PSF is circularly symmetric.
        '''
        PSF_image = self.get_PSF_image(PSF)
        return chroma.utils.FWHM(PSF_image, scale=self.oversample_factor)

    def galcvl_FWHM(self, obj_list):
        '''Estimate FWHM of galaxy convolved with PSF.
        '''
        return chroma.utils.FWHM(self.get_overimage(obj_list), scale=self.oversample_factor)

    def _get_subpix_centers(self):
        '''Calculate the coordinates of the centers of each subpixel.

        Note that the coordinate axis used assumes that the center of the central pixel is (0,0)
        '''
        pix_start = -(self.padded_size - 1.0) / 2.0 # center of the first full pixel
        pix_end = (self.padded_size - 1.0) / 2.0 # center of the last full pixel
        subpix_start = pix_start - (self.oversample_factor - 1.0) / (2.0 * self.oversample_factor)
        subpix_end = pix_end + (self.oversample_factor - 1.0) / (2.0 * self.oversample_factor)
        subpix_centers = numpy.linspace(subpix_start, subpix_end, self.padded_oversize)
        # note tricksy x, y convention for numpy.meshgrid
        xsub, ysub = numpy.meshgrid(subpix_centers, subpix_centers)
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
        xsubsub = numpy.linspace(xlow, xhigh, self.HD_size * self.HD_factor)
        ysubsub = numpy.linspace(ylow, yhigh, self.HD_size * self.HD_factor)
        # note tricksy x, y convention for numpy.meshgrid
        xsubsub, ysubsub = numpy.meshgrid(xsubsub, ysubsub)
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
        oversampled_image = numpy.zeros((self.padded_oversize, self.padded_oversize),
                                        dtype=numpy.float64)
        for gal, PSF in obj_list:
            galim = gal(self.ysub, self.xsub)
            # do the high-def resampling if needed
            if self.HD_size > 0:
                w=numpy.where(galim == galim.max()) # center high-def region on brightest subpixel
                HD_center = numpy.array([w[0][0], w[1][0]])
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
                galim = scipy.signal.fftconvolve(galim, PSFim, mode='same')
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

class VoigtBDEngine(VoigtEngine):
    def gparam_to_voigt(self, gparam):
        bulge = chroma.SBProfile.Sersic(gparam['b_y0'].value,
                                        gparam['b_x0'].value,
                                        gparam['b_n'].value,
                                        flux=gparam['b_flux'].value,
                                        r_e=gparam['b_r_e'].value,
                                        gmag=gparam['b_gmag'].value,
                                        phi=gparam['b_phi'].value)
        disk = chroma.SBProfile.Sersic(gparam['d_y0'].value,
                                       gparam['d_x0'].value,
                                       gparam['d_n'].value,
                                       flux=gparam['d_flux'].value,
                                       r_e=gparam['d_r_e'].value,
                                       gmag=gparam['d_gmag'].value,
                                       phi=gparam['d_phi'].value)
        return bulge, disk

    def bdcvl_FWHM(self, gparam, bulge_PSF, disk_PSF):
        bulge, disk = self.gparam_to_voigt(gparam)
        return self.galcvl_FWHM([(bulge, bulge_PSF), (disk, disk_PSF)])

    def bd_image(self, gparam, bulge_PSF, disk_PSF):
        '''Use Voigt+12 procedure to make a galaxy image from params in gparam and using the bulge
        and disk PSFs `bulge_PSF` and `disk_PSF`.

        Arguments
        ---------
        gparam -- lmfit.Parameters object with Sersic parameters for both the bulge and disk:
                  `b_` prefix for bulge, `d_` prefix for disk.
                  Suffixes are all init arguments for the Sersic object.

        Note that you can specify the composite PSF `c_PSF` for both bulge and disk PSF when using
        during ringtest fits.
        '''
        bulge, disk = self.gparam_to_voigt(gparam)
        return self.get_image([(bulge, bulge_PSF), (disk, disk_PSF)])
