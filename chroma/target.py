import galsim
import astropy.io.fits as fits
import chroma


class TargetImageGenerator:
    """A class to generate target images in a ring test.

    @param gparam      lmfit.Parameters object containing galaxy attributes
    @param galtool     Used to convert lmfit.Parameters object into an actual image using GalSim.
    @param hdulist     If not None, then write images to this FITS file hdulist.
    @param oversample  Amount by which to oversample images optionally placed in FITS file.
    """

    def __init__(self, gparam, galtool, hdulist=None, oversample=4):
        self.gparam = gparam.copy()
        self.galtool = galtool
        self.hdulist = hdulist
        self.oversample = oversample

    def __call__(self, gamma, beta):
        """Return ring test image, optionally drawing to hdulist along the way.

        @param gamma  Shear to apply to galaxy before drawing as a complex number.
        @param beta   Angle around ellipticity ring for ring test in radians.  Corresponds to
                      rotating the fiducial galaxy by `beta/2.0` radians.
        @returns      Sheared and rotated galaxy image.
        """
        shear = galsim.Shear(g1=gamma.real, g2=gamma.imag)
        target_image = self.galtool.get_image(
            self.gparam, ring_beta=beta, ring_shear=shear
        )
        if self.hdulist is not None:
            hdu = fits.ImageHDU(target_image.array, name="TARGET")
            hdu.header.append(("GAMMA1", gamma.real))
            hdu.header.append(("GAMMA2", gamma.imag))
            hdu.header.append(("BETA", beta))
            for k, v in self.gparam.items():
                hdu.header.append((k, v.value))
            self.hdulist.append(hdu)
            target_high_res = self.galtool.get_image(
                self.gparam,
                ring_beta=beta,
                ring_shear=shear,
                oversample=self.oversample,
            )
            self.hdulist.append(fits.ImageHDU(target_high_res.array, name="TARGETHR"))
            target_uncvl = self.galtool.get_uncvl_image(
                self.gparam,
                ring_beta=beta,
                ring_shear=shear,
                oversample=self.oversample,
                center=True,
            )
            self.hdulist.append(fits.ImageHDU(target_uncvl.array, name="TARGETUC"))
        return target_image
