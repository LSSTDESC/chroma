"""An exposure time calculator for LSST.  Uses GalSim to draw a galaxy with specified magnitude,
shape, etc, and then uses the same image as the optimal weight function.  Derived from D. Kirkby's
notes on deblending.
"""

import numpy as np

import galsim

# Some constants
pixel_scale = 0.2 # arcsec
# LSST effective area in meters^2
A = 6.4**2 * np.pi
# zeropoints from DK notes in photons per second per pixel
# should eventually compute these on the fly from filter throughput functions.
s0 = A * np.r_[0.732, 2.124, 1.681, 1.249, 0.862, 0.452]
# Sky brightnesses in AB mag / arcsec^2.
# stole these from http://www.lsst.org/files/docs/gee_137.28.pdf
B = np.r_[22.8, 22.2, 21.3, 20.3, 19.1, 18.1]
# number of visits
# From http://www.lsst.org/files/docs/137.03_Pinto_Cadence_Design_8x10.pdf
nvisits = np.r_[56, 80, 184, 184, 160, 160]
# exposure time per visit
visit_time = 30.0
# Sky brightness per arcsec^2 per second
sbar = s0 * 10**(-0.4*(B-24.0))
banddict = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'y':5}

class ETC(object):
    def __init__(self, profile, pixel_scale=0.2, stamp_size=15):
        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.profile = profile

    def SNR(self, mag, band='r'):
        img = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        iband = banddict[band]
        exptime = nvisits[iband] * visit_time
        flux = s0[iband] * 10**(-0.4*(mag - 24.0)) * exptime
        self.profile.setFlux(flux)
        self.profile.drawImage(image=img)
        imgsqr = img.array**2
        signal = imgsqr.sum()
        noise = np.sqrt((imgsqr * sbar[iband] * exptime * self.pixel_scale**2).sum())
        return signal / noise

    def err(self, mag, band='r'):
        snr = self.SNR(mag, band)
        return 2.5 / np.log(10) / snr


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Filter
    parser.add_argument("--band", default='i',
                        help="band for simulation (Default 'i')")

    # PSF structural arguments
    PSF_profile = parser.add_mutually_exclusive_group()
    PSF_profile.add_argument("--kolmogorov", action="store_true",
                             help="Use Kolmogorov PSF (Default Gaussian)")
    PSF_profile.add_argument("--moffat", action="store_true",
                             help="Use Moffat PSF (Default Gaussian)")
    parser.add_argument("--PSF_beta", type=float, default=3.0,
                        help="Set beta parameter of Moffat profile PSF. (Default 2.5)")
    parser.add_argument("--PSF_FWHM", type=float, default=0.7,
                        help="Set FWHM of PSF in arcsec (Default 0.7).")
    parser.add_argument("--PSF_phi", type=float, default=0.0,
                        help="Set position angle of PSF in degrees (Default 0.0).")
    parser.add_argument("--PSF_ellip", type=float, default=0.0,
                        help="Set ellipticity of PSF (Default 0.0)")

    # Galaxy structural arguments
    parser.add_argument("-n", "--sersic_n", type=float, default=0.5,
                        help="Sersic index (Default 0.5)")
    parser.add_argument("--gal_ellip", type=float, default=0.3,
                        help="Set ellipticity of galaxy (Default 0.3)")
    parser.add_argument("--gal_phi", type=float, default=0.0,
                        help="Set position angle of galaxy in radians (Default 0.0)")
    parser.add_argument("--gal_HLR", type=float, default=0.5,
                        help="Set galaxy half-light-radius. (default 0.5 arcsec)")

    # Simulation input arguments
    parser.add_argument("--pixel_scale", type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument("--stamp_size", type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")

    # Magnitude!
    parser.add_argument("--mag", type=float, default=25.3,
                        help="magnitude of galaxy")

    args = parser.parse_args()

    if args.kolmogorov:
        psf = galsim.Kolmogorov(fwhm=args.PSF_FWHM)
    elif args.moffat:
        psf = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    else:
        psf = galsim.Gaussian(fwhm=args.PSF_FWHM)
    psf = psf.shear(e=args.PSF_ellip, beta=args.PSF_phi*galsim.radians)

    gal = galsim.Sersic(n=args.sersic_n, half_light_radius=args.gal_HLR)
    gal = gal.shear(e=args.gal_ellip, beta=args.gal_phi*galsim.radians)

    profile = galsim.Convolve(psf, gal)

    etc = ETC(profile, pixel_scale=args.pixel_scale, stamp_size=args.stamp_size)

    print "input magnitude: {}".format(args.mag)
    print "output SNR: {}".format(etc.SNR(args.mag, args.band))
    print "output mag err: {}".format(etc.err(args.mag, args.band))
