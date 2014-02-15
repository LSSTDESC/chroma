from argparse import ArgumentParser

import numpy as np
from scipy.integrate import simps
import galsim
import lmfit

import matplotlib.pyplot as plt

import _mypath
import chroma

def fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('hlr', value=0.27)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)
    return gparam

def image_comparison(args):
    # slow method
    bandpass = galsim.Bandpass(args.datadir+args.filter)
    bandpass = bandpass.thin(100)

    gal_SED = galsim.SED(args.datadir+args.spec)
    gal_SED.setRedshift(args.redshift)
    gal_SED.setFlux(bandpass, 1.0)

    PSF685 = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    PSF685.applyShear(g=args.PSF_ellip, beta=args.PSF_phi * galsim.radians)
    PSF = galsim.ChromaticAtmosphere(PSF685, base_wavelength=685.0,
                                     zenith_angle=args.zenith_angle * galsim.degrees,
                                     alpha=0.0)
    gtool = chroma.SersicTool(gal_SED, bandpass, PSF, args.stamp_size, args.pixel_scale)

    gparam = fiducial_galaxy()
    gparam['n'].value = args.sersic_n
    gparam['x0'].value = args.gal_x0
    gparam['y0'].value = args.gal_y0
    gparam['gmag'].value = args.gal_ellip
    gparam['phi'].value = args.gal_phi
    gparam = gtool.set_uncvl_r2(gparam, (args.gal_r2)**2)
    image = gtool.get_image(gparam, ring_shear=galsim.Shear(g1=args.g1, g2=args.g2))


    # fast method
    fast_gtool = chroma.SersicFastTool(gal_SED, bandpass, PSF, args.stamp_size, args.pixel_scale)
    fast_image = fast_gtool.get_image(gparam, ring_shear=galsim.Shear(g1=args.g1, g2=args.g2))

    # rescale = (old_image * new_image.array).sum() / (new_image.array**2).sum()
    # new_image *= rescale

    mom = list(chroma.moments(image))
    e1 = (mom[2]-mom[3])/(mom[2]+mom[3])
    e2 = 2*mom[4]/(mom[2]+mom[3])
    mom.extend([e1, e2])
    mom.insert(0, image.array.sum())

    fast_mom = list(chroma.moments(fast_image))
    fast_e1 = (fast_mom[2]-fast_mom[3])/(fast_mom[2]+fast_mom[3])
    fast_e2 = 2*fast_mom[4]/(fast_mom[2]+fast_mom[3])
    fast_mom.extend([fast_e1, fast_e2])
    fast_mom.insert(0, fast_image.array.sum())

    diff_mom = [n-o for n, o in zip(fast_mom, mom)]
    frac_mom = [d/o for d, o in zip(diff_mom, mom)]

    print '      {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}'.format(
        'flux', 'x0', 'y0', 'Ixx', 'Iyy', 'Ixy', 'e1', 'e2')
    fmt = '{:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'
    print ('old   '+fmt).format(*mom)
    print ('new   '+fmt).format(*fast_mom)
    print ('diff  '+fmt).format(*diff_mom)
    print ('frac  '+fmt).format(*frac_mom)
    rms = np.sqrt(((image.array-fast_image.array)**2).sum()/args.stamp_size**2)
    print 'rms image difference: {:12.8f}'.format(rms)

    if args.plot:
        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(221)
        ax.set_title('SersicTool')
        im1 = chroma.my_imshow(image.array, ax)
        plt.colorbar(im1)
        ax = f.add_subplot(222)
        ax.set_title('FastSersicTool')
        im2 = chroma.my_imshow(fast_image.array, ax)
        plt.colorbar(im2)
        ax = f.add_subplot(223)
        ax.set_title('Residual')
        im3 = chroma.my_imshow(image.array - fast_image.array, ax)
        plt.colorbar(im3)
        ax = f.add_subplot(224)
        ax.set_title('log10(abs(Fractional Residual))')
        im4 = chroma.my_imshow(np.log10(abs(image.array - fast_image.array)/image.array), ax)
        plt.colorbar(im4)
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--spec', default='SEDs/CWW_E_ext.ascii',
                        help="spectrum used to create target image " +
                             "(Default 'SEDs/CWW_E_ext.ascii')")
    parser.add_argument('-z', '--redshift', type=float, default=0.0,
                        help="galaxy redshift (Default 0.0)")
    parser.add_argument('-f', '--filter', default='filters/LSST_r.dat',
                        help="filter for simulation (Default 'filters/LSST_r.dat')")
    parser.add_argument('--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('--datadir', default='../data/',
                        help="directory to find SED and filter files.")
    parser.add_argument('--PSF_beta', type=float, default=2.5,
                        help="Set beta parameter of PSF Moffat profile. (Default 2.5)")
    parser.add_argument('--PSF_FWHM', type=float, default=0.7,
                        help="Set FWHM of PSF in arcsec (Default 0.7).")
    parser.add_argument('--PSF_phi', type=float, default=0.0,
                        help="Set position angle of PSF in radians (Default 0.0).")
    parser.add_argument('--PSF_ellip', type=float, default=0.0,
                        help="Set ellipticity of PSF (Default 0.0)")
    parser.add_argument('-n', '--sersic_n', type=float, default=0.5,
                        help='Sersic index (Default 0.5)')
    parser.add_argument('--gal_ellip', type=float, default=0.3,
                        help="Set ellipticity of galaxy (Default 0.3)")
    parser.add_argument('--gal_phi', type=float, default=0.0,
                        help="Set position angle of galaxy (Default 0.0)")
    parser.add_argument('--gal_x0', type=float, default=0.0,
                        help="Set galaxy center x-offset in pixels (Default 0.0)")
    parser.add_argument('--gal_y0', type=float, default=0.0,
                        help="Set galaxy center y-offset in pixels (Default 0.0)")
    parser.add_argument('--gal_r2', type=float, default=0.27,
                        help="Set galaxy second moment radius sqrt(r^2) in arcsec (Default 0.27)")
    parser.add_argument('--nring', type=int, default=3,
                        help="Set number of angles in ring test (Default 3)")
    parser.add_argument('--pixel_scale', type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument('--stamp_size', type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")
    parser.add_argument('--g1', type=float, default=0.0,
                        help="Apply '+' shear g1 to galaxy")
    parser.add_argument('--g2', type=float, default=0.0,
                        help="Apply 'x' shear g2 to galaxy")
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    image_comparison(args)
