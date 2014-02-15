from argparse import ArgumentParser

import numpy as np
from scipy.integrate import simps
import galsim
import lmfit

import matplotlib.pyplot as plt

import _mypath
import chroma

def moments(data, pixsize=1.0):
    '''Compute first and second (quadrupole) moments of `data`.  Scales result by `pixsize` for
    non-unit width pixels.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    '''
    xs, ys = np.meshgrid(np.arange(data.shape[0], dtype=np.float64) * pixsize,
                            np.arange(data.shape[0], dtype=np.float64) * pixsize)
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs-xbar)**2).sum() / total
    Iyy = (data * (ys-ybar)**2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return xbar, ybar, Ixx, Iyy, Ixy

def old_fiducial_galaxy():
    gparam = lmfit.Parameters()
    gparam.add('x0', value=0.1)
    gparam.add('y0', value=0.3)
    gparam.add('n', value=4.0, vary=False)
    gparam.add('r_e', value=1.0)
    gparam.add('flux', value=1.0, vary=False)
    gparam.add('gmag', value=0.2, min=0.0, max=1.0)
    gparam.add('phi', value=0.0)
    return gparam

def new_fiducial_galaxy():
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
    #old method
    s_engine = chroma.ImageEngine.GalSimSEngine(size=args.stamp_size)
    galtool = chroma.GalTools.SGalTool(s_engine)
    if args.GSAtmPSF2:
        PSF_model = chroma.PSF_model.GSAtmPSF2
    else:
        PSF_model = chroma.PSF_model.GSAtmPSF
    filter_file = args.datadir+args.filter
    gal_SED_file = args.datadir+args.galspec
    star_SED_file = args.datadir+args.starspec
    swave, sphotons = chroma.utils.get_photons(star_SED_file, filter_file, 0.0)
    sphotons /= simps(sphotons, swave)
    star_PSF = PSF_model(swave, sphotons, zenith=args.zenith_angle * np.pi/180.,
                         moffat_beta=args.PSF_beta, moffat_FWHM=args.PSF_FWHM/args.pixel_scale,
                         moffat_phi=args.PSF_phi, moffat_ellip=args.PSF_ellip,
                         pixel_scale=args.pixel_scale)
    smom = chroma.disp_moments(swave, sphotons, zenith=args.zenith_angle * np.pi/180.)
    old_gparam = old_fiducial_galaxy()
    old_gparam['n'].value = args.sersic_n
    old_gparam['x0'].value = args.gal_x0
    old_gparam['y0'].value = args.gal_y0
    old_gparam['gmag'].value = args.gal_ellip
    old_gparam['phi'].value = args.gal_phi
    old_gparam = galtool.set_uncvl_r2(old_gparam, (args.gal_r2/args.pixel_scale)**2)
    gamma = args.g1 + 1.0j * args.g2
    old_gparam = galtool.get_ring_params(old_gparam, gamma, 0.0)

    gwave, gphotons = chroma.utils.get_photons(gal_SED_file, filter_file, args.redshift)
    gphotons /= simps(gphotons, gwave)
    gal_PSF = PSF_model(gwave, gphotons, zenith=args.zenith_angle * np.pi/180.,
                        moffat_beta=args.PSF_beta, moffat_FWHM=args.PSF_FWHM/args.pixel_scale,
                        moffat_phi=args.PSF_phi, moffat_ellip=args.PSF_ellip,
                        pixel_scale=args.pixel_scale)
    old_image = s_engine.get_image(old_gparam, gal_PSF)

    #new method
    f_wave, f_throughput = np.genfromtxt(args.datadir+args.filter).T
    bandpass = galsim.Bandpass(f_wave, f_throughput)
    #bandpass.truncate(relative_throughput=0.002)
    g_wave, g_flambda = np.genfromtxt(args.datadir+args.galspec).T
    gal_SED = galsim.SED(wave=g_wave, flambda=g_flambda)
    gal_SED.setRedshift(args.redshift)
    s_wave, s_flambda = np.genfromtxt(args.datadir+args.starspec).T
    star_SED = galsim.SED(wave=s_wave, flambda=s_flambda)
    gal_SED.setFlux(bandpass, 1.0)
    star_SED.setFlux(bandpass, 1.0)
    PSF685 = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    PSF685.applyShear(g=args.PSF_ellip, beta=args.PSF_phi * galsim.radians)
    PSF = galsim.ChromaticAtmosphere(PSF685, base_wavelength=685.0,
                                     zenith_angle=args.zenith_angle * galsim.degrees,
                                     alpha=0.0)
    gtool = chroma.new_galtool.SersicTool(gal_SED, bandpass, PSF,
                                          args.stamp_size, args.pixel_scale)
    new_gparam = new_fiducial_galaxy()
    new_gparam['n'].value = args.sersic_n
    new_gparam['x0'].value = args.gal_x0 * args.pixel_scale
    new_gparam['y0'].value = args.gal_y0 * args.pixel_scale
    new_gparam['gmag'].value = args.gal_ellip
    new_gparam['phi'].value = args.gal_phi
    new_gparam = gtool.set_uncvl_r2(new_gparam, (args.gal_r2)**2)
    new_image = gtool.get_image2(new_gparam, ring_shear=galsim.Shear(g1=args.g1, g2=args.g2))

    rescale = (old_image * new_image.array).sum() / (new_image.array**2).sum()
    #new_image *= rescale

    old_mom = list(moments(old_image))
    old_e1 = (old_mom[2]-old_mom[3])/(old_mom[2]+old_mom[3])
    old_e2 = 2*old_mom[4]/(old_mom[2]+old_mom[3])
    new_mom = list(moments(new_image.array))
    new_e1 = (new_mom[2]-new_mom[3])/(new_mom[2]+new_mom[3])
    new_e2 = 2*new_mom[4]/(new_mom[2]+new_mom[3])
    old_mom.extend([old_e1, old_e2])
    new_mom.extend([new_e1, new_e2])
    old_flux = old_image.sum()
    new_flux = new_image.array.sum()
    old_mom.insert(0, old_flux)
    new_mom.insert(0, new_flux)
    diff_mom = [n-o for n, o in zip(new_mom, old_mom)]
    frac_mom = [d/o for d, o in zip(diff_mom, old_mom)]

    print '      {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}'.format(
        'flux', 'x0', 'y0', 'Ixx', 'Iyy', 'Ixy', 'e1', 'e2')
    fmt = '{:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'
    print ('old   '+fmt).format(*old_mom)
    print ('new   '+fmt).format(*new_mom)
    print ('diff  '+fmt).format(*diff_mom)
    print ('frac  '+fmt).format(*frac_mom)
    rms = np.sqrt(((old_image-new_image.array)**2).sum()/args.stamp_size**2)
    print 'rms image difference: {:12.8f}'.format(rms)

    if args.plot:
        f = plt.figure(figsize=(5,5))
        ax = f.add_subplot(221)
        im1 = ax.imshow(old_image)
        plt.colorbar(im1)
        ax = f.add_subplot(222)
        im2 = ax.imshow(new_image.array)
        plt.colorbar(im2)
        ax = f.add_subplot(223)
        im3 = ax.imshow(old_image - new_image.array)
        plt.colorbar(im3)
        ax = f.add_subplot(224)
        im4 = ax.imshow(np.log10(abs(old_image - new_image.array)/old_image))
        plt.colorbar(im4)
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--starspec', default='SEDs/ukg5v.ascii',
                        help="stellar spectrum to use when fitting (Default 'SEDs/ukg5v.ascii')")
    parser.add_argument('-g', '--galspec', default='SEDs/CWW_E_ext.ascii',
                        help="galactic spectrum used to create target image " +
                             "(Default 'SEDs/CWW_E_ext.ascii')")
    parser.add_argument('-z', '--redshift', type=float, default=0.0,
                        help="galaxy redshift (Default 0.0)")
    parser.add_argument('-f', '--filter', default='filters/LSST_r.dat',
                        help="filter for simulation (Default 'filters/LSST_r.dat')")
    parser.add_argument('--zenith_angle', default=45.0, type=float,
                        help="zenith angle in degrees for differential chromatic refraction " +
                             "computation (Default 45.0)")
    parser.add_argument('--datadir', default='../../../data/',
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
    parser.add_argument('--GSAtmPSF2', action='store_true')

    args = parser.parse_args()

    image_comparison(args)
