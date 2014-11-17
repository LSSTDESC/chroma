import cPickle
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import treecorr as tc

def corr(args, gals_S, dV, ra, dec):
    # chromatic seeing
    dI_xx = args.PSF_r2**2/2.0 * gals_S
    dI_yy = args.PSF_r2**2/2.0 * gals_S
    # DCR
    dI_xx += dV

    m = -(dI_xx + dI_yy) / args.gal_r2**2
    c1 = (dI_xx - dI_yy) / (2.0 * args.gal_r2**2)
    c2 = np.zeros_like(c1)

    mcat = tc.Catalog(ra=ra, dec=dec, g1=m, g2=m, ra_units='deg', dec_units='deg')
    mm = tc.GGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    mm.process(mcat)

    ccat = tc.Catalog(ra=ra, dec=dec, g1=c1, g2=c2, ra_units='deg', dec_units='deg')
    cc = tc.GGCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    cc.process(ccat)

    return mm, cc

def calibcorr(args):
    gals = cPickle.load(open(args.galfile))
    gals = gals[gals['redshift'] <= args.zmax]
    gals = gals[gals['redshift'] >= args.zmin]
    stars = cPickle.load(open(args.starfile))
    ra = gals['raJ2000']
    dec = gals['decJ2000']

    #######################
    # uncorrected version #
    #######################

    stars_S = stars['S_m02'][args.band]
    gals_S = gals['S_m02'][args.band]
    star_Smean = np.mean(stars_S)
    gals_S = (gals_S - star_Smean)/star_Smean

    stars_V = stars['V'][args.band]
    gals_V = gals['V'][args.band]
    star_Vmean = np.mean(stars_V)
    dV = (gals_V - star_Vmean) * (3600 * 180 / np.pi)**2

    mm, cc = corr(args, gals_S, dV, ra, dec)

    # multiplicative bias plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.exp(mm.logr), mm.xip, np.sqrt(mm.varxi), c='r')
    ax.text(0.8, 0.92, r"{} < z < {}".format(args.zmin, args.zmax), transform=ax.transAxes)
    ax.set_xlabel(r"separation (arcmin)")
    ax.set_ylabel(r"$\xi_m$")
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig("output/xi_m.png", dpi=300)

    # additive bias plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.exp(cc.logr), cc.xip, np.sqrt(cc.varxi), c='r')
    ax.text(0.8, 0.92, r"{} < z < {}".format(args.zmin, args.zmax), transform=ax.transAxes)
    ax.set_xlabel(r"separation (arcmin)")
    ax.set_ylabel(r"$\xi_c$")
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig("output/xi_c.png", dpi=300)

    #####################
    # corrected version #
    #####################

    gals_S = ((gals['S_m02'][args.band] - gals['photo_S_m02'][args.band])
              / gals['photo_S_m02'][args.band])
    dV = (gals['V'][args.band] - gals['photo_V'][args.band]) * (3600 * 180./np.pi)**2

    mm, cc = corr(args, gals_S, dV, ra, dec)

    # multiplicative bias plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.exp(mm.logr), mm.xip, np.sqrt(mm.varxi), c='r')
    ax.text(0.8, 0.92, r"{} < z < {}".format(args.zmin, args.zmax), transform=ax.transAxes)
    ax.set_xlabel(r"separation (arcmin)")
    ax.set_ylabel(r"$\xi_m$")
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig("output/xi_m_corrected.png", dpi=300)

    # additive bias plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.exp(cc.logr), cc.xip, np.sqrt(cc.varxi), c='r')
    ax.text(0.8, 0.92, r"{} < z < {}".format(args.zmin, args.zmax), transform=ax.transAxes)
    ax.set_xlabel(r"separation (arcmin)")
    ax.set_ylabel(r"$\xi_c$")
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig("output/xi_c_corrected.png", dpi=300)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = 'output/corrected_galaxy_data.pkl',
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = 'output/corrected_star_data.pkl',
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('--band', default="LSST_r", nargs='?',
                        help="band of chromatic bias to plot (Default: 'LSST_r')")
    parser.add_argument('--zmin', default=0.2, type=float,
                        help="minimum of redshift bin")
    parser.add_argument('--zmax', default=0.3, type=float,
                        help="maximum of redshift bin")
    parser.add_argument('--zenith_angle', default=30.0, type=float,
                        help="zenith angle for DCR in degreeds (default: 30 degrees)")
    parser.add_argument('--PSF_r2', default=0.7, type=float,
                        help="PSF root second moment radius sqrt(r^2) (default: 0.7 arcsec)")
    parser.add_argument('--gal_r2', default=0.3, type=float,
                        help="Galaxy root second moment radius sqrt(r^2) (default: 0.3 arcsec)")
    args = parser.parse_args()

    calibcorr(args)
