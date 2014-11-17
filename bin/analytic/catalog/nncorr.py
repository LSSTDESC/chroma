import cPickle
from argparse import ArgumentParser

import numpy as np
import treecorr as tc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def nncorr(args):
    gals = cPickle.load(open(args.galfile))
    gals = gals[gals['redshift'] <= args.zmax]
    gals = gals[gals['redshift'] >= args.zmin]

    ri = gals['mag']['LSST_r'] - gals['mag']['LSST_i']

    # blue galaxies
    median_ri = np.median(ri)
    blue_gals = gals[ri < median_ri]
    ra = blue_gals['raJ2000']
    dec = blue_gals['decJ2000']
    data = tc.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
    random_x = np.random.rand(500000)*2 + 199
    random_y = np.random.rand(500000)*2 + (-11)
    random_r2 = (random_x-200)**2 + (random_y+10)**2
    w = random_r2 < 1.0
    if w.sum() < 250000:
        import sys; sys.exit()
    rand = tc.Catalog(ra=random_x[w[0:250000]], dec=random_y[w[0:250000]],
                      ra_units='deg', dec_units='deg')
    dd = tc.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    dr = tc.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    rr = tc.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    dd.process(data)
    dr.process(data, rand)
    rr.process(rand)
    blue_xi, blue_varxi = dd.calculateXi(rr, dr)

    # red galaxies
    median_ri = np.median(ri)
    red_gals = gals[ri > median_ri]
    ra = red_gals['raJ2000']
    dec = red_gals['decJ2000']
    data = tc.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
    dd = tc.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    dr = tc.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    rr = tc.NNCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    dd.process(data)
    dr.process(data, rand)
    rr.process(rand)
    red_xi, red_varxi = dd.calculateXi(rr, dr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.exp(dd.logr), red_xi, red_varxi, c='r')
    ax.errorbar(np.exp(dd.logr), blue_xi, blue_varxi, c='b')
    ax.text(0.8, 0.92, r"{} < z < {}".format(args.zmin, args.zmax), transform=ax.transAxes)
    ax.set_xlabel(r"separation (arcmin)")
    ax.set_ylabel(r"$\xi$")
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig("output/nn_xi.png", dpi=300)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--galfile', default = 'output/corrected_galaxy_data.pkl',
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--zmin', default=0.2, type=float,
                        help="minimum of redshift bin")
    parser.add_argument('--zmax', default=0.3, type=float,
                        help="maximum of redshift bin")
    args = parser.parse_args()

    nncorr(args)
