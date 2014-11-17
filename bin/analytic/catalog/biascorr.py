import cPickle
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import treecorr as tc

def biascorr(args):
    gals = cPickle.load(open(args.galfile))
    gals = gals[gals['redshift'] <= args.zmax]
    gals = gals[gals['redshift'] >= args.zmin]
    stars = cPickle.load(open(args.starfile))
    # for now, just treat bias as scalar quantity
    ra = gals['raJ2000']
    dec = gals['decJ2000']

    if args.bias in ['S_m02', 'S_p06', 'S_p10']:
        stardata = stars[args.bias][args.band]
        galdata = gals[args.bias][args.band]
        starmean = np.mean(stardata)
        galdata = (galdata - starmean)/starmean
        ylabel = r"$\langle \frac{\Delta r^2}{r^2} \frac{\Delta r^2}{r^2}\rangle$"
        xlabel = r"separation (arcmin)"
        if args.corrected:
            stardata = ((stars[args.bias][args.band] - stars['photo_'+args.bias][args.band])
                        / stars['photo_'+args.bias][args.band])
            galdata = ((gals[args.bias][args.band] - gals['photo_'+args.bias][args.band])
                       / gals['photo_'+args.bias][args.band])
            ylabel = r"$\langle \delta\frac{\Delta r^2}{r^2} \delta\frac{\Delta r^2}{r^2}\rangle$"

    cat = tc.Catalog(ra=ra, dec=dec, k=galdata, ra_units='deg', dec_units='deg')
    kk = tc.KKCorrelation(bin_size=0.1, min_sep=0.5, max_sep=30, sep_units='arcmin')
    kk.process(cat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.exp(kk.logr), kk.xi, kk.varxi, c='r')
    ax.errorbar(np.exp(kk.logr), kk.xi, kk.varxi, c='r')
    ax.text(0.8, 0.92, r"{} < z < {}".format(args.zmin, args.zmax), transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=300)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('bias', default="Rbar", nargs='?',
                        help="""which chromatic bias to plot (Default: 'Rbar')
                             Other possibilities include: 'V', 'S_m02', 'S_p06', 'S_p10'""")
    parser.add_argument('--galfile', default = 'output/corrected_galaxy_data.pkl',
                        help="input galaxy file. Default 'output/corrected_galaxy_data.pkl'")
    parser.add_argument('--starfile', default = 'output/corrected_star_data.pkl',
                        help="input star file. Default 'output/corrected_star_data.pkl'")
    parser.add_argument('--band', default="LSST_r", nargs='?',
                        help="band of chromatic bias to plot (Default: 'LSST_r')")
    parser.add_argument('--corrected', action='store_true',
                        help="plot learning residuals instead of G5v residuals.")
    parser.add_argument('--outfile', default="output/biascorr.png", nargs='?',
                        help="output filename (Default: 'output/biascorr.png')")
    parser.add_argument('--zmin', default=0.0, type=float)
    parser.add_argument('--zmax', default=3.0, type=float)
    args = parser.parse_args()
    biascorr(args)
