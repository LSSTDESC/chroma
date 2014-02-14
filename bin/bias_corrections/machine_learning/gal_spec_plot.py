import sys
import os
from argparse import ArgumentParser

import numpy
import matplotlib.pyplot as plt
import astropy.utils.console as console

import _mypath
import chroma
import phot

def file_len(fname):
    """Count '\n's in file.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def spec_plot(infile, nmax=None, debug=False, randomize=True):
    ground_filters = phot.load_LSST_filters()
    space_filters = phot.load_Euclid_filters()
    filters = ground_filters.copy()
    for fname, f in space_filters.iteritems(): #Add `Euclid_filters` to `filters`
        if fname != 'norm':
            filters[fname] = f
    wave_match = filters['norm']['wave']
    filters = phot.match_filter_wavelengths(filters, wave_match)
    ground_filters = phot.match_filter_wavelengths(ground_filters, wave_match)
    space_filters = phot.match_filter_wavelengths(space_filters, wave_match)
    zps = phot.AB_zeropoints(filters)
    nrows = file_len(infile)
    if nmax is None:
        nmax = nrows
    if nmax > (nrows-1):
        nmax = nrows-1
    data = numpy.recarray((nmax,),
                          dtype = [('galTileID', numpy.uint64),
                                   ('objectID', numpy.uint64),
                                   ('raJ2000', numpy.float64),
                                   ('decJ2000', numpy.float64),
                                   ('redshift', numpy.float32),
                                   ('sedPathBulge', numpy.str_, 64),
                                   ('sedPathDisk', numpy.str_, 64),
                                   ('sedPathAGN', numpy.str_, 64),
                                   ('magNormBulge', numpy.float32),
                                   ('magNormDisk', numpy.float32),
                                   ('magNormAGN', numpy.float32),
                                   ('internalAVBulge', numpy.float32),
                                   ('internalRVBulge', numpy.float32),
                                   ('internalAVDisk', numpy.float32),
                                   ('internalRVDisk', numpy.float32)])

    order = [d+1 for d in xrange(nrows)]
    if randomize:
        import random
        random.shuffle(order)
    order = order[0:nmax]
    order.sort()
    with open(infile) as f:
        if not debug:
            outdev = sys.stdout
        else:
            outdev = open(os.devnull, 'w')
        with console.ProgressBar(nmax, file=outdev) as bar:
            j = 0
            for i, line in enumerate(f):
                if i == 0 : continue #ignore column labels row
                if j >= nmax : break
                if order[j] != i : continue
                bar.update()
                s = line.split(', ')
                data[j].galTileID = int(s[0])
                data[j].objectID = int(s[1])
                data[j].raJ2000 = float(s[2])
                data[j].decJ2000 = float(s[3])
                data[j].redshift = float(s[4])
                data[j].sedPathBulge = s[11]
                data[j].sedPathDisk = s[12]
                data[j].sedPathAGN = s[13]
                data[j].magNormBulge = float(s[14])
                data[j].magNormDisk = float(s[15])
                data[j].magNormAGN = float(s[16])
                data[j].internalAVBulge = float(s[17])
                data[j].internalRVBulge = float(s[18])
                data[j].internalAVDisk = float(s[19])
                data[j].internalRVDisk = float(s[20])
                spec = phot.make_composite_spec(data[j], filters, zps, wave_match)
                spec2 = phot.make_composite_spec_with_emission_lines(data[j], filters,
                                                                     zps, wave_match)
                f = plt.figure(figsize=(8,5))
                ax = f.add_subplot(111)
                w = spec['wave']
                p = spec['flambda']*w
                w2 = spec2['wave']
                p2 = spec2['flambda']*w2
                ax.plot(w, p)
                ax.plot(w2, p2)
                ax.set_xlim(500, 950)
                ylim = ax.get_ylim()
                ax.set_ylim(0, ylim[1])
                y0 = numpy.interp(700, w, p)
                for index in [-4, -2, 0, 2, 4, 6]:
                    k = y0/(700**index)
                    ys = k * (w ** index)
                    ax.plot(w, ys, label=str(index))

                ax.fill_between([550, 900], [ylim[1]*0.7]*2, [0,0], color='black', alpha=0.1)
                ax.fill_between([550, 685], [ylim[1]/2]*2, [0,0], color='blue', alpha=0.1)
                ax.fill_between([685, 815], [ylim[1]/2]*2, [0,0], color='red', alpha=0.1)
                f.savefig('output/{:04d}.png'.format(j), dpi=100)
                plt.close(f)
                j += 1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nmax', type=int, default=10,
                        help="maximum number of galaxies for which to plot spectra")
    parser.add_argument('--infile', default = 'output/galaxy_catalog.dat',
                        help="input filename")
    args = parser.parse_args()
    spec_plot(args.infile, nmax=args.nmax)
