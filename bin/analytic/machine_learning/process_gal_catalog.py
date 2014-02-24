"""Process galaxy catalog produced by make_catalogs.py to add columns for
DCR biases, chromatic seeing biases, and chromatic diffraction limit biases.
"""

import os
import sys
import cPickle
from argparse import ArgumentParser

import numpy
from scipy.interpolate import interp1d

import _mypath
import chroma

datadir = '../../../data/'

def file_len(fname):
    """Count '\n's in file.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def composite_spectrum(gal, norm_bandpass):
    SED_dir = os.environ['CAT_SHARE_DATA'] + 'data/'
    if gal['sedPathBulge'] != 'None':
        bulge_SED = chroma.SampledSED(SED_dir+gal['sedPathBulge'])
        bulge_SED = bulge_SED.createWithMagnitude(norm_bandpass, gal['magNormBulge'])
        bulge_SED = bulge_SED.createExtincted(A_v=gal['internalAVBulge'],
                                              R_v=gal['internalRVBulge'])
        bulge_SED = bulge_SED.createRedshifted(gal['redshift'])
        SED = bulge_SED
    if gal['sedPathDisk'] != 'None':
        disk_SED = chroma.SampledSED(SED_dir+gal['sedPathDisk'])
        disk_SED = disk_SED.createWithMagnitude(norm_bandpass, gal['magNormDisk'])
        disk_SED = disk_SED.createExtincted(A_v=gal['internalAVDisk'],
                                              R_v=gal['internalRVDisk'])
        disk_SED = disk_SED.createRedshifted(gal['redshift'])
        if 'SED' in locals():
            SED += disk_SED
        else:
            SED = disk_SED
    if gal['sedPathAGN'] != 'None':
        AGN_SED = chroma.SampledSED(SED_dir+gal['sedPathAGN'])
        AGN_SED = AGN_SED.createWithMagnitude(norm_bandpass, gal['magNormAGN'])
        AGN_SED = AGN_SED.createRedshifted(gal['redshift'])
        if 'SED' in locals():
            SED += AGN_SED
        else:
            SED = AGN_SED

    return SED

def process_gal_file(filename, nmax=None, debug=False, randomize=True, emission=False, start=0):
    filters = {}
    for f in 'ugrizy':
        ffile = datadir+'filters/LSST_{}.dat'.format(f)
        filters['LSST_{}'.format(f)] = chroma.SampledBandpass(ffile).createThinned(10)
    for width in [150,250,350,450]:
        ffile = datadir+'filters/Euclid_{}.dat'.format(width)
        filters['Euclid_{}'.format(width)] = chroma.SampledBandpass(ffile).createThinned(10)
    filters['norm'] = chroma.SampledBandpass(interp1d([499, 500, 501], [0, 1, 0]))

    nrows = file_len(filename)
    if nmax is None:
        nmax = nrows
    if nmax > (nrows-1):
        nmax = nrows-1
    ugrizy = [('LSST_u', numpy.float32),
              ('LSST_g', numpy.float32),
              ('LSST_r', numpy.float32),
              ('LSST_i', numpy.float32),
              ('LSST_z', numpy.float32),
              ('LSST_y', numpy.float32)]
    ugrizyE = [('LSST_u', numpy.float32),
               ('LSST_g', numpy.float32),
               ('LSST_r', numpy.float32),
               ('LSST_i', numpy.float32),
               ('LSST_z', numpy.float32),
               ('LSST_y', numpy.float32),
               ('Euclid_150', numpy.float32),
               ('Euclid_250', numpy.float32),
               ('Euclid_350', numpy.float32),
               ('Euclid_450', numpy.float32)]
    E = [('Euclid_150', numpy.float32),
         ('Euclid_250', numpy.float32),
         ('Euclid_350', numpy.float32),
         ('Euclid_450', numpy.float32)]

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
                                   ('internalRVDisk', numpy.float32),
                                   ('mag', ugrizy),
                                   ('magCalc', ugrizyE),
                                   ('R', ugrizy),
                                   ('V', ugrizy),
                                   ('dVcg', ugrizy),
                                   ('S_m02', ugrizy),
                                   ('S_p06', E),
                                   ('S_p10', E)])

    order = [d+1 for d in xrange(nrows)]
    if randomize:
        import random
        random.seed(123456789)
        random.shuffle(order)
    order = order[start:start+nmax]
    order.sort()
    with open(filename) as f:
        if not debug:
            outdev = sys.stdout
        else:
            outdev = open(os.devnull, 'w')
        with chroma.ProgressBar(nmax, file=outdev) as bar:
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

                spec = composite_spectrum(data[j], filters['norm'])
                for k, f in enumerate('ugrizy'):
                    # grab catalog magnitude
                    data[j]['mag']['LSST_'+f] = float(s[5+k])
                    bp = filters['LSST_'+f] # for brevity
                    try:
                        data[j]['magCalc']['LSST_'+f] = spec.getMagnitude(bp)
                        dcr = spec.getDCRMomentShifts(bp, numpy.pi/4)
                        data[j]['R']['LSST_'+f] = dcr[0]
                        data[j]['V']['LSST_'+f] = dcr[1]
                        data[j]['S_m02']['LSST_'+f] = spec.getSeeingShift(bp, alpha=-0.2)
                    except:
                        data[j]['magCalc']['LSST_'+f] = numpy.nan
                        data[j]['R']['LSST_'+f] = numpy.nan
                        data[j]['V']['LSST_'+f] = numpy.nan
                        data[j]['S_m02']['LSST_'+f] = numpy.nan
                for fw in [150, 250, 350, 450]:
                    fname = 'Euclid_{}'.format(fw)
                    bp = filters[fname]
                    try:
                        data[j]['magCalc'][fname] = spec.getMagnitude(bp)
                        data[j]['S_p06'][fname] = spec.getSeeingShift(bp, alpha=0.6)
                        data[j]['S_p10'][fname] = spec.getSeeingShift(bp, alpha=1.0)
                    except:
                        data[j]['magCalc'][fname] = numpy.nan
                        data[j]['S_p06'][fname] = numpy.nan
                        data[j]['S_p10'][fname] = numpy.nan

                # if emission:
                #     spec = phot.make_composite_spec_with_emission_lines(data[j], filters,
                #                                                         zps, wave_match)
                # else:
                #     spec = phot.make_composite_spec(data[j], filters, zps, wave_match)
                if debug:
                    print
                    print 'syn mag:' + ' '.join(['{:6.3f}'.format(data[j]['magCalc']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'cat mag:' + ' '.join(['{:6.3f}'.format(data[j]['mag']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'Euclid: ' + ' '.join(['{:6.3f}'.format(data[j]['magCalc']['Euclid_{}'.format(fw)])
                                                 for fw in [150, 250, 350, 450]])
                j += 1
    return data

def runme():
    junk = process_gal_file('output/galaxy_catalog.dat', nmax=25, emission=False, start=0, debug=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--infile', default = 'output/galaxy_catalog.dat',
                        help="input filename. Default 'output/galaxy_catalog.dat'")
    parser.add_argument('--outfile', default = 'output/galaxy_data.pkl',
                        help="output filename. Default 'output/galaxy_data.pkl'")
    parser.add_argument('--nmax', type=int, default=30000,
                        help="maximum number of galaxies to process. Default 30000")
    parser.add_argument('--start', type=int, default=0,
                        help="starting index for catalog.  Default 0")
    parser.add_argument('--emission', action='store_true',
                        help="add emission lines to spectra")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cPickle.dump(process_gal_file(args.infile,
                                  nmax=args.nmax,
                                  emission=args.emission,
                                  start=args.start,
                                  debug=args.debug),
                 open(args.outfile, 'wb'))
