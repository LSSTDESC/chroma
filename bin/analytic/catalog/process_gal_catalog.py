"""Process galaxy catalog produced by make_catalogs.py to add columns for DCR biases, chromatic
seeing biases, and chromatic diffraction limit biases.  This script requires that the LSST CatSim
SED files are downloaded and that either the environment variable $CAT_SHARE_DATA (for older versions
of the LSST DM stack) or SIMS_SED_LIBRARY_DIR (for the current version of the stack) points to them.
Note that you might need to source the `loadLSST.sh` file and run `setup sims_sed_library` to get
these paths to work for the current version of the lsst stack.

Chromatic biases include:
  Rbar - centroid shift due to differential chromatic refraction.
  V - zenith-direction second moment shift due to differential chromatic refraction
  S - shift in "size" of the PSF due to a power-law dependence of the FWHM with wavelength:
      FWHM \propto \lambda^{\alpha}.  S = the second moment square radius r^2 = Ixx + Iyy.
      Three cases are tabulated:
        \alpha = -0.2 : appropriate for atmospheric chromatic seeing.  denoted 'S_m02'
        \alpha = 1.0 : appropriate for a pure diffraction limited PSF.  denoted 'S_p10'
        \alpha = 0.6 : appropriate for Euclid (see Voigt+12 or Cypriano+10).  denoted 'S_p06'
"""

import os
import sys
import cPickle
from argparse import ArgumentParser

import numpy as np
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

def composite_spectrum(gal, norm_bandpass, emission=False):
    if 'CAT_SHARE_DATA' in os.environ:
        SED_dir = os.environ['CAT_SHARE_DATA'] + 'data/'
    elif 'SIMS_SED_LIBRARY_DIR' in os.environ:
        SED_dir = os.environ['SIMS_SED_LIBRARY_DIR']
    else:
        raise ValueError("Cannot find CatSim SED files.")
    if gal['sedPathBulge'] != 'None':
        bulge_SED = chroma.SED(SED_dir+gal['sedPathBulge'])
        bulge_SED = bulge_SED.withMagnitude(gal['magNormBulge'], norm_bandpass)
        if emission:
            bulge_SED = bulge_SED.addEmissionLines()
        bulge_SED = bulge_SED.redden(A_v=gal['internalAVBulge'],
                                     R_v=gal['internalRVBulge'])
        bulge_SED = bulge_SED.atRedshift(gal['redshift'])
        SED = bulge_SED
    if gal['sedPathDisk'] != 'None':
        disk_SED = chroma.SED(SED_dir+gal['sedPathDisk'])
        disk_SED = disk_SED.withMagnitude(gal['magNormDisk'], norm_bandpass)
        if emission:
            disk_SED = disk_SED.addEmissionLines()
        disk_SED = disk_SED.redden(A_v=gal['internalAVDisk'],
                                   R_v=gal['internalRVDisk'])
        disk_SED = disk_SED.atRedshift(gal['redshift'])
        if 'SED' in locals():
            SED += disk_SED
        else:
            SED = disk_SED
    if gal['sedPathAGN'] != 'None':
        AGN_SED = chroma.SED(SED_dir+gal['sedPathAGN'])
        AGN_SED = AGN_SED.withMagnitude(gal['magNormAGN'], norm_bandpass)
        AGN_SED = AGN_SED.atRedshift(gal['redshift'])
        if 'SED' in locals():
            SED += AGN_SED
        else:
            SED = AGN_SED

    return SED

def process_gal_file(filename, nmax=None, debug=False, randomize=True, emission=False, start=0):
    filters = {}
    for f in 'ugrizy':
        ffile = datadir+'filters/LSST_{}.dat'.format(f)
        filters['LSST_{}'.format(f)] = (chroma.Bandpass(ffile)
                                        .thin(1.e-5)
                                        .withZeropoint('AB', effective_diameter=6.4, exptime=30.0))
    for width in [150,250,350,450]:
        ffile = datadir+'filters/Euclid_{}.dat'.format(width)
        filters['Euclid_{}'.format(width)] = (chroma.Bandpass(ffile)
                                              .thin(1.e-5)
                                              .withZeropoint('AB',
                                                             effective_diameter=6.4,
                                                             exptime=30.0))
    filters['norm'] = (chroma.Bandpass(interp1d([499, 500, 501], [0, 1, 0]))
                       .withZeropoint('AB', effective_diameter=6.4, exptime=30.0))

    nrows = file_len(filename)
    if nmax is None:
        nmax = nrows
    if nmax > (nrows-1):
        nmax = nrows-1
    ugrizy = [('LSST_u', np.float32),
              ('LSST_g', np.float32),
              ('LSST_r', np.float32),
              ('LSST_i', np.float32),
              ('LSST_z', np.float32),
              ('LSST_y', np.float32)]
    ugrizyE = [('LSST_u', np.float32),
               ('LSST_g', np.float32),
               ('LSST_r', np.float32),
               ('LSST_i', np.float32),
               ('LSST_z', np.float32),
               ('LSST_y', np.float32),
               ('Euclid_150', np.float32),
               ('Euclid_250', np.float32),
               ('Euclid_350', np.float32),
               ('Euclid_450', np.float32)]
    E = [('Euclid_150', np.float32),
         ('Euclid_250', np.float32),
         ('Euclid_350', np.float32),
         ('Euclid_450', np.float32)]

    data = np.recarray((nmax,),
                       dtype = [('galTileID', np.uint64),
                                ('objectID', np.uint64),
                                ('raJ2000', np.float64),
                                ('decJ2000', np.float64),
                                ('redshift', np.float32),
                                ('sedPathBulge', np.str_, 64),
                                ('sedPathDisk', np.str_, 64),
                                ('sedPathAGN', np.str_, 64),
                                ('magNormBulge', np.float32),
                                ('magNormDisk', np.float32),
                                ('magNormAGN', np.float32),
                                ('internalAVBulge', np.float32),
                                ('internalRVBulge', np.float32),
                                ('internalAVDisk', np.float32),
                                ('internalRVDisk', np.float32),
                                ('mag', ugrizy),
                                ('magCalc', ugrizyE),
                                ('Rbar', ugrizy),
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

                spec = composite_spectrum(data[j], filters['norm'], emission=emission)
                for k, f in enumerate('ugrizy'):
                    # grab catalog magnitude
                    data[j]['mag']['LSST_'+f] = float(s[5+k])
                    bp = filters['LSST_'+f] # for brevity
                    try:
                        data[j]['magCalc']['LSST_'+f] = spec.getMagnitude(bp)
                        dcr = spec.getDCRMomentShifts(bp, np.pi/4)
                        data[j]['Rbar']['LSST_'+f] = dcr[0]
                        data[j]['V']['LSST_'+f] = dcr[1]
                        data[j]['S_m02']['LSST_'+f] = spec.getSeeingShift(bp, alpha=-0.2)
                    except:
                        data[j]['magCalc']['LSST_'+f] = np.nan
                        data[j]['Rbar']['LSST_'+f] = np.nan
                        data[j]['V']['LSST_'+f] = np.nan
                        data[j]['S_m02']['LSST_'+f] = np.nan
                for fw in [150, 250, 350, 450]:
                    fname = 'Euclid_{}'.format(fw)
                    bp = filters[fname]
                    try:
                        data[j]['magCalc'][fname] = spec.getMagnitude(bp)
                        data[j]['S_p06'][fname] = spec.getSeeingShift(bp, alpha=0.6)
                        data[j]['S_p10'][fname] = spec.getSeeingShift(bp, alpha=1.0)
                    except:
                        data[j]['magCalc'][fname] = np.nan
                        data[j]['S_p06'][fname] = np.nan
                        data[j]['S_p10'][fname] = np.nan

                if debug:
                    print
                    print 'syn mag:' + ' '.join(['{:6.3f}'.format(data[j]['magCalc']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'cat mag:' + ' '.join(['{:6.3f}'.format(data[j]['mag']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'Euclid: ' + ' '.join(['{:6.3f}'.format(data[j]['magCalc']
                                                                  ['Euclid_{}'.format(fw)])
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
