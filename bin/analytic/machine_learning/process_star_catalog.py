"""Process star catalog produced by make_catalogs.py to add columns for DCR biases, chromatic
seeing biases, and chromatic diffraction limit biases.  This script requires that the LSST CatSim
SED files are downloaded and that the environment variable $CAT_SHARE_DATA points to them.

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

import sys
import os
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

def stellar_spectrum(star, norm_bandpass):
    SED_dir = os.environ['CAT_SHARE_DATA'] + 'data/'
    SED = chroma.SampledSED(SED_dir+star['sedFilePath'])
    SED = SED.createWithMagnitude(norm_bandpass, star['magNorm'])
    SED = SED.createExtincted(A_v=star['galacticAv'])
    return SED

def process_star_file(filename, nmax=None, debug=False, randomize=True, start=0):
    filters = {}
    for f in 'ugrizy':
        ffile = datadir+'filters/LSST_{}.dat'.format(f)
        filters['LSST_{}'.format(f)] = chroma.SampledBandpass(ffile).createThinned(10) #thin for speed
    for width in [150,250,350,450]:
        ffile = datadir+'filters/Euclid_{}.dat'.format(width)
        filters['Euclid_{}'.format(width)] = chroma.SampledBandpass(ffile).createThinned(10)
    # LSST SED catalog entries are normalized by their AB magnitude at 500 nm.  So define a narrow
    # filter at 500nm to use for normalization.
    filters['norm'] = chroma.SampledBandpass(interp1d([499, 500, 501], [0, 1, 0]))

    nrows = file_len(filename)
    if nmax is None:
        nmax = nrows-1
    if nmax > (nrows-1):
        nmax = nrows-1

    # Define some useful np dtypes
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

    # Define the output compound dtype
    data = np.recarray((nmax,),
                          dtype = [('objectID', np.int64),
                                   ('raJ2000', np.float64),
                                   ('decJ2000', np.float64),
                                   ('magNorm', np.float32),
                                   ('sedFilePath', np.str_, 64),
                                   ('galacticAv', np.float32),
                                   ('mag', ugrizy),
                                   ('magCalc', ugrizyE),
                                   ('Rbar', ugrizy),
                                   ('V', ugrizy),
                                   ('S_m02', ugrizy),
                                   ('S_p06', E),
                                   ('S_p10', E)])

    with open(filename) as f:
        if not debug:
            outdev = sys.stdout
        else:
            outdev = open(os.devnull, 'w')
        with chroma.ProgressBar(nmax, file=outdev) as bar:
            for i, line in enumerate(f):
                if i == 0 : continue #ignore column labels row
                if i >= nmax : break
                bar.update()
                s = line.split(', ')
                data[i-1].objectID = int(s[0])
                data[i-1].raJ2000 = float(s[1])
                data[i-1].decJ2000 = float(s[2])
                data[i-1].magNorm = float(s[3])
                data[i-1].sedFilePath = s[10]
                data[i-1].galacticAv = float(s[11])
                spec = stellar_spectrum(data[i-1], filters['norm'])

                # fill in magnitudes and chromatic biases
                for k, f in enumerate('ugrizy'):
                    # also append magnitude from catalog as a sanity check
                    data[i-1]['mag']['LSST_'+f] = float(s[4+k])
                    bp = filters['LSST_'+f] # for brevity
                    try:
                        data[i-1]['magCalc']['LSST_'+f] = spec.getMagnitude(bp)
                        dcr = spec.getDCRMomentShifts(bp, np.pi/4)
                        data[i-1]['Rbar']['LSST_'+f] = dcr[0]
                        data[i-1]['V']['LSST_'+f] = dcr[1]
                        data[i-1]['S_m02']['LSST_'+f] = spec.getSeeingShift(bp, alpha=-0.2)
                    except:
                        data[i-1]['magCalc']['LSST_'+f] = np.nan
                        data[i-1]['Rbar']['LSST_'+f] = np.nan
                        data[i-1]['V']['LSST_'+f] = np.nan
                        data[i-1]['S_m02']['LSST_'+f] = np.nan
                for fw in [150, 250, 350, 450]:
                    fname = 'Euclid_{}'.format(fw)
                    bp = filters[fname]
                    try:
                        data[i-1]['magCalc'][fname] = spec.getMagnitude(bp)
                        data[i-1]['S_p06'][fname] = spec.getSeeingShift(bp, alpha=0.6)
                        data[i-1]['S_p10'][fname] = spec.getSeeingShift(bp, alpha=1.0)
                    except:
                        data[i-1]['magCalc'][fname] = np.nan
                        data[i-1]['S_p06'][fname] = np.nan
                        data[i-1]['S_p10'][fname] = np.nan

                if debug:
                    print
                    print 'syn mag:' + ' '.join(['{:6.3f}'.format(data[i-1]['magCalc']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'cat mag:' + ' '.join(['{:6.3f}'.format(data[i-1]['mag']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'Euclid: ' + ' '.join(['{:6.3f}'.format(data[i-1]['magCalc']['Euclid_{}'.format(fw)])
                                                 for fw in [150, 250, 350, 450]])
    return data

def runme():
    junk = process_star_file('output/star_catalog.dat', nmax=25, debug=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nmax', type=int, default=30000,
                        help="maximum number of stars to process")
    parser.add_argument('--outfile', default = 'output/star_data.pkl',
                        help="output filename (Default: output/star_data.pkl)")
    parser.add_argument('--infile', default = 'output/star_catalog.dat',
                        help="input filename (Default: output/star_catalog.dat)")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cPickle.dump(process_star_file(args.infile, nmax=args.nmax, debug=args.debug),
                 open(args.outfile, 'wb'))
