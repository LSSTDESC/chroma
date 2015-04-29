"""Process star catalog produced by make_catalogs.py to add columns for DCR biases, chromatic
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

import sys
import os
import cPickle
from argparse import ArgumentParser

import numpy as np
from scipy.interpolate import interp1d
import galsim

import _mypath
import chroma
import chroma.lsstetc

# Exposure Time Calculator for magnitude error estimates
psf = galsim.Kolmogorov(fwhm = 0.67)
etc = {f:chroma.lsstetc.ETC(f) for f in 'ugrizy'}

datadir = '../../../data/'

if 'CAT_SHARE_DATA' in os.environ:
    SED_dir = os.environ['CAT_SHARE_DATA'] + 'data'
elif 'SIMS_SED_LIBRARY_DIR' in os.environ:
    SED_dir = os.environ['SIMS_SED_LIBRARY_DIR']
else:
    raise ValueError("Cannot find CatSim SED files.")

def file_len(fname):
    """Count '\n's in file.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def stellar_spectrum(star, norm_bandpass):
    sed = chroma.SED(os.path.join(SED_dir, star['sedFilePath']))
    sed = sed.withMagnitude(star['magNorm'], norm_bandpass)
    sed.blue_limit = 91
    sed.red_limit = 6000
    sed = sed.redden(A_v=star['galacticAv'])
    return sed

def process_star_file(filename, nmax=None, debug=False, seed=None, start=0):
    filters = {}
    for f in 'ugrizy':
        ffile = datadir+'filters/LSST_{}.dat'.format(f)
        filters['LSST_{}'.format(f)] = (galsim.Bandpass(ffile)
                                        .thin(1.e-5) # thin for speed
                                        .withZeropoint('AB',
                                                       effective_diameter=6.4,
                                                       exptime=30.0))
    for width in [150,250,350,450]:
        ffile = datadir+'filters/Euclid_{}.dat'.format(width)
        filters['Euclid_{}'.format(width)] = (galsim.Bandpass(ffile)
                                              .thin(1.e-5)
                                              .withZeropoint('AB',
                                                             effective_diameter=6.4, # huh?
                                                             exptime=30.0))
    for f in 'ugriz':
        ffile = datadir+'filters/sdss/SDSS_{}.dat'.format(f)
        filters['SDSS_{}'.format(f)] = (galsim.Bandpass(ffile, wave_type='a')
                                        .withZeropoint('AB',
                                                       effective_diameter=6.4, # huh?
                                                       exptime=30.0))


    # LSST SED catalog entries are normalized by their AB magnitude at 500 nm.  So define a narrow
    # filter at 500nm to use for normalization.
    filters['norm'] = (galsim.Bandpass(galsim.LookupTable([499, 500, 501], [0, 1, 0]))
                       .withZeropoint('AB', effective_diameter=6.4, exptime=30.0))

    nrows = file_len(filename)
    if nmax is None:
        nmax = nrows-1
    if nmax > (nrows-1):
        nmax = nrows-1

    # Define some useful np dtypes
    Lbands = [('LSST_u', np.float),
              ('LSST_g', np.float),
              ('LSST_r', np.float),
              ('LSST_i', np.float),
              ('LSST_z', np.float),
              ('LSST_y', np.float)]
    Ebands = [('Euclid_150', np.float),
              ('Euclid_250', np.float),
              ('Euclid_350', np.float),
              ('Euclid_450', np.float)]
    LSbands = [('LSST_u', np.float),
               ('LSST_g', np.float),
               ('LSST_r', np.float),
               ('LSST_i', np.float),
               ('LSST_z', np.float),
               ('LSST_y', np.float),
               ('SDSS_u', np.float),
               ('SDSS_g', np.float),
               ('SDSS_r', np.float),
               ('SDSS_i', np.float),
               ('SDSS_z', np.float)]
    LEbands = [('LSST_u', np.float),
               ('LSST_g', np.float),
               ('LSST_r', np.float),
               ('LSST_i', np.float),
               ('LSST_z', np.float),
               ('LSST_y', np.float),
               ('Euclid_150', np.float),
               ('Euclid_250', np.float),
               ('Euclid_350', np.float),
               ('Euclid_450', np.float)]
    LSEbands = [('LSST_u', np.float),
                ('LSST_g', np.float),
                ('LSST_r', np.float),
                ('LSST_i', np.float),
                ('LSST_z', np.float),
                ('LSST_y', np.float),
                ('SDSS_u', np.float),
                ('SDSS_g', np.float),
                ('SDSS_r', np.float),
                ('SDSS_i', np.float),
                ('SDSS_z', np.float),
                ('Euclid_150', np.float),
                ('Euclid_250', np.float),
                ('Euclid_350', np.float),
                ('Euclid_450', np.float)]

    # Define the output compound dtype
    data = np.recarray((nmax,),
                          dtype = [('objectID', np.int64),
                                   ('raJ2000', np.float),
                                   ('decJ2000', np.float),
                                   ('magNorm', np.float),
                                   ('sedFilePath', np.str_, 64),
                                   ('galacticAv', np.float),
                                   ('mag', Lbands), # only LSST since read straight from CatSim
                                   ('magCalc', LSEbands),
                                   ('magErr', LSEbands),
                                   ('Rbar', LSbands), # doesn't make sense for space mission
                                   ('V', LSbands),
                                   ('S_m02', LSbands),
                                   ('S_p06', Ebands),
                                   ('S_p10', Ebands)])
    data[:] = np.nan

    order = [d+1 for d in xrange(nrows)]
    if seed is not None:
        import random
        random.seed(seed)
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
                if i == 0 : continue # ignore column labels row
                if j >= nmax : break
                if order[j] != i : continue
                bar.update()
                s = line.split(', ')

                # position
                data[j].objectID = int(s[0])
                data[j].raJ2000 = float(s[1])
                data[j].decJ2000 = float(s[2])

                # flux
                data[j].magNorm = float(s[3])
                data[j].sedFilePath = s[10]
                data[j].galacticAv = float(s[11])
                spec = stellar_spectrum(data[j], filters['norm'])

                # loop through filters and fill in database columns
                for k, f in enumerate('ugrizy'):
                    # also append magnitude from catalog as a sanity check
                    data[j]['mag']['LSST_'+f] = float(s[4+k])
                    bp = filters['LSST_'+f] # for brevity
                    try:
                        data[j]['magCalc']['LSST_'+f] = spec.calculateMagnitude(bp)
                        dcr = spec.calculateDCRMomentShifts(bp, zenith_angle=np.pi/4)
                        data[j]['Rbar']['LSST_'+f] = dcr[0][1,0]
                        data[j]['V']['LSST_'+f] = dcr[1][1,1]
                        data[j]['S_m02']['LSST_'+f] = spec.calculateSeeingMomentRatio(bp)
                        data[j]['magErr']['LSST_'+f] = etc[f].err(psf,
                                                                    data[j]['magCalc']['LSST_'+f])
                    except:
                        pass
                # separate loop for Euclid filters
                for fw in [150, 250, 350, 450]:
                    fname = 'Euclid_{}'.format(fw)
                    bp = filters[fname]
                    try:
                        data[j]['magCalc'][fname] = spec.calculateMagnitude(bp)
                        data[j]['S_p06'][fname] = spec.calculateSeeingMomentRatio(bp, alpha=0.6)
                        data[j]['S_p10'][fname] = spec.calculateSeeingMomentRatio(bp, alpha=1.0)
                    except:
                        pass
                # separate loop for SDSS filters
                for f in 'ugriz':
                    fname = 'SDSS_{}'.format(f)
                    bp = filters[fname]
                    try:
                        data[j]['magCalc'][fname] = spec.calculateMagnitude(bp)
                        dcr = spec.calculateDCRMomentShifts(bp, zenith_angle=np.pi/4)
                        data[j]['Rbar'][fname] = dcr[0][1,0]
                        data[j]['V'][fname] = dcr[1][1,1]
                        data[j]['S_m02'][fname] = spec.calculateSeeingMomentRatio(bp)
                    except:
                        pass
                if debug:
                    print
                    print 'syn mag:' + ' '.join(['{:6.3f}'.format(
                        data[j]['magCalc']['LSST_'+fname])
                        for fname in 'ugrizy'])
                    print 'syn err:' + ' '.join(['{:6.3f}'.format(
                        data[j]['magErr']['LSST_'+fname])
                        for fname in 'ugrizy'])
                    print 'cat mag:' + ' '.join(['{:6.3f}'.format(data[j]['mag']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'Euclid: ' + ' '.join(['{:6.3f}'.format(
                        data[j]['magCalc']['Euclid_{}'.format(fw)])
                        for fw in [150, 250, 350, 450]])
                j += 1
    return data

def runme():
    junk = process_star_file('output/star_catalog.dat', nmax=25, debug=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nmax', type=int, default=30000,
                        help="maximum number of stars to process (default: 30000)")
    parser.add_argument('--seed', type=int, default=None,
                        help="randomize order of stars in catalog")
    parser.add_argument('--outfile', default = 'output/star_data.pkl',
                        help="output filename (Default: output/star_data.pkl)")
    parser.add_argument('--infile', default = 'output/star_catalog.dat',
                        help="input filename (Default: output/star_catalog.dat)")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cPickle.dump(process_star_file(args.infile, nmax=args.nmax, 
                                   debug=args.debug, seed=args.seed),
                 open(args.outfile, 'wb'))
