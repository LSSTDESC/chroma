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

# For use in Exposure Time Calculator for magnitude error estimates
psf = galsim.Kolmogorov(fwhm = 0.67)

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

def component_spectrum(sedfile, magnorm, av, rv, redshift, norm_bandpass, emission=False):
    sed = chroma.SED(os.path.join(SED_dir, sedfile))
    sed = sed.withMagnitude(magnorm, norm_bandpass)
    sed.blue_limit = 91
    sed.red_limit = 6000
    if emission:
        sed = sed.addEmissionLines()
    sed = sed.redden(A_v=av, R_v=rv)
    sed = sed.atRedshift(redshift)
    return sed

def process_gal_file(filename, nmax=None, debug=False, seed=None, emission=False, start=0):
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
    ugrizy = [('LSST_u', np.float),
              ('LSST_g', np.float),
              ('LSST_r', np.float),
              ('LSST_i', np.float),
              ('LSST_z', np.float),
              ('LSST_y', np.float)]
    ugrizyE = [('LSST_u', np.float),
               ('LSST_g', np.float),
               ('LSST_r', np.float),
               ('LSST_i', np.float),
               ('LSST_z', np.float),
               ('LSST_y', np.float),
               ('Euclid_150', np.float),
               ('Euclid_250', np.float),
               ('Euclid_350', np.float),
               ('Euclid_450', np.float)]
    E = [('Euclid_150', np.float),
         ('Euclid_250', np.float),
         ('Euclid_350', np.float),
         ('Euclid_450', np.float)]

    data = np.recarray((nmax,),
                       dtype = [('galTileID', np.uint64),
                                ('objectID', np.uint64),
                                ('raJ2000', np.float),
                                ('decJ2000', np.float),
                                ('redshift', np.float),
                                ('BulgeHLR', np.float),
                                ('DiskHLR', np.float),
                                ('BulgeE1', np.float),
                                ('BulgeE2', np.float),
                                ('DiskE1', np.float),
                                ('DiskE2', np.float),
                                ('sedPathBulge', np.str_, 64),
                                ('sedPathDisk', np.str_, 64),
                                ('sedPathAGN', np.str_, 64),
                                ('magNormBulge', np.float),
                                ('magNormDisk', np.float),
                                ('magNormAGN', np.float),
                                ('internalAVBulge', np.float),
                                ('internalRVBulge', np.float),
                                ('internalAVDisk', np.float),
                                ('internalRVDisk', np.float),
                                ('mag', ugrizy),
                                ('magCalc', ugrizyE),
                                ('magErr', ugrizy),
                                ('BulgeFrac', ugrizyE),
                                ('Rbar', ugrizy),
                                ('V', ugrizy),
                                ('dVcg', ugrizy),
                                ('S_m02', ugrizy),
                                ('S_p06', E),
                                ('S_p10', E)])

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
                data[j].galTileID = int(s[0])
                data[j].objectID = int(s[1])
                data[j].raJ2000 = float(s[2])
                data[j].decJ2000 = float(s[3])
                data[j].redshift = float(s[4])

                # size and ellipticity
                A_Bulge = float(s[5])
                B_Bulge = float(s[6])
                A_Disk = float(s[7])
                B_Disk = float(s[8])
                PA_Bulge = float(s[9])
                PA_Disk = float(s[10])
                if A_Bulge != 0.0:
                    data[j].BulgeHLR = np.sqrt(A_Bulge * B_Bulge)
                    ellip = (A_Bulge**2 - B_Bulge**2) / (A_Bulge**2 + B_Bulge**2)
                    if ellip > 0.99:
                        ellip = 0.99
                    data[j].BulgeE1 = ellip * np.cos(2 * PA_Bulge * np.pi/180.0)
                    data[j].BulgeE2 = ellip * np.sin(2 * PA_Bulge * np.pi/180.0)
                if A_Disk != 0.0:
                    data[j].DiskHLR = np.sqrt(A_Disk * B_Disk)
                    ellip = (A_Disk**2 - B_Disk**2) / (A_Disk**2 + B_Disk**2)
                    if ellip > 0.99:
                        ellip = 0.99
                    data[j].DiskE1 = ellip * np.cos(2 * PA_Disk * np.pi/180.0)
                    data[j].DiskE2 = ellip * np.sin(2 * PA_Disk * np.pi/180.0)

                # spectrum
                data[j].sedPathBulge = s[17]
                data[j].sedPathDisk = s[18]
                data[j].sedPathAGN = s[19]
                data[j].magNormBulge = float(s[20])
                data[j].magNormDisk = float(s[21])
                data[j].magNormAGN = float(s[22])
                data[j].internalAVBulge = float(s[23])
                data[j].internalRVBulge = float(s[24])
                data[j].internalAVDisk = float(s[25])
                data[j].internalRVDisk = float(s[26])

                # create indiv and composite SEDs, and galsim SBProfiles
                if 'bulge_spec' in locals():
                    del bulge_spec
                if 'disk_spec' in locals():
                    del disk_spec
                if A_Bulge != 0.0:
                    bulge_spec = component_spectrum(
                        data[j].sedPathBulge, data[j].magNormBulge,
                        data[j].internalAVBulge, data[j].internalRVBulge,
                        data[j].redshift, filters['norm'], emission=emission)
                    spec = bulge_spec
                    bulge = galsim.DeVaucouleurs(half_light_radius = data[j].BulgeHLR)
                    bulge = bulge.shear(e1=data[j].BulgeE1, e2=data[j].BulgeE2)
                if A_Disk != 0.0:
                    disk_spec = component_spectrum(
                        data[j].sedPathDisk, data[j].magNormDisk,
                        data[j].internalAVDisk, data[j].internalRVDisk,
                        data[j].redshift, filters['norm'], emission=emission)
                    spec = disk_spec
                    disk = galsim.Exponential(half_light_radius = data[j].DiskHLR)
                    disk = disk.shear(e1=data[j].DiskE1, e2=data[j].DiskE2)
                if 'bulge_spec' in locals() and 'disk_spec' in locals():
                    spec = bulge_spec + disk_spec

                # loop through filters and fill in database columns
                for k, f in enumerate('ugrizy'):
                    # grab catalog magnitude
                    data[j]['mag']['LSST_'+f] = float(s[11+k])
                    bp = filters['LSST_'+f] # for brevity
                    try:
                        data[j]['magCalc']['LSST_'+f] = spec.calculateMagnitude(bp)
                        dcr = spec.calculateDCRMomentShifts(bp, zenith_angle=np.pi/4)
                        data[j]['Rbar']['LSST_'+f] = dcr[0][1,0]
                        data[j]['V']['LSST_'+f] = dcr[1][1,1]
                        data[j]['S_m02']['LSST_'+f] = spec.calculateSeeingMomentRatio(bp)
                        if 'bulge_spec' in locals() and 'disk_spec' not in locals():
                            data[j]['BulgeFrac']['LSST_'+f] = 1.0
                            gal = bulge
                        elif 'disk_spec' in locals() and 'bulge_spec' not in locals():
                            data[j]['BulgeFrac']['LSST_'+f] = 0.0
                            gal = disk
                        else:
                            bulge_flux = bulge_spec.calculateFlux(bp)
                            disk_flux = disk_spec.calculateFlux(bp)
                            data[j]['BulgeFrac']['LSST_'+f] = bulge_flux / disk_flux
                            gal = bulge*bulge_flux + disk*disk_flux
                        profile = galsim.Convolve(psf, gal)
                        etc = chroma.lsstetc.ETC(f)
                        data[j]['magErr']['LSST_'+f] = etc.err(profile,
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
                if debug:
                    print
                    print 'syn mag:' + ' '.join(['{:6.3f}'.format(data[j]['magCalc']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'syn err:' + ' '.join(['{:6.3f}'.format(data[j]['magErr']['LSST_'+fname])
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
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed to randomize order of galaxies in catalog. [Default: None]")
    parser.add_argument('--start', type=int, default=0,
                        help="starting index for catalog.  Default 0")
    parser.add_argument('--emission', action='store_true',
                        help="add emission lines to spectra")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cPickle.dump(process_gal_file(args.infile,
                                  nmax=args.nmax,
                                  seed=args.seed,
                                  emission=args.emission,
                                  start=args.start,
                                  debug=args.debug),
                 open(args.outfile, 'wb'))
