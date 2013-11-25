import sys
import os
import cPickle
from argparse import ArgumentParser

import numpy
import scipy
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

def compute_ground_chromatic_corrections(spec, filters):
    """ Compute the chromatic correction coefficients for given spectrum and filter set.
    """
    wave = spec['wave']
    photons = wave * spec['flambda']
    R = {} # centroid shift due to differential chromatic refraction (DCR)
    V = {} # zenith direction second moment shift due to DCR
    S_m02 = {} # relative scale of PSF second moments from chromatic seeing

    for filter_name, filter_ in filters.iteritems():
        detected_photons = photons * filter_['throughput']
        R[filter_name], V[filter_name] = \
          chroma.disp_moments(wave, detected_photons, zenith=numpy.pi/4.0)
        S_m02[filter_name] = chroma.relative_second_moment_radius(wave, detected_photons, -0.2)
    return R, V, S_m02

def compute_space_chromatic_corrections(spec, filters):
    """ Compute the chromatic correction coefficients for given spectrum and filter set.
    """
    wave = spec['wave']
    photons = wave * spec['flambda']
    S_p06 = {} # relative scale of PSF second moments from total Euclid PSF
    S_p10 = {} # relative scale of PSF second moments from Diffraction limit

    for filter_name, filter_ in filters.iteritems():
        detected_photons = photons * filter_['throughput']
        S_p06[filter_name] = chroma.relative_second_moment_radius(wave, detected_photons, 0.6)
        S_p10[filter_name] = chroma.relative_second_moment_radius(wave, detected_photons, 1.0)
    return S_p06, S_p10

def readfile(filename, nmax=None, debug=False):
    SED_dir = os.environ['CAT_SHARE_DATA']+'data/'
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
                          dtype = [('objectID', numpy.int64),
                                   ('raJ2000', numpy.float64),
                                   ('decJ2000', numpy.float64),
                                   ('magNorm', numpy.float32),
                                   ('sedFilePath', numpy.str_, 64),
                                   ('galacticAv', numpy.float32),
                                   ('mag', ugrizy),
                                   ('magCalc', ugrizyE),
                                   ('R', ugrizy),
                                   ('V', ugrizy),
                                   ('S_m02', ugrizy),
                                   ('S_p06', E),
                                   ('S_p10', E)])
    with open(filename) as f:
        if not debug:
            outdev = sys.stdout
        else:
            outdev = open(os.devnull, 'w')
        with console.ProgressBar(nmax, file=outdev) as bar:
            for i, line in enumerate(f):
                if i == 0 : continue #ignore column labels row
                if i > nmax : break
                bar.update()
                s = line.split(', ')
                data[i-1].objectID = int(s[0])
                data[i-1].raJ2000 = float(s[1])
                data[i-1].decJ2000 = float(s[2])
                data[i-1].magNorm = float(s[3])
                data[i-1].sedFilePath = s[10]
                data[i-1].galacticAv = float(s[11])
                spec = phot.read_spec(SED_dir+data[i-1].sedFilePath)
                spec = phot.scale_spec(spec, data[i-1].magNorm, filters['norm'], zps['norm'])
                spec = phot.apply_extinction(spec, data[i-1].galacticAv)
                spec = phot.match_wavelengths(spec, wave_match)
                magCalcs = phot.compute_mags(spec, filters, zps)
                R, V, S_m02 = compute_ground_chromatic_corrections(spec, ground_filters)
                S_p06, S_p10 = compute_space_chromatic_corrections(spec, space_filters)
                for j, fname in enumerate('ugrizy'):
                    data[i-1]['mag']['LSST_'+fname] = float(s[4+j])
                    data[i-1]['magCalc']['LSST_'+fname] = magCalcs['LSST_'+fname]
                    data[i-1]['R']['LSST_'+fname] = R['LSST_'+fname]
                    data[i-1]['V']['LSST_'+fname] = V['LSST_'+fname]
                    data[i-1]['S_m02']['LSST_'+fname] = S_m02['LSST_'+fname]
                for fw in [150, 250, 350, 450]:
                    fname = 'Euclid_{}'.format(fw)
                    data[i-1]['magCalc'][fname] = magCalcs[fname]
                    data[i-1]['S_p06'][fname] = S_p06[fname]
                    data[i-1]['S_p10'][fname] = S_p10[fname]
                if debug:
                    print
                    print 'mag:    ' + ' '.join(['{:6.3f}'.format(magCalcs['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'syn:    ' + ' '.join(['{:6.3f}'.format(data[i-1]['mag']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'Euclid: ' + ' '.join(['{:6.3f}'.format(magCalcs['Euclid_{}'.format(fw)])
                                                 for fw in [150, 250, 350, 450]])
    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nmax', type=int, default=1000,
                        help="maximum number of stars to process")
    parser.add_argument('--outfile', default = 'star_data.pkl',
                        help="output filename")
    parser.add_argument('--infile', default = 'output/star_catalog.dat',
                        help="input filename")
    args = parser.parse_args()

    cPickle.dump(readfile(args.infile, nmax=args.nmax), open(args.outfile, 'wb'))
