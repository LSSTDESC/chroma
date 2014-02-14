import os
import sys
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

def compute_color_gradient_separation_correction(b_spec, d_spec, filters):
    """ Compute the difference in zenith-direction second moment due to spatial separation
    of bulge and disk components."""

    b_wave = b_spec['wave']
    b_photons = b_wave * b_spec['flambda']
    d_wave = d_spec['wave']
    d_photons = d_wave * d_spec['flambda']
    dVcg = {} # change in second moment due to bulge-disk color gradient spatial separation.

    for filter_name, filter_ in filters.iteritems():
        b_detected_photons = b_photons * filter_['throughput']
        b_flux = scipy.integrate.simps(b_detected_photons, b_wave)
        b_R = chroma.atm_refrac(b_wave, zenith=numpy.pi/4.0)
        b_Rbar = scipy.integrate.simps(b_R * b_detected_photons, b_wave) / b_flux

        d_detected_photons = d_photons * filter_['throughput']
        d_flux = scipy.integrate.simps(d_detected_photons, d_wave)
        d_R = chroma.atm_refrac(d_wave, zenith=numpy.pi/4.0)
        d_Rbar = scipy.integrate.simps(d_R * d_detected_photons, d_wave) / d_flux

        c_flux = b_flux + d_flux
        b_frac = b_flux/c_flux
        d_frac = d_flux/c_flux

        c_Rbar = b_frac * b_Rbar  +  d_frac * d_Rbar

        dVcg[filter_name] = b_frac * (b_Rbar - c_Rbar)**2 + d_frac * (d_Rbar - c_Rbar)**2
    return dVcg

def readfile(filename, nmax=None, debug=False, randomize=True, emission=False, start=0):
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
                if emission:
                    spec = phot.make_composite_spec_with_emission_lines(data[j], filters,
                                                                        zps, wave_match)
                else:
                    spec = phot.make_composite_spec(data[j], filters, zps, wave_match)
                magCalcs = phot.compute_mags(spec, filters, zps)
                R, V, S_m02 = compute_ground_chromatic_corrections(spec, ground_filters)
                S_p06, S_p10 = compute_space_chromatic_corrections(spec, space_filters)
                if data[j].sedPathBulge != 'None' and data[j].sedPathDisk != 'None':
                    b_spec = phot.make_bulge_spec(data[j], filters, zps, wave_match)
                    d_spec = phot.make_disk_spec(data[j], filters, zps, wave_match)
                    dVcg = compute_color_gradient_separation_correction(b_spec, d_spec,
                                                                        ground_filters)
                else:
                    dVcg = {}
                    for fname in 'ugrizy':
                        dVcg['LSST_'+fname] = numpy.nan
                for k, fname in enumerate('ugrizy'):
                    data[j]['mag']['LSST_'+fname] = float(s[5+k])
                    data[j]['magCalc']['LSST_'+fname] = magCalcs['LSST_'+fname]
                    data[j]['R']['LSST_'+fname] = R['LSST_'+fname]
                    data[j]['V']['LSST_'+fname] = V['LSST_'+fname]
                    data[j]['S_m02']['LSST_'+fname] = S_m02['LSST_'+fname]
                    data[j]['dVcg']['LSST_'+fname] = dVcg['LSST_'+fname]
                for fw in [150, 250, 350, 450]:
                    fname = 'Euclid_{}'.format(fw)
                    data[j]['magCalc'][fname] = magCalcs[fname]
                    data[j]['S_p06'][fname] = S_p06[fname]
                    data[j]['S_p10'][fname] = S_p10[fname]
                if debug:
                    print
                    print 'mag:    ' + ' '.join(['{:6.3f}'.format(magCalcs['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'syn:    ' + ' '.join(['{:6.3f}'.format(data[j]['mag']['LSST_'+fname])
                                                 for fname in 'ugrizy'])
                    print 'Euclid: ' + ' '.join(['{:6.3f}'.format(magCalcs['Euclid_{}'.format(fw)])
                                                 for fw in [150, 250, 350, 450]])
                j += 1
    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--infile', default = 'output/galaxy_catalog.dat',
                        help="input filename. Default 'output/galaxy_catalog.dat'")
    parser.add_argument('--outfile', default = 'galaxy_data.pkl',
                        help="output filename. Default 'galaxy_data.pkl'")
    parser.add_argument('--nmax', type=int, default=30000,
                        help="maximum number of galaxies to process. Default 30000")
    parser.add_argument('--start', type=int, default=0,
                        help="starting index for catalog.  Default 0")
    parser.add_argument('--emission', dest='emission', action='store_true',
                        help="add emission lines to spectra")
    args = parser.parse_args()

    cPickle.dump(readfile(args.infile,
                          nmax=args.nmax,
                          emission=args.emission,
                          start=args.start),
                 open(args.outfile, 'wb'))
