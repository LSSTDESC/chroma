import os
import numpy

import _mypath
import chroma

def load_LSST_filters():
    """ Read the LSST filter definitions from disk.  Also add a normalization filter that is
    only non-zero at 500nm, which is used to define the CatSim/PhoSim normalization.
    """
    filter_dir = '../../data/filters/'
    filter_names = 'ugrizy'
    # establish wavelength array
    filters = {}
    for filter_name in filter_names:
        ffile = filter_dir+'LSST_{}.dat'.format(filter_name)
        wave, throughput = numpy.genfromtxt(ffile).T
        filters['LSST_'+filter_name] = {'wave':wave,
                                        'throughput':throughput}
    norm_throughput = numpy.zeros_like(wave)
    norm_throughput[wave==500] = 1.0
    filters['norm'] = {'wave':wave,
                       'throughput':norm_throughput}
    return filters

def load_Euclid_filters():
    """ Load Euclid filters as defined by Voigt+12, centered at 775nm with widths of 150, 250, 350,
    and 450 nm.  These are stored in the data directory."""

    filter_dir = '../../data/filters/'
    filter_names = ['Euclid_{}'.format(width) for width in [150,250,350,450]]

    filters = {}
    for filter_name in filter_names:
        ffile = filter_dir+'{}.dat'.format(filter_name)
        wave, throughput = numpy.genfromtxt(ffile).T
        filters[filter_name] = {'wave':wave,
                                'throughput':throughput}
    norm_throughput = numpy.zeros_like(wave)
    norm_throughput[wave==500] = 1.0
    filters['norm'] = {'wave':wave,
                       'throughput':norm_throughput}
    return filters

def match_filter_wavelengths(filters, wave_match):
    for fname, f in filters.iteritems():
        new_throughput = numpy.interp(wave_match, f['wave'], f['throughput'])
        filters[fname] = {'wave':wave_match,
                          'throughput':new_throughput}
    return filters

def AB_zeropoints(filters):
    """Compute AB zeropoints for given filters.
    """
    # define AB source in flambda
    ABsource = 3631e-23 # 3631 Jy -> erg/s/Hz/cm^2
    c = 29979245800.0 # cm/s
    nm_to_cm = 1.0e-7

    zps = {}
    for filter_name, filter_ in filters.iteritems():
        fwave = filter_['wave']
        throughput = filter_['throughput']
        ABflambda = ABsource * c / fwave**2 / nm_to_cm # erg/s/Hz/cm^2*cm/s/nm^2 -> erg/s/cm^2/nm
        AB_photons = ABflambda * fwave * throughput
        dlambda = fwave[1] - fwave[0] # assuming linear wavelength bins!
        AB_sumphotons = (AB_photons * dlambda).sum()
        zps[filter_name] = -2.5 * numpy.log10(AB_sumphotons)
    return zps

def read_spec(specfile):
    wave, flambda = numpy.genfromtxt(specfile).T
    return {'wave':wave, 'flambda':flambda}

def match_wavelengths(spec, wave_match):
    """ Interpolate spectrum onto given wavelength array.
    """
    flux_i = numpy.interp(wave_match, spec['wave'], spec['flambda'])
    return {'wave':wave_match, 'flambda':flux_i}

def apply_redshift(spec, redshift):
    flux_i = numpy.interp(spec['wave'], spec['wave'] * (1.0 + redshift), spec['flambda'])
    return {'wave':spec['wave'], 'flambda':flux_i / (1.0 + redshift)}

def scale_spec(spec, target_mag, normfilter, normzp):
    """ Multiply spectrum flux such that the normalization magnitude matches the given target
    magnitude.
    """
    fwave = normfilter['wave']
    throughput = normfilter['throughput']
    flambda_i = numpy.interp(fwave, spec['wave'], spec['flambda'])
    photons = flambda_i * fwave * throughput
    dlambda = fwave[1] - fwave[0]
    sumphotons = (photons * dlambda).sum()
    current_mag = -2.5 * numpy.log10(sumphotons) - normzp
    multiplier = 10**(-0.4 * (target_mag - current_mag))
    spec['flambda'] *= multiplier
    return spec

def apply_extinction(spec, A_v, R_v=3.1):
    wave = spec['wave']
    valid = (wave > 92) & (wave < 1200)
    wave = wave[valid]
    ext = chroma.extinction.reddening(wave*10, a_v=A_v, r_v=R_v, model='f99')
    flambda = spec['flambda']
    flambda[valid] /= ext
    spec = {'wave':spec['wave'], 'flambda':flambda}
    return spec

def compute_mags(spec, filters, zps):
    """ Compute magnitudes from spectrum.  Assume that spectrum wavelengths are already matched to
    filter wavelengths.
    """
    mags = {}
    for filter_name, filter_ in filters.iteritems():
        fwave = filter_['wave']
        throughput = filter_['throughput']
        photons = spec['flambda'] * fwave * throughput
        dlambda = fwave[1] - fwave[0] # assuming linear wavelength bins!
        sumphotons = (photons * dlambda).sum()
        mags[filter_name] = -2.5 * numpy.log10(sumphotons) - zps[filter_name]
    return mags

def make_composite_spec(gal, filters, zps, wave_match):
    SED_dir = os.environ['CAT_SHARE_DATA']+'data/'

    if gal['sedPathBulge'] != 'None':
        bulge_spec = read_spec(SED_dir+gal['sedPathBulge'])
        bulge_spec = scale_spec(bulge_spec, gal['magNormBulge'],
                                filters['norm'], zps['norm'])
        bulge_spec = apply_extinction(bulge_spec,
                                      A_v=gal['internalAVBulge'],
                                      R_v=gal['internalRVBulge'])
        bulge_spec = apply_redshift(bulge_spec, gal['redshift'])
        bulge_spec = match_wavelengths(bulge_spec, wave_match)
    else:
        bulge_spec = {'wave':wave_match, 'flambda':numpy.zeros_like(wave_match)}
    if gal['sedPathDisk'] != 'None':
        disk_spec = read_spec(SED_dir+gal['sedPathDisk'])
        disk_spec = scale_spec(disk_spec, gal['magNormDisk'],
                                filters['norm'], zps['norm'])
        disk_spec = apply_extinction(disk_spec,
                                      A_v=gal['internalAVDisk'],
                                      R_v=gal['internalRVDisk'])
        disk_spec = apply_redshift(disk_spec, gal['redshift'])
        disk_spec = match_wavelengths(disk_spec, wave_match)
    else:
        disk_spec = {'wave':wave_match, 'flambda':numpy.zeros_like(wave_match)}
    if gal['sedPathAGN'] != 'None':
        AGN_spec = read_spec(SED_dir+gal['sedPathAGN'])
        AGN_spec = scale_spec(AGN_spec, gal['magNormAGN'],
                              filters['norm'], zps['norm'])
        AGN_spec = apply_redshift(AGN_spec, gal['redshift'])
        AGN_spec = match_wavelengths(AGN_spec, wave_match)
    else:
        AGN_spec = {'wave':wave_match, 'flambda':numpy.zeros_like(wave_match)}
    return {'wave':wave_match, 'flambda':(bulge_spec['flambda'] +
                                          disk_spec['flambda'] +
                                          AGN_spec['flambda'])}
