import cPickle

import numpy as np
from numpy.lib.recfunctions import append_fields
import astropy.utils.console as console

import _mypath
import chroma

ugrizy =  [('LSST_u', np.float32),
           ('LSST_g', np.float32),
           ('LSST_r', np.float32),
           ('LSST_i', np.float32),
           ('LSST_z', np.float32),
           ('LSST_y', np.float32)]
E =       [('Euclid_150', np.float32),
           ('Euclid_250', np.float32),
           ('Euclid_350', np.float32),
           ('Euclid_450', np.float32)]
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

def compute_mags_moments(sed, filters):
    out = np.recarray((1,), dtype = [('mag', ugrizyE),
                                        ('Rbar', ugrizy),
                                        ('V', ugrizy),
                                        ('S_m02', ugrizy),
                                        ('S_p06', E),
                                        ('S_p10', E)])
    for filter_name, bandpass in filters.iteritems():
        try:
            out[0]['mag'][filter_name] = sed.magnitude(bandpass)
        except ValueError:
            out[0]['mag'][filter_name] = np.nan
        if filter_name.startswith('Euclid'):
            try:
                out[0]['S_p06'][filter_name] = sed.seeing_shift(bandpass, alpha=0.6)
                out[0]['S_p10'][filter_name] = sed.seeing_shift(bandpass, alpha=1.0)
            except ValueError:
                out[0]['S_p06'][filter_name] = np.nan
                out[0]['S_p10'][filter_name] = np.nan
        else:
            try:
                DCR_mom = sed.DCR_moment_shifts(bandpass, zenith=45.0)
                out[0]['Rbar'][filter_name] = DCR_mom[0]
                out[0]['V'][filter_name] = DCR_mom[1]
                out[0]['S_m02'][filter_name] = sed.seeing_shift(bandpass, alpha=-0.2)
            except ValueError:
                out[0]['Rbar'][filter_name] = np.nan
                out[0]['V'][filter_name] = np.nan
                out[0]['S_m02'][filter_name] = np.nan
    return out

def construct_analytic_table():
    spec_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'

    # define SED types
    star_types = ['uko5v',
                  'ukb5iii',
                  'uka5v',
                  'ukf5v',
                  'ukg5v',
                  'ukk5v',
                  'ukm5v']
    gal_types= ['CWW_E_ext',
                'KIN_Sa_ext',
                'KIN_Sb_ext',
                'CWW_Sbc_ext',
                'CWW_Scd_ext',
                'CWW_Im_ext',
                'KIN_SB1_ext',
                'KIN_SB6_ext']

    # load filters
    filter_names = ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y',
                   'Euclid_150', 'Euclid_250', 'Euclid_350', 'Euclid_450']
    filters = {}
    for f in filter_names:
        f_wave, f_throughput = np.genfromtxt(filter_dir + '{}.dat'.format(f)).T
        filters[f] = chroma.Bandpass(f_wave, f_throughput)
        filters[f].truncate(rel_throughput=0.01)

    # start with stars
    star_data = np.recarray((len(star_types),), dtype = [('star_type', 'a11'),
                                                         ('mag', ugrizyE),
                                                         ('Rbar', ugrizy),
                                                         ('V', ugrizy),
                                                         ('S_m02', ugrizy),
                                                         ('S_p06', E),
                                                         ('S_p10', E)])
    for i, star_type in enumerate(star_types):
        s_wave, s_flambda = np.genfromtxt(spec_dir + star_type + '.ascii').T
        star_SED = chroma.SED(s_wave, s_flambda)
        data = compute_mags_moments(star_SED, filters)
        star_data[i]['star_type'] = star_type
        for name in data.dtype.names:
            star_data[i][name] = data[name]
    cPickle.dump(star_data, open('stars.pkl', 'wb'))

    # now onto galaxies
    gal_data = np.recarray((len(gal_types)*100,), dtype = [('gal_type', 'a11'),
                                                           ('redshift', np.float32),
                                                           ('mag', ugrizyE),
                                                           ('Rbar', ugrizy),
                                                           ('V', ugrizy),
                                                           ('S_m02', ugrizy),
                                                           ('S_p06', E),
                                                           ('S_p10', E)])
    i=0
    with console.ProgressBar(100 * len(gal_types)) as bar:
        for gal_type in gal_types:
            g_wave, g_flambda = np.genfromtxt(spec_dir + gal_type + '.ascii').T
            gal_SED = chroma.SED(g_wave, g_flambda)
            for z in np.arange(0.0, 3.0, 0.03):
                bar.update()
                gal_SED.set_redshift(z)
                data = compute_mags_moments(gal_SED, filters)
                gal_data[i]['gal_type'] = gal_type
                gal_data[i]['redshift'] = z
                for name in data.dtype.names:
                    gal_data[i][name] = data[name]
                i += 1
    cPickle.dump(gal_data, open('galaxies.pkl', 'wb'))

if __name__ == '__main__':
    construct_analytic_table()
