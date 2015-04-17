""" Create tables of AB magnitudes and chromatic biases from a variety of spectra.

For stars, we use the spectra from Pickles (1998), (the 'uk*.ascii' files in the data/SEDs/
subdirectory).

For galaxies, we use spectra from Coleman et al. (1980) (CWW*.ascii files), and Kinney et al. (1996)
(KIN*.ascii files)

The chromatic biases computed are:
  - the shift in PSF centroid due to differential chromatic refraction
  - the shift in PSF zenith-direction second moment due to differential chromatic refraction
  For LSST filters:
  - the shift in PSF second moment square radius (Ixx + Iyy) due to chromatic seeing
  For Euclid filters:
  - the shift in PSF second moment square radius due to a pure diffraction limited PSF
  - the shift in PSF second moment square radius with a FWHM \propto \lambda^{0.6} dependence,
    which is appropriate for Euclid (Cypriano et al. (2010), Voigt et al. (2012)).

For stars, only redshift 0 is considered.
For galaxies, magnitudes and biases are computed over a range of redshifts.
"""

import cPickle
import os

import numpy as np

import _mypath
import chroma

# Define some useful numpy dtypes.
# LSST filters
ugrizy =  [('LSST_u', np.float),
           ('LSST_g', np.float),
           ('LSST_r', np.float),
           ('LSST_i', np.float),
           ('LSST_z', np.float),
           ('LSST_y', np.float)]
# The Euclid telescope will fly with one optical filter, nominally of width 350nm.  We include
# additional potential optical filters here to see the effect of filter width on chromatic biases,
# as was done in Voigt et al. (2012)
E =       [('Euclid_150', np.float),
           ('Euclid_250', np.float),
           ('Euclid_350', np.float),
           ('Euclid_450', np.float)]
# Both LSST and Euclid
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

def compute_mags_moments(sed, filters):
    """ Given an SED and some filters, compute magnitudes and chromatic biases.
    """
    out = np.recarray((1,), dtype = [('mag', ugrizyE),   # AB magnitude
                                     ('Rbar', ugrizy),   # DCR centroid shift
                                     ('V', ugrizy),      # DCR second moment shift
                                     ('S_m02', ugrizy),  # PSF size shift with exponent -0.2
                                     ('S_p06', E),       # same, but exponent = +0.6
                                     ('S_p10', E),       # same, but exponent = +1.0
                                     ('linear', ugrizyE)]) #LinearSecondMomentShift
    for filter_name, bandpass in filters.iteritems():
        w_eff = bandpass.effective_wavelength
        beta_slope = 1.e-5 #arcsec^2/nm
        beta_slope *= (1./3600 * np.pi/180)**2 # -> rad^2/nm
        # some magnitude calculations will fail because the SED doesn't cover the wavelength range
        # of the bandpass filter.  Catch these here.
        try:
            out[0]['mag'][filter_name] = sed.calculateMagnitude(bandpass)
        except ValueError:
            out[0]['mag'][filter_name] = np.nan
        if filter_name.startswith('Euclid'):
            # For Euclid filters, investigate both a FWHM \propto \lambda^1.0 chromatic PSF (pure
            # diffraction limit), and a FWHM \propto \lambda^0.6 chromatic PSF (which is more
            # likely given the additional contribution to the Euclid PSF from CCDs and jitter (see
            # Cypriano et al. 2010)
            try:
                out[0]['S_p06'][filter_name] = sed.calculateSeeingMomentRatio(bandpass, alpha=0.6)
                out[0]['S_p10'][filter_name] = sed.calculateSeeingMomentRatio(bandpass, alpha=1.0)
                out[0]['linear'][filter_name] = sed.calculateLinearMomentShift(bandpass,
                                                                               beta_slope,
                                                                               700.0)
            except ValueError:
                out[0]['S_p06'][filter_name] = np.nan
                out[0]['S_p10'][filter_name] = np.nan
                out[0]['linear'][filter_name] = np.nan
        else:
            # For LSST filters, compute shifts in the zenith-direction first and second moments of
            # the PSF due to differential chromatic refraction.  Store results for a zenith angle of
            # 45 degrees, which can easily be scaled later by tan(zenith_angle) and
            # tan^2(zenith_angle) for the shift in first and second moments respectively.
            #
            # Also compute shift in PSF second moment square radius due to \lambda^{-0.2} chromatic
            # seeing.
            try:
                DCR_mom = sed.calculateDCRMomentShifts(bandpass, zenith_angle=np.pi/4)
                out[0]['Rbar'][filter_name] = DCR_mom[0][1,0]
                out[0]['V'][filter_name] = DCR_mom[1][1,1]
                out[0]['S_m02'][filter_name] = sed.calculateSeeingMomentRatio(bandpass, alpha=-0.2)
                out[0]['linear'][filter_name] = sed.calculateLinearMomentShift(bandpass,
                                                                               beta_slope,
                                                                               w_eff)
            except ValueError:
                out[0]['Rbar'][filter_name] = np.nan
                out[0]['V'][filter_name] = np.nan
                out[0]['S_m02'][filter_name] = np.nan
                out[0]['linear'][filter_name] = np.nan
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
        filters[f] = (chroma.Bandpass(filter_dir + '{}.dat'.format(f))
                      .withZeropoint('AB', effective_diameter=6.4, exptime=30.0))

    # start with stars
    star_data = np.recarray((len(star_types),), dtype = [('star_type', 'a11'),
                                                         ('mag', ugrizyE),
                                                         ('Rbar', ugrizy),
                                                         ('V', ugrizy),
                                                         ('S_m02', ugrizy),
                                                         ('S_p06', E),
                                                         ('S_p10', E),
                                                         ('linear', ugrizyE)])
    for i, star_type in enumerate(star_types):
        star_SED = chroma.SED(spec_dir + star_type + '.ascii')
        data = compute_mags_moments(star_SED, filters)
        star_data[i]['star_type'] = star_type
        for name in data.dtype.names:
            star_data[i][name] = data[name]
    if not os.path.isdir('output'):
        os.mkdir('output')
    cPickle.dump(star_data, open('output/stars.pkl', 'wb'))

    # now onto galaxies
    gal_data = np.recarray((len(gal_types)*100,), dtype = [('gal_type', 'a11'),
                                                           ('redshift', np.float),
                                                           ('mag', ugrizyE),
                                                           ('Rbar', ugrizy),
                                                           ('V', ugrizy),
                                                           ('S_m02', ugrizy),
                                                           ('S_p06', E),
                                                           ('S_p10', E),
                                                           ('linear', ugrizyE)])
    i=0
    with chroma.ProgressBar(100 * len(gal_types)) as bar:
        for gal_type in gal_types:
            gal_SED0 = chroma.SED(spec_dir + gal_type + '.ascii')
            for z in np.arange(0.0, 3.0, 0.03):
                bar.update()
                gal_SED = gal_SED0.atRedshift(z)
                data = compute_mags_moments(gal_SED, filters)
                gal_data[i]['gal_type'] = gal_type
                gal_data[i]['redshift'] = z
                for name in data.dtype.names:
                    gal_data[i][name] = data[name]
                i += 1
    cPickle.dump(gal_data, open('output/galaxies.pkl', 'wb'))

if __name__ == '__main__':
    construct_analytic_table()
