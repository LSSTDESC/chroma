'''
Want to measure first and second moments of simulated stars relative to the simulated G5v stars.
'''

import pickle

import numpy
import pyfits
import astropy.utils.console

def encode_obshistid(SED_type, filter_name, zenith, seed, redshift):
    SED_types = {'G5v':'1', 'star':'2', 'gal':'3'}
    SED_digit = SED_types[SED_type]
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    redshift_digits = '{:02d}'.format(int(round((redshift / 0.03))))
    return SED_digit + filter_digit + zenith_digit + seed_digit + redshift_digits

def encode_obstypeid(filter_name, zenith, seed):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return filter_digit + zenith_digit + seed_digit

def relative_moments(filter_name, zenith, seed):
    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    output_columns = ['ALPHAWIN_SKY', 'DELTAWIN_SKY',
                      'X2_WORLD', 'Y2_WORLD',
                      'VX10', 'VY10',
                      'VX20', 'VY20',
                      'VX40', 'VY40',
                      'VX80', 'VY80',
                      'VX120', 'VY120',
                      'WVX10', 'WVY10',
                      'WVX20', 'WVY20',
                      'WVX40', 'WVY40',
                      'WVX80', 'WVY80',
                      'WVX120', 'WVY120']

    G5v_obshistid = encode_obshistid('G5v', filter_name, zenith, seed, 0.0)
    G5v_cat = pyfits.getdata('output/{}_cat_V.fits.gz'.format(G5v_obshistid))
    G5v_cat['ALPHAWIN_SKY'][G5v_cat['ALPHAWIN_SKY'] > 180.0] -= 360.0

    star_obshistid = encode_obshistid('star', filter_name, zenith, seed, 0.0)
    star_cat = pyfits.getdata('output/{}_cat_V.fits.gz'.format(star_obshistid))
    star_cat['ALPHAWIN_SKY'][star_cat['ALPHAWIN_SKY'] > 180.0] -= 360.0
    star_types = ['uko5v',
                  'ukb5iii',
                  'uka5v',
                  'ukf5v',
                  'ukg5v',
                  'ukk5v',
                  'ukm5v',
                  'ukg5v'] #extra G5v star to make 8

    star_diffs = {}
    # single star type for each RA
    for RA, star_type in zip(RAs, star_types):
        star_diffs[star_type] = {}
        collect_columns = {}
        for col in output_columns:
            collect_columns[col] = []
        for DEC in DECs:
            # find closest star to predicted location in both catalogs
            dist2 = (G5v_cat['ALPHAWIN_SKY'] - RA)**2 + (G5v_cat['DELTAWIN_SKY']-DEC)**2
            G5v_obj = G5v_cat[numpy.argmin(dist2)]
            dist2 = (star_cat['ALPHAWIN_SKY'] - RA)**2 + (star_cat['DELTAWIN_SKY']-DEC)**2
            star_obj = star_cat[numpy.argmin(dist2)]
            # go through output columns and append differences
            for col in output_columns:
                collect_columns[col].append(star_obj[col] - G5v_obj[col])

        # average over stars at different DECs
        for col in output_columns:
            star_diffs[star_type][col] = numpy.array(collect_columns[col]).mean()

    gal_types= ['CWW_E_ext',
                'KIN_Sa_ext',
                'KIN_Sb_ext',
                'CWW_Sbc_ext',
                'CWW_Scd_ext',
                'CWW_Im_ext',
                'KIN_SB1_ext',
                'KIN_SB6_ext']

    gal_diffs = {}
    for gal_type in gal_types:
        gal_diffs[gal_type] = {}
        for col in output_columns:
            gal_diffs[gal_type][col] = []

    with astropy.utils.console.ProgressBar(100) as bar:
        for z in numpy.arange(0.0, 3.0, 0.03):
            bar.update()
            gal_obshistid = encode_obshistid('gal', filter_name, zenith, seed, z)
            gal_cat = pyfits.getdata('output/{}_cat_V.fits.gz'.format(gal_obshistid))
            gal_cat['ALPHAWIN_SKY'][gal_cat['ALPHAWIN_SKY'] > 180.0] -= 360.0
            for RA, gal_type in zip(RAs, gal_types):
                collect_columns = {}
                for col in output_columns:
                    collect_columns[col] = []
                for DEC in DECs:
                    # find closest star to predicted location in both catalogs
                    dist2 = (G5v_cat['ALPHAWIN_SKY'] - RA)**2 + (G5v_cat['DELTAWIN_SKY']-DEC)**2
                    G5v_obj = G5v_cat[numpy.argmin(dist2)]
                    dist2 = (gal_cat['ALPHAWIN_SKY'] - RA)**2 + (gal_cat['DELTAWIN_SKY']-DEC)**2
                    gal_obj = gal_cat[numpy.argmin(dist2)]
                    # go through output columns and append differences
                    for col in output_columns:
                        collect_columns[col].append(gal_obj[col] - G5v_obj[col])
                # average over gals at different DECs
                for col in output_columns:
                    gal_diffs[gal_type][col].append(numpy.array(collect_columns[col]).mean())

    obstypeid = encode_obstypeid(filter_name, zenith, seed)
    pickle.dump((star_diffs, gal_diffs), open('relative_moments.{}.pik'.format(obstypeid), 'wb'))

#if __name__ == '__main__':
#    relative_moments('r', 30.0, 1000)
