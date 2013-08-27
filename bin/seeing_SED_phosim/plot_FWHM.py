import pickle

import numpy
import matplotlib.pyplot as plt

def plot_FWHM():
    # r-band
    G5v = pickle.load(open('pickles/G5v.r.pik', 'r'))

    # stars first
    stars = pickle.load(open('pickles/star.r.pik', 'r'))

    star_types = ['uko5v',
                  'ukb5iii',
                  'uka5v',
                  'ukf5v',
                  'ukg5v',
                  'ukk5v',
                  'ukm5v',
                  'ukg5v'] #extra G5v star to make 8
    star_vals = {}
    for star_type in star_types:
        w = (stars['type'] == star_type)
        d_ln_r2 = numpy.log(stars[w]['fwhm_x'] * stars[w]['fwhm_y']
                            / G5v[w]['fwhm_x'] / G5v[w]['fwhm_y'])
        star_vals[star_type] = (numpy.mean(d_ln_r2), numpy.std(d_ln_r2))

    gal_types = ['E',
                 'Sa',
                 'Sb',
                 'Sbc',
                 'Scd',
                 'Im',
                 'SB1',
                 'SB6']

    gal_vals = {}
    for gal_type in gal_types:
        gal_vals[gal_type] = numpy.empty(100, dtype=[('z', 'f4'),
                                                     ('fwhm', 'f4'),
                                                     ('err', 'f4')])
    i=0
    for z in numpy.arange(0.0, 3.0, 0.03):
        gals = pickle.load(open('pickles/gal.r.{:02d}.pik'.format(int(round(z / 0.03))), 'r'))
        for gal_type in gal_types:
            if len(gals) == 3: continue
            w = (gals['type'] == gal_type)
            d_ln_r2 = numpy.log(gals[w]['fwhm_x'] * gals[w]['fwhm_y']
                                / G5v[w]['fwhm_x'] / G5v[w]['fwhm_y'])
            gal_vals[gal_type][i]['z'] = z
            gal_vals[gal_type][i]['fwhm'] = numpy.mean(d_ln_r2)
            gal_vals[gal_type][i]['err'] = numpy.std(d_ln_r2)
        i += 1

    f=plt.figure(figsize=(8,6), dpi=100)
    ax=f.add_subplot(111)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlim(-0.1, 3.1)
    for star_type in star_types:
        ax.scatter(0.0, star_vals[star_type][0], label=star_type)
        ax.errorbar(0.0, star_vals[star_type][0], star_vals[star_type][1], label=star_type)

    for gal_type in gal_types:
        ax.scatter(gal_vals[gal_type]['z'], gal_vals[gal_type]['fwhm'])
#        ax.errorbar(gal_vals[gal_type]['z'], gal_vals[gal_type]['fwhm'], gal_vals[gal_type]['err'])

    f.savefig('plots/rband.pdf')

#     for i in range(64):
#         print stars[i]['type'], G5v[i]['fwhm_x'], stars[i]['fwhm_x']

if __name__ == '__main__':
    plot_FWHM()
