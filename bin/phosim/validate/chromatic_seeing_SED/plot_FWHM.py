import pickle

import numpy
import matplotlib.pyplot as plt

import _mypath
import chroma

def get_r2(s_wave, s_flux, f_wave, f_throughput):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    r2 = chroma.relative_second_moment_radius(wave, photons)
    return r2

def compute_second_moment_radii(filter_name):
    spec_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'

    f_data = numpy.genfromtxt(filter_dir + 'LSST_{}.dat'.format(filter_name))
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    G5v_data = numpy.genfromtxt(spec_dir + 'ukg5v.ascii')
    G5v_wave, G5v_flambda = G5v_data[:,0], G5v_data[:,1]
    G5v_r2 = get_r2(G5v_wave, G5v_flambda, f_wave, f_throughput)

    star_types = ['uko5v',
                  'ukb5iii',
                  'uka5v',
                  'ukf5v',
                  'ukg5v',
                  'ukk5v',
                  'ukm5v',
                  'ukg5v'] #extra G5v star to make 8
    star_diffs = {}
    for star_type in star_types:
        star_diffs[star_type] = {}
        SED_data = numpy.genfromtxt(spec_dir + star_type + '.ascii')
        wave, flambda = SED_data[:,0], SED_data[:,1]

        r2 = get_r2(wave, flambda, f_wave, f_throughput)
        star_diffs[star_type]['dlogr2'] = numpy.log(r2 / G5v_r2)

    gal_types= ['CWW_E_ext',
                'KIN_Sa_ext',
                'KIN_Sb_ext',
                'CWW_Sbc_ext',
                'CWW_Scd_ext',
                'CWW_Im_ext',
                'KIN_SB1_ext',
                'KIN_SB6_ext']

    gal_diffs = {}
    with chroma.ProgressBar(100 * len(gal_types)) as bar:
        for gal_type in gal_types:
            gal_diffs[gal_type] = {'dlogr2':[]}
            SED_data = numpy.genfromtxt(spec_dir + gal_type + '.ascii')
            wave, flambda = SED_data[:,0], SED_data[:,1]
            for z in numpy.arange(0.0, 3.0, 0.03):
                r2 = get_r2(wave * (1.0 + z), flambda, f_wave, f_throughput)
                gal_diffs[gal_type]['dlogr2'].append( numpy.log(r2 / G5v_r2))
                bar.update()
    return star_diffs, gal_diffs

def plot_FWHM():
    data = pickle.load(open('data.pik'))
    G5vlib, starlib, gallib = data

    star_types = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    #r-band
    f=plt.figure(figsize=(8,6), dpi=100)
    ax=f.add_subplot(111)
    ax.set_ylim(-0.02, 0.01)
    ax.set_xlim(-0.1, 3.1)
    ax.set_xlabel('redshift')
    ax.set_ylabel('$\delta(\mathrm{ln}\, r^2_\mathrm{PSF})$')

    for i in range(len(star_types)):
        w = (starlib['type'] == star_types[i]) & (starlib['filter'] == 'r')
        star_r2 = starlib[w]['FWHM']
        G5v_r2 = G5vlib[w]['FWHM']
        d_ln_r2 = numpy.log(numpy.mean(star_r2 / G5v_r2))
        ax.scatter(0.0, d_ln_r2, label=star_names[i], color=star_colors[i], marker='*', s=40)


    gal_types = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Magenta']

    for i in range(len(gal_types)):
        w = (gallib['type'] == gal_types[i]) & (gallib['filter'] == 'r')
        gal_r2 = gallib[w]['FWHM']
        G5v_r2 = G5vlib[i]['FWHM']
        d_ln_r2 = numpy.empty(100, dtype='f4')
        for j in range(100):
            d_ln_r2[j] = numpy.log(numpy.mean(gal_r2[j,:] / G5v_r2))
        ax.scatter(gallib[w]['z'], d_ln_r2, label=gal_types[i], color=gal_colors[i], s=10)

    ax.legend(prop={"size":9})

    # theory
    a_star_diff, a_gal_diff = compute_second_moment_radii('r')
    #plot stars
    for star_type, star_name, star_color in zip(star_types, star_names, star_colors):
        val = a_star_diff[star_type]['dlogr2']
        ax.scatter(0.0, val, c=star_color, marker='*', s=160, label=star_name)
    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    gal_long_types = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
                      'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    for gal_type, gal_long_type, gal_color in zip(gal_types, gal_long_types, gal_colors):
        vals = a_gal_diff[gal_long_type]['dlogr2']
        ax.plot(zs, vals, c=gal_color, label=gal_type)

    f.savefig('plots/rband.pdf')

if __name__ == '__main__':
    plot_FWHM()
