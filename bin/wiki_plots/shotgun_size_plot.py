import os

import numpy
import astropy.utils.console
import scipy.integrate
import matplotlib.pyplot as plt

import _mypath
import chroma

def get_r2(s_wave, s_flux, f_wave, f_throughput, n=-0.2):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    return (scipy.integrate.simps(photons * (wave/500) ** (2*n), wave) /
            scipy.integrate.simps(photons, wave))

def compute_second_moment_radii(filter_name, n=-0.2):
    spec_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'

    f_data = numpy.genfromtxt(filter_dir + 'LSST_{}.dat'.format(filter_name))
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    G5v_data = numpy.genfromtxt(spec_dir + 'ukg5v.ascii')
    G5v_wave, G5v_flambda = G5v_data[:,0], G5v_data[:,1]
    G5v_r2 = get_r2(G5v_wave, G5v_flambda, f_wave, f_throughput, n)

    star_types = ['uko5v',
                  'uka5v',
                  'ukg5v',
                  'ukm5v']
    star_diffs = {}
    for star_type in star_types:
        star_diffs[star_type] = {}
        SED_data = numpy.genfromtxt(spec_dir + star_type + '.ascii')
        wave, flambda = SED_data[:,0], SED_data[:,1]

        r2 = get_r2(wave, flambda, f_wave, f_throughput, n)
        star_diffs[star_type]['dlogr2'] = numpy.log(r2 / G5v_r2)

    gal_types= ['CWW_E_ext',
                'KIN_Sb_ext',
                'CWW_Im_ext',
                'KIN_SB6_ext']

    gal_diffs = {}
    with astropy.utils.console.ProgressBar(100 * len(gal_types)) as bar:
        for gal_type in gal_types:
            gal_diffs[gal_type] = {'dlogr2':[]}
            SED_data = numpy.genfromtxt(spec_dir + gal_type + '.ascii')
            wave, flambda = SED_data[:,0], SED_data[:,1]
            for z in numpy.arange(0.0, 3.0, 0.03):
                r2 = get_r2(wave * (1.0 + z), flambda, f_wave, f_throughput, n)
                gal_diffs[gal_type]['dlogr2'].append( numpy.log(r2 / G5v_r2))
                bar.update()
    return star_diffs, gal_diffs

def shotgun_size_plot(filter_name):
    a_star_diff, a_gal_diff = compute_second_moment_radii(filter_name, -0.2)

    f = plt.figure(figsize=(8,3.0), dpi=100)
    ax1 = plt.subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_ylabel('$\delta(\mathrm{ln}\, r_{PSF}^2)$')
    ax1.set_xlabel('redshift')
    ax1.set_title('filter = {}'.format(filter_name))
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'A5v', 'G5v', 'M5v']
    stars = ['uko5v', 'uka5v', 'ukg5v', 'ukm5v']
    star_colors = ['Violet', 'Cyan', 'Gold', 'Red']

    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        val = a_star_diff[star]['dlogr2']
        ax1.scatter(0.0, val, c=star_color, marker='*', s=160, label=star_name, zorder=10)

    gal_names = ['E', 'Sb', 'Im', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sb_ext', 'CWW_Im_ext', 'KIN_SB6_ext']
    gal_colors = ['Magenta', 'Orange', 'Blue', 'Green']

    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        vals = a_gal_diff[gal]['dlogr2']
        ax1.plot(zs, vals, c=gal_color, label=gal_name, zorder=9)
    ax1.legend(prop={"size":9})

    ax1.fill_between([-0.1, 4.0], [-0.0004, -0.0004], [0.0004, 0.0004], facecolor='gray',
                     edgecolor="none", alpha=0.4)
    ax1.fill_between([-0.1, 4.0], [-0.004, -0.004], [0.004, 0.004], facecolor='gray',
                     edgecolor="none", alpha=0.2)

    f.tight_layout()
    f.savefig('output/dlogR2.shotgun.{}.pdf'.format(filter_name))

if __name__ == '__main__':
    shotgun_size_plot('r')
    shotgun_size_plot('i')
