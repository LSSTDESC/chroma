import os

import numpy
import astropy.utils.console
import scipy.integrate
import matplotlib.pyplot as plt

import _mypath
import chroma


def get_r2(s_wave, s_flux, f_wave, f_throughput, alpha=-0.2):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    r2 = chroma.relative_second_moment_radius(wave, photons, alpha=alpha)
    return r2

def compute_second_moment_radii(filter_name, alpha=-0.2):
    spec_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'

    f_data = numpy.genfromtxt(filter_dir + '{}.dat'.format(filter_name))
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    G5v_data = numpy.genfromtxt(spec_dir + 'ukg5v.ascii')
    G5v_wave, G5v_flambda = G5v_data[:,0], G5v_data[:,1]
    G5v_r2 = get_r2(G5v_wave, G5v_flambda, f_wave, f_throughput, alpha)

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

        r2 = get_r2(wave, flambda, f_wave, f_throughput, alpha)
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
    with astropy.utils.console.ProgressBar(100 * len(gal_types)) as bar:
        for gal_type in gal_types:
            gal_diffs[gal_type] = {'dlogr2':[]}
            SED_data = numpy.genfromtxt(spec_dir + gal_type + '.ascii')
            wave, flambda = SED_data[:,0], SED_data[:,1]
            for z in numpy.arange(0.0, 3.0, 0.03):
                r2 = get_r2(wave * (1.0 + z), flambda, f_wave, f_throughput, alpha)
                gal_diffs[gal_type]['dlogr2'].append( numpy.log(r2 / G5v_r2))
                bar.update()
    return star_diffs, gal_diffs

def plot_size_error(filter_name, alpha, yrange=None):
    a_star_diff, a_gal_diff = compute_second_moment_radii(filter_name, alpha)

    f = plt.figure(figsize=(6,4))
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 3.0)
    if yrange is not None:
        ax1.set_ylim(yrange)
    ax1.set_ylabel('$\delta r^2_\mathrm{psf} / r^2_\mathrm{psf}$')
    ax1.set_xlabel('redshift')
    ax1.set_title('$\\alpha$ = {:3.1f}, filter = {}'.format(alpha, filter_name), fontsize=12)
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        val = a_star_diff[star]['dlogr2']
        ax1.scatter(0.0, val, c=star_color, marker='*', s=160, label=star_name, edgecolor='black')

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Magenta']

    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        vals = a_gal_diff[gal]['dlogr2']
        ax1.plot(zs, vals, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    f.tight_layout()
    f.savefig('output/dlogR2_{}.png'.format(filter_name), dpi=300)

if __name__ == '__main__':
    plot_size_error('LSST_r', -0.2, yrange=[-0.02, 0.01])
    plot_size_error('Euclid_350', 0.6)
