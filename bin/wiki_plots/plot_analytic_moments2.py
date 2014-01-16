import os
import pickle

import numpy
import astropy.utils.console
import matplotlib.pyplot as plt
import scipy.stats

import _mypath
import chroma

hist_axes_range = [0.15, 0.12, 0.1, 0.8]
scatter_axes_range = [0.25, 0.12, 0.55, 0.8]
colorbar_axes_range = [0.85, 0.12, 0.04, 0.8]

def moments(s_wave, s_flux, f_wave, f_throughput, zenith, **kwargs):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    m = chroma.disp_moments(wave, photons, zenith=zenith * numpy.pi / 180.0, **kwargs)
    gaussian_sigma = 1.0 / 2.35 # 1 arcsec FWHM -> sigma
    m2 = chroma.weighted_second_moment(wave, photons, 1.0,
                                       zenith=zenith * numpy.pi / 180.0,
                                       Rbar=m[0], **kwargs)
    return m[0], m[1], m2


def compute_relative_moments(filter_name, zenith, **kwargs):
    spec_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'

    f_data = numpy.genfromtxt(filter_dir + '{}.dat'.format(filter_name))
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    G5v_data = numpy.genfromtxt(spec_dir + 'ukg5v.ascii')
    G5v_wave, G5v_flambda = G5v_data[:,0], G5v_data[:,1]
    G5v_mom = moments(G5v_wave, G5v_flambda, f_wave, f_throughput, zenith, **kwargs)

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

        m = moments(wave, flambda, f_wave, f_throughput, zenith, **kwargs)
        star_diffs[star_type]['M1'] = (m[0] - G5v_mom[0]) * 180 / numpy.pi * 3600 # rad -> arcsec
        # rad^2 -> arcsec^2
        star_diffs[star_type]['M2'] = (m[1] - G5v_mom[1]) * (180 / numpy.pi * 3600)**2
        star_diffs[star_type]['wM2'] = (m[2] - G5v_mom[2]) * (180 / numpy.pi * 3600)**2

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
            gal_diffs[gal_type] = {'M1':[], 'M2':[], 'wM2':[]}
            SED_data = numpy.genfromtxt(spec_dir + gal_type + '.ascii')
            wave, flambda = SED_data[:,0], SED_data[:,1]
            for z in numpy.arange(0.0, 3.0, 0.03):
                bar.update()
                m = moments(wave * (1.0 + z), flambda, f_wave, f_throughput, zenith, **kwargs)
                # rad -> arcsec, rad^2 -> arcsec^2
                gal_diffs[gal_type]['M1'].append((m[0] - G5v_mom[0]) * 180 / numpy.pi * 3600)
                gal_diffs[gal_type]['M2'].append((m[1] - G5v_mom[1]) * (180 / numpy.pi * 3600)**2)
                gal_diffs[gal_type]['wM2'].append((m[2] - G5v_mom[2]) * (180 / numpy.pi * 3600)**2)
    return star_diffs, gal_diffs

def encode_obstypeid(filter_name, zenith, seed):
    filter_number = {'LSST_u':'0','LSST_g':'1','LSST_r':'2','LSST_i':'3','LSST_z':'4','LSST_y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return filter_digit + zenith_digit + seed_digit

def plot_analytic_moments(filter_name, zenith, seed):
    a_star_diff, a_gal_diff = compute_relative_moments(filter_name, zenith)
    obstypeid = encode_obstypeid(filter_name, zenith, seed)

    # Rbar plot
    ###########

    f = plt.figure(figsize=(8,5))
    ax1 = f.add_axes(scatter_axes_range)
    ax1.set_xlim(-0.1, 3.0)
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}$ (arcsec)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = 30, filter = {}'.format(filter_name), fontsize=12)
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals1 = numpy.empty(0)
    yvals2 = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['M1']
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, label=star_name)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['M1']
        ax1.plot(zs, analytic, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    f.savefig('output/Rbar.{}.png'.format(obstypeid), dpi=300)

    # V plot, unweighted
    ####################

    f = plt.figure(figsize=(8,5))
    ax1 = f.add_axes(scatter_axes_range)
    ax1.set_xlim(-0.1, 3.0)
    ax1.set_ylabel('$\Delta \mathrm{V}$ (arcsec$^2$)')
    ax1.set_xlabel('redshift')
    ax1.set_title('zenith angle = 30, filter = {}'.format(filter_name), fontsize=12)
    if not os.path.exists('output/'):
        os.mkdir('output/')

    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['M2']
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, label=star_name)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['M2']
        ax1.plot(zs, analytic, c=gal_color, label=gal_name)
    ax1.legend(prop={"size":9})

    ax1.set_ylim(-0.0008, 0.0005)
    f.savefig('output/V.{}.png'.format(obstypeid), dpi=300)

if __name__ == '__main__':
    plot_analytic_moments('LSST_r', 30.0, 1000)
    plot_analytic_moments('LSST_i', 30.0, 1000)
