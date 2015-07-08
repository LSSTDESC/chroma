import os
import pickle

import numpy
import matplotlib.pyplot as plt
import scipy.stats

import _mypath
import chroma


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

    f_data = numpy.genfromtxt(filter_dir + 'LSST_{}.dat'.format(filter_name))
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    G5V_data = numpy.genfromtxt(spec_dir + 'ukg5v.ascii')
    G5V_wave, G5V_flambda = G5V_data[:,0], G5V_data[:,1]
    G5V_mom = moments(G5V_wave, G5V_flambda, f_wave, f_throughput, zenith, **kwargs)

    star_types = ['uko5v',
                  'ukb5iii',
                  'uka5v',
                  'ukf5v',
                  'ukg5v',
                  'ukk5v',
                  'ukm5v',
                  'ukg5v'] #extra G5V star to make 8
    star_diffs = {}
    for star_type in star_types:
        star_diffs[star_type] = {}
        SED_data = numpy.genfromtxt(spec_dir + star_type + '.ascii')
        wave, flambda = SED_data[:,0], SED_data[:,1]

        m = moments(wave, flambda, f_wave, f_throughput, zenith, **kwargs)
        star_diffs[star_type]['M1'] = (m[0] - G5V_mom[0]) * 180 / numpy.pi * 3600 # rad -> arcsec
        # rad^2 -> arcsec^2
        star_diffs[star_type]['M2'] = (m[1] - G5V_mom[1]) * (180 / numpy.pi * 3600)**2
        star_diffs[star_type]['wM2'] = (m[2] - G5V_mom[2]) * (180 / numpy.pi * 3600)**2

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
            gal_diffs[gal_type] = {'M1':[], 'M2':[], 'wM2':[]}
            SED_data = numpy.genfromtxt(spec_dir + gal_type + '.ascii')
            wave, flambda = SED_data[:,0], SED_data[:,1]
            for z in numpy.arange(0.0, 3.0, 0.03):
                bar.update()
                m = moments(wave * (1.0 + z), flambda, f_wave, f_throughput, zenith, **kwargs)
                # rad -> arcsec, rad^2 -> arcsec^2
                gal_diffs[gal_type]['M1'].append((m[0] - G5V_mom[0]) * 180 / numpy.pi * 3600)
                gal_diffs[gal_type]['M2'].append((m[1] - G5V_mom[1]) * (180 / numpy.pi * 3600)**2)
                gal_diffs[gal_type]['wM2'].append((m[2] - G5V_mom[2]) * (180 / numpy.pi * 3600)**2)
    return star_diffs, gal_diffs

def encode_obstypeid(filter_name, zenith, seed):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return filter_digit + zenith_digit + seed_digit

def plot_relative_moments(filter_name, zenith, seed):
    a_star_diff, a_gal_diff = compute_relative_moments(filter_name, zenith)
    obstypeid = encode_obstypeid(filter_name, zenith, seed)
    m_star_diff, m_gal_diff = pickle.load(open('relative_moments.{}.pik'.format(obstypeid), 'rb'))

    M1_key = 'DELTAWIN_SKY'

    # Rbar plot
    ###########

    f = plt.figure(figsize=(8,6), dpi=100)
    f.subplots_adjust(hspace=0)
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(-0.1, 4.0)
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlim(-0.1, 4.0)
    ax2.set_xlabel('Redshift')
    ax2.set_ylabel('measured - analytic')
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}$ (arcsec)')
    ax1.set_title('zenith = 30, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals1 = numpy.empty(0)
    yvals2 = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['M1']
        measured = m_star_diff[star][M1_key] * 3600
        diff = measured - analytic
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, alpha=0.4)
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        ax2.scatter(0.0, diff, marker='*', s=80, c=star_color)
        yvals1 = numpy.append(yvals1, measured)
        yvals1 = numpy.append(yvals1, analytic)
        yvals2 = numpy.append(yvals2, diff)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['M1']
        measured = numpy.array(m_gal_diff[gal][M1_key]) * 3600
        diff = measured - analytic
        ax1.plot(zs, analytic, c=gal_color)
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        ax2.plot(zs, diff, c=gal_color)
        yvals1 = numpy.append(yvals1, measured)
        yvals1 = numpy.append(yvals1, analytic)
        yvals2 = numpy.append(yvals2, diff)
    ax1.legend(prop={"size":9})

    yrange1 = numpy.array([scipy.stats.scoreatpercentile(yvals1, 1),
                           scipy.stats.scoreatpercentile(yvals1, 99)])
    yrange2 = numpy.array([scipy.stats.scoreatpercentile(yvals2, 1),
                           scipy.stats.scoreatpercentile(yvals2, 99)])

    yspan1 = yrange1[1] - yrange1[0]
    yspan2 = yrange2[1] - yrange2[0]
    yrange1 = yrange1 + yspan1 * numpy.array([-0.3, 0.3])
    yrange2 = yrange2 + yspan2 * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange1[0], yrange1[1])
    ax2.set_ylim(yrange2[0], yrange2[1])

    f.savefig('plots/Rbar.{}.png'.format(obstypeid))

    # V plot, unweighted
    ####################

    M2_key = 'VY40'

    f = plt.figure(figsize=(8,6), dpi=100)
    f.subplots_adjust(hspace=0)
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(-0.1, 4.0)
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlim(-0.1, 4.0)
    ax2.set_xlabel('Redshift')
    ax2.set_ylabel('measured - analytic')
    ax1.set_ylabel('$\Delta \mathrm{V}$ (arcsec$^2$)')
    ax1.set_title('zenith = 30, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals1 = numpy.empty(0)
    yvals2 = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['M2']
        measured = m_star_diff[star][M2_key] * 3600**2
        diff = measured - analytic
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, alpha=0.4)
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        ax2.scatter(0.0, diff, marker='*', s=80, c=star_color)
        yvals1 = numpy.append(yvals1, measured)
        yvals1 = numpy.append(yvals1, analytic)
        yvals2 = numpy.append(yvals2, diff)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['M2']
        measured = numpy.array(m_gal_diff[gal][M2_key]) * 3600**2
        diff = measured - analytic
        ax1.plot(zs, analytic, c=gal_color)
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        ax2.plot(zs, diff, c=gal_color)
        yvals1 = numpy.append(yvals1, measured)
        yvals1 = numpy.append(yvals1, analytic)
        yvals2 = numpy.append(yvals2, diff)
    ax1.legend(prop={"size":9})

    yrange1 = numpy.array([scipy.stats.scoreatpercentile(yvals1, 1),
                           scipy.stats.scoreatpercentile(yvals1, 99)])
    yrange2 = numpy.array([scipy.stats.scoreatpercentile(yvals2, 1),
                           scipy.stats.scoreatpercentile(yvals2, 99)])

    yspan1 = yrange1[1] - yrange1[0]
    yspan2 = yrange2[1] - yrange2[0]
    yrange1 = yrange1 + yspan1 * numpy.array([-0.3, 0.3])
    yrange2 = yrange2 + yspan2 * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange1[0], yrange1[1])
    ax2.set_ylim(yrange2[0], yrange2[1])

    f.savefig('plots/V.{}.png'.format(obstypeid))


    # V plot, weighted
    ####################

    M2_key = 'WVY40'

    f = plt.figure(figsize=(8,6), dpi=100)
    f.subplots_adjust(hspace=0)
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(-0.1, 4.0)
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlim(-0.1, 4.0)
    ax2.set_xlabel('Redshift')
    ax2.set_ylabel('measured - analytic')
    ax1.set_ylabel('$\Delta \mathrm{wV}$ (arcsec$^2$)')
    ax1.set_title('zenith = 30, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals1 = numpy.empty(0)
    yvals2 = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        analytic = a_star_diff[star]['wM2']
        measured = m_star_diff[star][M2_key] * 3600**2
        diff = measured - analytic
        ax1.scatter(0.0, analytic, c=star_color, marker='*', s=160, alpha=0.4)
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        ax2.scatter(0.0, diff, marker='*', s=80, c=star_color)
        yvals1 = numpy.append(yvals1, measured)
        yvals1 = numpy.append(yvals1, analytic)
        yvals2 = numpy.append(yvals2, diff)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        analytic = a_gal_diff[gal]['wM2']
        measured = numpy.array(m_gal_diff[gal][M2_key]) * 3600**2
        diff = measured - analytic
        ax1.plot(zs, analytic, c=gal_color)
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        ax2.plot(zs, diff, c=gal_color)
        yvals1 = numpy.append(yvals1, measured)
        yvals1 = numpy.append(yvals1, analytic)
        yvals2 = numpy.append(yvals2, diff)
    ax1.legend(prop={"size":9})

    yrange1 = numpy.array([scipy.stats.scoreatpercentile(yvals1, 1),
                           scipy.stats.scoreatpercentile(yvals1, 99)])
    yrange2 = numpy.array([scipy.stats.scoreatpercentile(yvals2, 1),
                           scipy.stats.scoreatpercentile(yvals2, 99)])

    yspan1 = yrange1[1] - yrange1[0]
    yspan2 = yrange2[1] - yrange2[0]
    yrange1 = yrange1 + yspan1 * numpy.array([-0.3, 0.3])
    yrange2 = yrange2 + yspan2 * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange1[0], yrange1[1])
    ax2.set_ylim(yrange2[0], yrange2[1])

    f.savefig('plots/wV.{}.png'.format(obstypeid))

if __name__ == '__main__':
    plot_relative_moments('r', 30.0, 1000)
    plot_relative_moments('i', 30.0, 1000)
