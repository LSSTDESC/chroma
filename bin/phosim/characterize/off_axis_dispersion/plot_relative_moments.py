import os
import pickle

import numpy
import matplotlib.pyplot as plt
import scipy.stats

import _mypath
import chroma

def encode_obstypeid(filter_name, zenith, seed):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return filter_digit + zenith_digit + seed_digit

def plot_relative_moments(filter_name, zenith, seed):
    obstypeid = encode_obstypeid(filter_name, zenith, seed)
    m_star_diff, m_gal_diff = pickle.load(open('relative_moments.{}.pik'.format(obstypeid), 'rb'))

    # Rbar plot (x)
    ###############
    M1_key = 'ALPHAWIN_SKY'

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}_x}$ (arcsec)')
    ax1.set_title('field angle = 1.6447 degrees, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        measured = m_star_diff[star][M1_key] * 3600
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        yvals = numpy.append(yvals, measured)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']

    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        measured = numpy.array(m_gal_diff[gal][M1_key]) * 3600
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        yvals = numpy.append(yvals, measured)
    ax1.legend(prop={"size":9})

    yrange = numpy.array([scipy.stats.scoreatpercentile(yvals, 1),
                          scipy.stats.scoreatpercentile(yvals, 99)])

    yspan = yrange[1] - yrange[0]
    yrange = yrange + yspan * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange[0], yrange[1])

    f.savefig('plots/Rbar_x.{}.png'.format(obstypeid))

    # Rbar plot (y)
    ###############
    M1_key = 'DELTAWIN_SKY'

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}_y}$ (arcsec)')
    ax1.set_title('field angle = 1.6447 degrees, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        measured = m_star_diff[star][M1_key] * 3600
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        yvals = numpy.append(yvals, measured)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']

    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        measured = numpy.array(m_gal_diff[gal][M1_key]) * 3600
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        yvals = numpy.append(yvals, measured)
    ax1.legend(prop={"size":9})

    yrange = numpy.array([scipy.stats.scoreatpercentile(yvals, 1),
                          scipy.stats.scoreatpercentile(yvals, 99)])

    yspan = yrange[1] - yrange[0]
    yrange = yrange + yspan * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange[0], yrange[1])

    f.savefig('plots/Rbar_y.{}.png'.format(obstypeid))

    # V plot (x), unweighted
    ########################

    M2_key = 'VX40'

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('$\Delta \mathrm{V}_x$ (arcsec$^2$)')
    ax1.set_title('field angle = 1.6447 degrees, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        measured = m_star_diff[star][M2_key] * 3600**2
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        yvals = numpy.append(yvals, measured)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        measured = numpy.array(m_gal_diff[gal][M2_key]) * 3600**2
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        yvals = numpy.append(yvals, measured)
    ax1.legend(prop={"size":9})

    yrange = numpy.array([scipy.stats.scoreatpercentile(yvals, 1),
                           scipy.stats.scoreatpercentile(yvals, 99)])

    yspan = yrange[1] - yrange[0]
    yrange = yrange + yspan * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange[0], yrange[1])

    f.savefig('plots/V_x.{}.png'.format(obstypeid))


    # V plot (y), unweighted
    ########################

    M2_key = 'VY40'

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('$\Delta \mathrm{V}_y$ (arcsec$^2$)')
    ax1.set_title('field angle = 1.6447 degrees, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        measured = m_star_diff[star][M2_key] * 3600**2
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        yvals = numpy.append(yvals, measured)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        measured = numpy.array(m_gal_diff[gal][M2_key]) * 3600**2
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        yvals = numpy.append(yvals, measured)
    ax1.legend(prop={"size":9})

    yrange = numpy.array([scipy.stats.scoreatpercentile(yvals, 1),
                           scipy.stats.scoreatpercentile(yvals, 99)])

    yspan = yrange[1] - yrange[0]
    yrange = yrange + yspan * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange[0], yrange[1])

    f.savefig('plots/V_y.{}.png'.format(obstypeid))


    # V plot (x), weighted
    ######################

    M2_key = 'WVX40'

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('$\Delta \mathrm{wV}_x$ (arcsec$^2$)')
    ax1.set_title('zenith = 30, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        measured = m_star_diff[star][M2_key] * 3600**2
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        yvals = numpy.append(yvals, measured)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        measured = numpy.array(m_gal_diff[gal][M2_key]) * 3600**2
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        yvals = numpy.append(yvals, measured)
    ax1.legend(prop={"size":9})

    yrange = numpy.array([scipy.stats.scoreatpercentile(yvals, 1),
                           scipy.stats.scoreatpercentile(yvals, 99)])

    yspan = yrange[1] - yrange[0]
    yrange = yrange + yspan * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange[0], yrange[1])

    f.savefig('plots/wV_x.{}.png'.format(obstypeid))


    # V plot (y), weighted
    ######################

    M2_key = 'WVY40'

    f = plt.figure(figsize=(8,6), dpi=100)
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 4.0)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('$\Delta \mathrm{wV}_y$ (arcsec$^2$)')
    ax1.set_title('zenith = 30, filter = {}'.format(filter_name))
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    star_names = ['O5V', 'B5III', 'A5V', 'F5V', 'G5V', 'K5V', 'M5V']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    yvals = numpy.empty(0)
    #plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        measured = m_star_diff[star][M2_key] * 3600**2
        ax1.scatter(0.0, measured, marker='*', c=star_color, s=80, label=star_name)
        yvals = numpy.append(yvals, measured)

    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue']


    #plot gals
    zs = numpy.arange(0.0, 3.0, 0.03)
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        measured = numpy.array(m_gal_diff[gal][M2_key]) * 3600**2
        ax1.scatter(zs, measured, marker='.', c=gal_color, s=50, label=gal_name)
        yvals = numpy.append(yvals, measured)
    ax1.legend(prop={"size":9})

    yrange = numpy.array([scipy.stats.scoreatpercentile(yvals, 1),
                           scipy.stats.scoreatpercentile(yvals, 99)])

    yspan = yrange[1] - yrange[0]
    yrange = yrange + yspan * numpy.array([-0.3, 0.3])

    ax1.set_ylim(yrange[0], yrange[1])

    f.savefig('plots/wV_y.{}.png'.format(obstypeid))

if __name__ == '__main__':
    plot_relative_moments('r', 0, 1000)
    plot_relative_moments('i', 0, 1000)
