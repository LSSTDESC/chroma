import os

import scipy.integrate
import numpy
import matplotlib.pyplot as plt

import _mypath
import chroma


def moments(s_wave, s_flux, f_wave, f_throughput, zenith, **kwargs):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    m = chroma.disp_moments(wave, photons, zenith=zenith, **kwargs)
    return m
    # gaussian_sigma = 1.0 / 2.35 # 1 arcsec FWHM -> sigma
    # m2 = chroma.weighted_second_moment(wave, photons, 1.0,
    #                                    zenith=zenith * numpy.pi / 180,
    #                                    Rbar=m[0], **kwargs)
    # return m[0], m[1], m2

def magnitude(s_wave, s_flux, f_wave, f_throughput):
    good = f_wave > 300
    wave = f_wave[good]
    f_throughput = f_throughput[good]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    photons = flambda_i * f_throughput * wave

    AB_fnu = 1.0e-26
    speed_of_light = 2.99792458e18
    AB_flambda = AB_fnu * speed_of_light / wave**2
    AB_photons = AB_flambda * f_throughput * wave

    return -2.5 * numpy.log10(scipy.integrate.simps(photons, wave) /
                              scipy.integrate.simps(AB_photons, wave))

def color(s_wave, s_flux, f_wave0, f_throughput0, f_wave1, f_throughput1):
    M0 = magnitude(s_wave, s_flux, f_wave0, f_throughput0)
    M1 = magnitude(s_wave, s_flux, f_wave1, f_throughput1)
    return M0 - M1

def pb12_color_corr(shape_filter, color_filters, zenith):
    SED_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'
    color_filter_file0 = filter_dir+'LSST_{}.dat'.format(color_filters[0])
    color_filter_file1 = filter_dir+'LSST_{}.dat'.format(color_filters[1])
    shape_filter_file = filter_dir+'LSST_{}.dat'.format(shape_filter)

    shape_f_wave, shape_f_throughput = numpy.genfromtxt(shape_filter_file).T
    color_f_wave0, color_f_throughput0 = numpy.genfromtxt(color_filter_file0).T
    color_f_wave1, color_f_throughput1 = numpy.genfromtxt(color_filter_file1).T

    #G5v is comparison SED
    G5v_wave, G5v_flux = numpy.genfromtxt(SED_dir+'ukg5v.ascii').T
    G5v_mom = moments(G5v_wave, G5v_flux, shape_f_wave, shape_f_throughput, zenith)

    #stars
    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    star_delta_Rbars = []
    star_delta_Vs = []
    star_cols = []
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        s_wave, s_flux = numpy.genfromtxt(SED_dir+star+'.ascii').T
        mom = moments(s_wave, s_flux, shape_f_wave, shape_f_throughput, zenith)
        c = color(s_wave, s_flux,
                  color_f_wave0, color_f_throughput0,
                  color_f_wave1, color_f_throughput1)

#        ax1.scatter(c, (mom[0] - G5v_mom[0]) * 180 * 3600 / numpy.pi,
#                    c=star_color, marker='*', s=80)


        star_delta_Rbars.append((mom[0] - G5v_mom[0]) * 180 * 3600 / numpy.pi)
        star_delta_Vs.append((mom[1] - G5v_mom[1]) * (180 * 3600 / numpy.pi)**2)
        star_cols.append(c)

    #galaxies
    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Gray']
    zs = numpy.arange(0.0, 3.0, 0.03)
    gal_delta_Rbars = []
    gal_delta_Vs = []
    gal_cols = []
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        g_wave, g_flux = numpy.genfromtxt(SED_dir+gal+'.ascii').T
        gal_delta_Rbar0s = []
        gal_delta_V0s = []
        gal_col0s = []
        for z in zs:
            mom = moments(g_wave * (1.0 + z), g_flux, shape_f_wave, shape_f_throughput, zenith)
            c = color(g_wave * (1.0 + z), g_flux,
                      color_f_wave0, color_f_throughput0,
                      color_f_wave1, color_f_throughput1)
            gal_delta_Rbar0s.append((mom[0] - G5v_mom[0]) * 180 * 3600 / numpy.pi)
            gal_delta_V0s.append((mom[1] - G5v_mom[1]) * (180 * 3600 / numpy.pi)**2)
            gal_col0s.append(c)
#        ax1.plot(gal_col0s, gal_delta_Rbar0s, c=gal_color)
        gal_delta_Rbars.append(gal_delta_Rbar0s)
        gal_delta_Vs.append(gal_delta_V0s)
        gal_cols.append(gal_col0s)

    A_star = numpy.vstack([star_cols[:-1], numpy.ones(len(star_cols[:-1]))]).T
    m_star_Rbar, c_star_Rbar = numpy.linalg.lstsq(A_star, star_delta_Rbars[:-1])[0]
    m_star_V, c_star_V = numpy.linalg.lstsq(A_star, star_delta_Vs[:-1])[0]

    gal_all_cols = numpy.array(gal_cols).flatten()
    gal_all_delta_Rbars = numpy.array(gal_delta_Rbars).flatten()
    gal_all_delta_Vs = numpy.array(gal_delta_Vs).flatten()
    A_gal = numpy.vstack([gal_all_cols, numpy.ones(len(gal_all_cols))]).T
    m_gal_Rbar, c_gal_Rbar = numpy.linalg.lstsq(A_gal, gal_all_delta_Rbars)[0]
    m_gal_V, c_gal_V = numpy.linalg.lstsq(A_gal, gal_all_delta_Vs)[0]

    #open Rbar plot
    f = plt.figure(figsize=(8,6), dpi=180)
    f.subplots_adjust(hspace=0)
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.set_ylabel('$\Delta \overline{\mathrm{R}}$')
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlabel('{} - {}'.format(*color_filters))

    #plot gal residuals
    for i in range(len(gals)):
        ax1.plot(gal_cols[i], gal_delta_Rbars[i], c=gal_colors[i])
        resid = (numpy.array(gal_delta_Rbars[i]) -
                 (m_gal_Rbar * numpy.array(gal_cols[i]) + c_gal_Rbar))
        ax2.plot(gal_cols[i], resid, c=gal_colors[i])

    #plot star residuals
    for i in range(len(stars)):
        ax1.scatter(star_cols[i], star_delta_Rbars[i], c=star_colors[i], marker='*', s=80)
        resid = star_delta_Rbars[i] - (m_star_Rbar * star_cols[i] + c_star_Rbar)
        ax2.scatter(star_cols[i], resid, c=star_colors[i], marker='*', s=80)

    #plot trendlines
    xlim = numpy.array(ax1.get_xlim())
    ylim = ax1.get_ylim()
    ax1.plot(xlim, xlim * m_star_Rbar + c_star_Rbar)
    ax1.plot(xlim, xlim * m_gal_Rbar + c_gal_Rbar)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)

    if not os.path.exists('output/'):
        os.mkdir('output/')

    plt.savefig('output/Rbar.{}.{}-{}.z{:02d}.png'.format(shape_filter,
                                                          color_filters[0],
                                                          color_filters[1],
                                                          int(round(zenith * 180 / numpy.pi))))

    #open V plot
    f = plt.figure(figsize=(8,6), dpi=180)
    f.subplots_adjust(hspace=0)
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax1.set_ylabel('$\Delta \mathrm{V}$')
    ax2 = plt.subplot2grid((3,1), (2,0))
    ax2.set_xlabel('{} - {}'.format(*color_filters))

    #plot gal residuals
    for i in range(len(gals)):
        ax1.plot(gal_cols[i], gal_delta_Vs[i], c=gal_colors[i])
        resid = (numpy.array(gal_delta_Vs[i]) -
                 (m_gal_V * numpy.array(gal_cols[i]) + c_gal_V))
        ax2.plot(gal_cols[i], resid, c=gal_colors[i])

    #plot star residuals
    for i in range(len(stars)):
        ax1.scatter(star_cols[i], star_delta_Vs[i], c=star_colors[i], marker='*', s=80)
        resid = star_delta_Vs[i] - (m_star_V * star_cols[i] + c_star_V)
        ax2.scatter(star_cols[i], resid, c=star_colors[i], marker='*', s=80)

    #plot trendlines
    xlim = numpy.array(ax1.get_xlim())
    ylim = ax1.get_ylim()
    ax1.plot(xlim, xlim * m_star_V + c_star_V)
    ax1.plot(xlim, xlim * m_gal_V + c_gal_V)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)

    if not os.path.exists('output/'):
        os.mkdir('output/')

    plt.savefig('output/V.{}.{}-{}.z{:02d}.png'.format(shape_filter,
                                                       color_filters[0],
                                                       color_filters[1],
                                                       int(round(zenith * 180 / numpy.pi))))

#if __name__ == '__main__':
#    pb12_color_corr(['g', 'r'], 'r', 30.0 * numpy.pi / 180)
