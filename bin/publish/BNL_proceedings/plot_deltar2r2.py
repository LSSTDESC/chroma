import cPickle
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_stars(ax, stardata, column, band):
    star_names = ['O5v', 'B5iii', 'A5v', 'F5v', 'G5v', 'K5v', 'M5v']
    stars = ['uko5v', 'ukb5iii', 'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']
    star_colors = ['Blue', 'Cyan', 'Green', 'Gold', 'Orange', 'Red', 'Violet']

    G_idx = stardata['star_type'] == 'ukg5v'
    # plot stars
    for star, star_name, star_color in zip(stars, star_names, star_colors):
        star_idx = stardata['star_type'] == star
        S = stardata[star_idx][column][band]
        dSbyS = (S - stardata[G_idx][column][band])/S
        ax.scatter(0.0, dSbyS, c=star_color, marker='*', s=160, label=star_name, edgecolor='black',
                   zorder=3)

def plot_gals(ax, stardata, galdata, column, band):
    gal_names = ['E', 'Sa', 'Sb', 'Sbc', 'Scd', 'Im', 'SB1', 'SB6']
    gals = ['CWW_E_ext', 'KIN_Sa_ext', 'KIN_Sb_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext',
            'CWW_Im_ext', 'KIN_SB1_ext', 'KIN_SB6_ext']
    gal_colors = ['Violet', 'Red', 'Orange', 'Gold', 'Green', 'Cyan', 'Blue', 'Magenta']

    G_idx = stardata['star_type'] == 'ukg5v'
    # plot gals
    w0 = galdata['gal_type'] == 'CWW_E_ext'
    zs = galdata[w0]['redshift']
    for gal, gal_name, gal_color in zip(gals, gal_names, gal_colors):
        gal_idx = galdata['gal_type'] == gal
        S = galdata[gal_idx][column][band]
        dSbyS = (S - stardata[G_idx][column][band])/S
        ax.plot(zs, dSbyS, c=gal_color, label=gal_name)

if __name__ == '__main__':
    stardata = cPickle.load(open('../../analytic/output/stars.pkl'))
    galdata = cPickle.load(open('../../analytic/output/galaxies.pkl'))

    # LSST r-band filter
    f = plt.figure(figsize=(6,4))
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 3.0)
    ax1.set_ylim(-0.02, 0.01)
    ax1.set_ylabel('$\delta r^2_\mathrm{psf} / r^2_\mathrm{psf}$')
    ax1.set_xlabel('redshift')
    ax1.set_title('$\\alpha$ = {:3.1f}, filter = {}'.format(-0.2, 'LSST_r'), fontsize=12)
    if not os.path.exists('output/'):
        os.mkdir('output/')
    plot_stars(ax1, stardata, 'S_m02', 'LSST_r')
    plot_gals(ax1, stardata, galdata, 'S_m02', 'LSST_r')

    ax1.fill_between([-0.1, 3.0], [-0.0014, -0.0014], [0.0014, 0.0014],
                     color='grey', edgecolor='None', alpha=0.25)
    ax1.fill_between([-0.1, 3.0], [-0.0004, -0.0004], [0.0004, 0.0004],
                     color='grey', edgecolor='None', alpha=0.5)
    ax1.annotate('DES requirement', xy=(0.3, 0.0012), xycoords='data',
                 xytext=(0.2, 0.005), textcoords='data', color='grey',
                 arrowprops={'arrowstyle':'->', 'color':'grey'})
    ax1.annotate('LSST requirement', xy=(0.7, 0.0002), xycoords='data',
                 xytext=(0.6, 0.0025), textcoords='data',
                 arrowprops={'arrowstyle':'->', 'color':'black'})

    ax1.legend(prop={"size":9})
    f.tight_layout()
    f.savefig('output/dlogR2_{}.png'.format('LSST_r'), dpi=220)


    # LSST i-band filter
    f = plt.figure(figsize=(6,4))
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 3.0)
    ax1.set_ylim(-0.02, 0.01)
    ax1.set_ylabel('$\delta r^2_\mathrm{psf} / r^2_\mathrm{psf}$')
    ax1.set_xlabel('redshift')
    ax1.set_title('$\\alpha$ = {:3.1f}, filter = {}'.format(-0.2, 'LSST_i'), fontsize=12)
    if not os.path.exists('output/'):
        os.mkdir('output/')
    plot_stars(ax1, stardata, 'S_m02', 'LSST_i')
    plot_gals(ax1, stardata, galdata, 'S_m02', 'LSST_i')

    ax1.fill_between([-0.1, 3.0], [-0.0014, -0.0014], [0.0014, 0.0014],
                     color='grey', edgecolor='None', alpha=0.25)
    ax1.fill_between([-0.1, 3.0], [-0.0004, -0.0004], [0.0004, 0.0004],
                     color='grey', edgecolor='None', alpha=0.5)
    ax1.annotate('DES requirement', xy=(0.3, 0.0012), xycoords='data',
                 xytext=(0.2, 0.005), textcoords='data', color='grey',
                 arrowprops={'arrowstyle':'->', 'color':'grey'})
    ax1.annotate('LSST requirement', xy=(0.7, 0.0002), xycoords='data',
                 xytext=(0.6, 0.0025), textcoords='data',
                 arrowprops={'arrowstyle':'->', 'color':'black'})

    ax1.legend(prop={"size":9})
    f.tight_layout()
    f.savefig('output/dlogR2_{}.png'.format('LSST_i'), dpi=220)

    # Euclid 350-nm wide filter
    f = plt.figure(figsize=(6,4))
    ax1 = f.add_subplot(111)
    ax1.set_xlim(-0.1, 3.0)
    ax1.set_ylabel('$\delta r^2_\mathrm{psf} / r^2_\mathrm{psf}$')
    ax1.set_xlabel('redshift')
    ax1.set_title('$\\alpha$ = {:3.1f}, filter = {}'.format(0.6, 'Euclid_350'), fontsize=12)
    if not os.path.exists('output/'):
        os.mkdir('output/')
    plot_stars(ax1, stardata, 'S_p06', 'Euclid_350')
    plot_gals(ax1, stardata, galdata, 'S_p06', 'Euclid_350')
    ax1.fill_between([-0.1, 3.0], [-0.0016, -0.0016], [0.0016, 0.0016],
                     color='grey', edgecolor='None', alpha=0.5)
    ax1.annotate('Euclid requirement', xy=(1.0, -0.0000), xycoords='data',
                 xytext=(0.7, -0.03), textcoords='data',
                 arrowprops={'arrowstyle':'->', 'color':'black'})
    ax1.legend(prop={"size":9})
    f.tight_layout()
    f.savefig('output/dlogR2_{}.png'.format('Euclid_350'), dpi=220)
