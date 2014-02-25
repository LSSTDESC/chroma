import matplotlib.pyplot as plt
import numpy

import _mypath
import chroma

def loadcat():
    return numpy.load('indata/opsim.npy')

def lensing_visits(cat):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    X_cond = cat['airmass'] < 2.0
    return numpy.logical_and(numpy.logical_or(r_cond, i_cond), X_cond)

def plot_field(cat, field, filter_name, align_SED, target_SED, target_z, ax,
               label, **kwargs):
    wobj = (cat['fieldID'] == field) & (cat['filter'] == filter_name)

    data_dir = '../../data/'
    filter_dir = data_dir+'filters/'
    SED_dir = data_dir+'SEDs/'
    filter_file = filter_dir+'LSST_{}.dat'.format(filter_name)
    swave, sphotons = chroma.utils.get_photons(SED_dir+align_SED, filter_file, 0)
    gwave, gphotons = chroma.utils.get_photons(SED_dir+target_SED, filter_file, target_z)
    s_mom = chroma.disp_moments(swave, sphotons, zenith=numpy.pi/4.0) # tan(z) = 1
    g_mom = chroma.disp_moments(gwave, gphotons, zenith=numpy.pi/4.0) # tan(z) = 1
    delta_R = (g_mom[0] - s_mom[0]) * 180./numpy.pi * 3600 * 1000 # milliarcseconds
    delta_V = (g_mom[1] - s_mom[1]) * (180./numpy.pi * 3600)**2 # square arcseconds
    delta_ellip = delta_V / (2.0 * 0.273) * numpy.tan(cat[wobj]['z_a'])**2

    r = delta_R * numpy.tan(cat[wobj]['z_a'])
    q = cat[wobj]['q']

    x0 = r * numpy.sin(q)
    y0 = r * numpy.cos(q)
    x1 = (r + delta_ellip*2000) * numpy.sin(q)
    y1 = (r + delta_ellip*2000) * numpy.cos(q)
    for i, (x00, x11, y00, y11) in enumerate(zip(x0, x1, y0, y1)):
        if i == 0:
            ax.plot([x00, x11], [y00, y11], label=label, **kwargs)
        else:
            ax.plot([x00, x11], [y00, y11], **kwargs)
    ax.plot([-80, -80 + 2000*0.01], [80, 80], color='black')
    ax.text(-82, 83, '0.01', size=18)
#    ax.scatter(numpy.r_[x0, x1], numpy.r_[y0,y1], s=1, label=target_SED, **kwargs)
#    ax.plot(numpy.r_[x0, x1].T, numpy.r_[y0,y1].T, label=target_SED, **kwargs)

if __name__ == '__main__':

    fig = plt.figure(figsize=(5.5,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('$\Delta$ RA (mas)', fontsize=18)
    ax.set_ylabel('$\Delta$ DEC (mas)', fontsize=18)
    ax.scatter([0], [0], s=150, marker='+', color='black', linewidth=2.5)
    ax.grid()

    cat = loadcat()
    cat = cat[lensing_visits(cat)]
    plot_field(cat, 598, 'r', 'ukg5v.ascii', 'KIN_Sa_ext.ascii', 1.2, ax,
               label='Sa gal', color='blue')
    plot_field(cat, 598, 'r', 'ukg5v.ascii', 'ukm5v.ascii', 1.2, ax,
               label='M5v star', color='magenta')
    plot_field(cat, 598, 'r', 'ukg5v.ascii', 'CWW_E_ext.ascii', 1.2, ax,
               label='E gal', color='red')

    plot_field(cat, 2036, 'r', 'ukg5v.ascii', 'KIN_Sa_ext.ascii', 1.2, ax,
               label='_no_legend_', color='blue', alpha=0.5)
    plot_field(cat, 2036, 'r', 'ukg5v.ascii', 'ukm5v.ascii', 1.2, ax,
               label='_no_legend_', color='magenta', alpha=0.5)
    plot_field(cat, 2036, 'r', 'ukg5v.ascii', 'CWW_E_ext.ascii', 1.2, ax,
               label='_no_legend_', color='red', alpha=0.5)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()
    #plt.show()
    plt.savefig('output/relative_alignment.png', dpi=220)
