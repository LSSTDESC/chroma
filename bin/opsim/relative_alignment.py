import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import galsim

import _mypath
import chroma

def loadcat():
    return np.load('indata/opsim.npy')

def lensing_visits(cat):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    X_cond = cat['airmass'] < 1.5
    return (r_cond | i_cond) & X_cond

def plot_field(cat, field, filter_name, align_SED_file, target_SED_file, target_z, ax,
               label, **kwargs):
    wobj = (cat['fieldID'] == field) & (cat['filter'] == filter_name)

    data_dir = '../../data/'
    filter_dir = data_dir+'filters/'
    SED_dir = data_dir+'SEDs/'
    align_SED = galsim.SED(SED_dir+align_SED_file)
    target_SED = galsim.SED(SED_dir+target_SED_file)
    bandpass = galsim.Bandpass(filter_dir+"LSST_{}.dat".format(filter_name))

    # align_SED = chroma.SED(SED_dir+align_SED_file)
    # target_SED = chroma.SED(SED_dir+target_SED_file).atRedshift(target_z)
    # bandpass = chroma.Bandpass(filter_dir+'LSST_{}.dat'.format(filter_name))

    align_moments = align_SED.calculateDCRMomentShifts(bandpass,
                                                       zenith_angle = np.pi/4.0 * galsim.radians)
    target_moments = target_SED.calculateDCRMomentShifts(bandpass,
                                                         zenith_angle = np.pi/4.0 * galsim.radians)

    # milliarcseconds
    delta_R = (target_moments[0][1,0] - align_moments[0][1,0]) * 180./np.pi * 3600 * 1000
    # square arcseconds
    delta_V = (target_moments[1][1,1] - align_moments[1][1,1]) * (180./np.pi * 3600)**2

    # delta_R = (target_moments[0] - align_moments[0]) * 180./np.pi * 3600 * 1000 # milliarcseconds
    # delta_V = (target_moments[1] - align_moments[1]) * (180./np.pi * 3600)**2 # square arcseconds

    rsquared = 0.4**2 # square arcseconds
    m = delta_V * np.tan(cat[wobj]['z_a'])**2 / rsquared

    r = delta_R * np.tan(cat[wobj]['z_a'])
    q = cat[wobj]['q']

    x0 = r * np.sin(q)
    y0 = r * np.cos(q)
    x1 = (r + m*2000) * np.sin(q)
    y1 = (r + m*2000) * np.cos(q)
    for i, (x00, x11, y00, y11) in enumerate(zip(x0, x1, y0, y1)):
        if i == 0:
            ax.plot([x00, x11], [y00, y11], label=label, **kwargs)
        else:
            ax.plot([x00, x11], [y00, y11], **kwargs)
    ax.plot([-80, -80 + 2000*0.01], [80, 80], color='black')
    ax.text(-82, 83, 'm=0.01', size=14)

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
    import os
    if not os.path.isdir('output/'):
        os.mkdir('output/')
    plt.savefig('output/relative_alignment.png', dpi=220)
