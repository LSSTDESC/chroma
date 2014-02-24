# Make some plots of atmospheric refraction vs. wavelength for different zenith angles.
# Overplot the LSST filter bandpasses for reference.

import os

import numpy as np
import matplotlib.pyplot as plt

import _mypath
import chroma

datadir = '../../data/'

def refraction_vs_zenith():
    zeniths = np.linspace(0, 70, 71) # degrees
    waves = np.arange(300, 1101, 200, dtype=np.float64) # nm

    # open figure for output
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.set_title('Atmospheric Refraction')
    ax.set_xlabel('zenith angle (degrees)')
    ax.set_xlim(0, 70)
    ax.set_ylabel('Refraction (arcsec)')
    ax.set_ylim(0, 100)

    for wave in waves:
        refrac_angles = np.array([chroma.get_refraction(wave, z * np.pi/180) for z in zeniths])
        ax.plot(zeniths, refrac_angles * 3600 * 180/np.pi, label=str(wave)+' nm')
    ax.legend(fontsize='small', title='wavelength', loc='upper left')
    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fig.savefig('output/refraction_vs_zenith.png', dpi=220)

def chromatic_biases():
    zeniths = [10, 20, 30, 40, 50]
    waves = np.arange(300, 1101, 200, dtype=np.float64)

    fig = plt.figure(figsize=(8,5))
    #ax = fig.add_subplot(111)
    ax = fig.add_axes([0.14, 0.13, 0.76, 0.78])
    ax.set_xlabel('Wavelength (nm)', fontsize=18)
    ax.set_xlim(300, 1200)
    ax.set_ylabel('Relative refraction (arcsec)', fontsize=18)
    ax.set_ylim(-1, 1.)

    for zenith in zeniths:
        # chroma expects angles in radians, so need to convert deg to radians
        refrac_angle = chroma.get_refraction(waves, zenith * np.pi/180)
        refrac_ref = refrac_angle[np.argmin(abs(waves - 500))]
        # chroma output is also in radians, so convert to arcsec here
        ax.plot(waves, (refrac_angle - refrac_ref) * 206265, label=str(zenith)+' deg')

    # 350nm wide Euclid filter.
    ax.fill_between([0., 550., 550., 900., 900., 1200.], [-1, -1, 0.25, 0.25, -1, -1], -1,
                    color='black', alpha=0.15)

    colors = ['purple', 'blue', 'green', 'gold', 'magenta', 'red']
    for i, filter_ in enumerate('ugrizy'):
        # filters are stored in two columns: wavelength (nm), and throughput
        fdata = chroma.SampledBandpass(datadir+'filters/LSST_{}.dat'.format(filter_))
        fwave, throughput = fdata.interp.x, fdata.interp.y
        ax.fill_between(fwave, throughput * 2.5 - 1, -1, color=colors[i], alpha=0.3)
    # Add in lambda^(-2/5) for chromatic seeing comparison integrand comparison
    ax2 = ax.twinx()
    ys = (waves/500.0)**(-2./5)
    ax2.plot(waves, ys, 'k', lw=3, label='$\lambda^{-2/5}$')
    ax.legend(fontsize=11, title='zenith angle')
    ax2.legend(fontsize=11, title='chromatic seeing', loc='upper right', bbox_to_anchor = (0.78, 1))
    ax2.set_xlim(300, 1100)
    ax2.set_ylabel('Relative $r^2_\mathrm{PSF}$', fontsize=18)

    for label in ax.get_xticklabels():
        label.set_fontsize(18)
    for label in ax.get_yticklabels():
        label.set_fontsize(18)
    for label in ax2.get_yticklabels():
        label.set_fontsize(18)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    fig.savefig('output/chromatic_biases.png', dpi=220)

if __name__ == '__main__':
    refraction_vs_zenith()
    chromatic_biases()
