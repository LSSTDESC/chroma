import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import _mypath
import chroma

def moffat1d(fwhm, beta, center=0.0):
    alpha = fwhm / (2.0 * np.sqrt(2.0**(1./beta) - 1.0))
    def f(x):
        u = ((x - center) / alpha)**2
        p = 1.0 / ((u + 1.0)**beta)
        return p / p.max()
    return f

cwave = np.linspace(500, 860, 256)
fig, ax = plt.subplots(2, 2, figsize=(8,5))

spectra = ['../../data/SEDs/'+s for s in ('ukg5v.ascii', 'CWW_Sbc_ext.ascii')]
redshifts = [0.0, 0.7]
filters = ['../../data/filters/'+s for s in ('LSST_r.dat', 'LSST_i.dat')]

ax[1,0].set_xlabel('Wavelength (nm)', fontsize=12)
ax[1,1].set_xlabel('Refraction (arcsec)', fontsize=12)
for i, s in enumerate(spectra):
    SED = chroma.SampledSED(s)
    SED = SED.createRedshifted(redshifts[i])
    wave = np.arange(500.0, 901.0, 1.0)
    photons = SED(wave)
    scale = 1.2 * photons[(wave > 500) & (wave < 900)].max()
    ax[i,0].plot(wave, photons/scale, color='black')
    ax[i,0].set_xlim(500, 900)
    ax[i,0].set_ylim(0.0, 1.0)
    ax[i,0].set_ylabel('$d(\mathrm{N_{photons}})/d\lambda$', fontsize=12)
    ax[i,1].set_ylim(0.0, 1.0)
    ax[i,1].set_ylabel('$d(\mathrm{N_{photons}})/d\mathrm{R}$', fontsize=12)
    xs = np.linspace(26.2, 27.2, 100)
    moffat = moffat1d(fwhm=0.6, beta=2.6, center=26.7)
    ax[i,1].plot(xs, moffat(xs)/1.2, color='black')

    for f in filters:
        filter_ = chroma.SampledBandpass(f).createTruncated(blue_limit=500, red_limit=1000)
        photons_filtered = photons * filter_(wave)
        color = np.interp(wave, cwave, np.linspace(1.0, 0.0, 256))

        R = chroma.get_refraction(wave, 35 * np.pi/180) * 3600 * 180/np.pi
        dR = np.diff(R)
        dwave = np.diff(wave)
        dwave_dR = dwave / dR # Jacobian
        dwave_dR = np.append(dwave_dR, dwave_dR[-1]) # fudge the last array element
        angle_dens = photons_filtered * np.abs(dwave_dR)


        # first axis is normal spectrum
        ax[i,0].plot(wave, 1.3 * photons_filtered/scale, color='black')
        ax[i,0].xaxis.set_ticks([500, 600, 700, 800, 900])
        chroma.chroma_fill_plot(wave, 1.3 * photons_filtered/scale, color, axes=ax[i,0])

        # second axis is photons binned by refraction
        ax[i,1].plot(R, angle_dens/angle_dens.max() / 1.2, color='black')
        chroma.chroma_fill_plot(R, angle_dens/angle_dens.max() / 1.2, color, axes=ax[i,1])

        for label in ax[i,1].get_xticklabels():
            label.set_fontsize(10)
        for label in ax[i,1].get_yticklabels():
            label.set_fontsize(10)
        for label in ax[i,0].get_xticklabels():
            label.set_fontsize(10)
        for label in ax[i,0].get_yticklabels():
            label.set_fontsize(10)

fig.subplots_adjust(hspace=0.23, wspace=0.32, bottom=0.12, right=0.92, top=0.92)
plt.savefig('output/photon_landings.png', dpi=220)
