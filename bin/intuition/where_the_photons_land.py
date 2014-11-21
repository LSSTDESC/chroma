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

#spectra = ['../../data/SEDs/'+s for s in ('ukg5v.ascii', 'CWW_Sbc_ext.ascii')]
spectra = ['../../data/SEDs/'+s for s in ('ukg5v.ascii', 'KIN_Sa_ext.ascii')]
specnames = ['G5V star', 'Sa galaxy\nat z=0.6']
ytextoffset = [0, -0.1]
redshifts = [0.0, 0.6]
filters = ['../../data/filters/'+s for s in ('LSST_r.dat', 'LSST_i.dat')]

psf_fwhm = 0.7
psf_beta = 3.0
zenith_angle = 35 * np.pi/180

ax[1,0].set_xlabel('Wavelength (nm)', fontsize=12)
ax[1,1].set_xlabel('Refraction (arcsec)', fontsize=12)
for i, s in enumerate(spectra):
    SED = chroma.SED(s)
    SED = SED.atRedshift(redshifts[i])
    wave = np.arange(500.0, 901.0, 1.0)
    photons = SED(wave)
    scale = 1.2 * photons[(wave > 500) & (wave < 900)].max()
    ax[i,0].plot(wave, photons/scale, color='black')
    ax[i,0].set_xlim(500, 900)
    ax[i,0].set_ylim(0.0, 1.0)
    ax[i,0].set_ylabel('$d\mathrm{N_{photons}}/d\lambda$', fontsize=12)
    ax[i,0].text(525, 0.85+ytextoffset[i], specnames[i], fontsize=11)
    ax[i,1].set_ylim(0.0, 1.0)
    ax[i,1].set_ylabel('$d\mathrm{N_{photons}}/d\mathrm{R}$', fontsize=12)

    ax[i,1].text(26.30, 0.85+ytextoffset[i], specnames[i], fontsize=11)
    xs = np.linspace(26.25, 27.25, 100)
    moffat = moffat1d(fwhm=psf_fwhm, beta=psf_beta, center=26.7)
    ax[i,1].plot(xs, moffat(xs)/1.2, color='black')
    ax[i,1].set_xlim(26.25, 27.25)

    for f in filters:
        filter_ = chroma.Bandpass(f).truncate(blue_limit=500, red_limit=1000)
        photons_filtered = photons * filter_(wave)
        color = np.interp(wave, cwave, np.linspace(1.0, 0.0, 256))

        R = chroma.get_refraction(wave, zenith_angle) * 3600 * 180/np.pi
        dR = np.diff(R)
        dwave = np.diff(wave)
        dwave_dR = dwave / dR # Jacobian
        dwave_dR = np.append(dwave_dR, dwave_dR[-1]) # fudge the last array element
        angle_dens = photons_filtered * np.abs(dwave_dR)

        # first axis is normal spectrum
        ax[i,0].plot(wave, 1.3 * photons_filtered/scale, color='black')
        ax[i,0].xaxis.set_ticks([500, 600, 700, 800, 900])
        chroma.chroma_fill_between(wave, 1.3 * photons_filtered/scale, 0, c=color, axes=ax[i,0])

        # second axis is photons binned by refraction
        ax[i,1].plot(R, angle_dens/angle_dens.max() / 1.2, color='black')
        chroma.chroma_fill_between(R, angle_dens/angle_dens.max() / 1.2, 0, c=color, axes=ax[i,1])
        if i==0:
            ax[1,1].plot(R, angle_dens/angle_dens.max() / 1.2, color='black', alpha=0.2)
            ax[1,1].text(26.95, 0.14, specnames[0], fontsize=11, alpha=0.2)

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
plt.savefig('output/photon_landings.pdf')
