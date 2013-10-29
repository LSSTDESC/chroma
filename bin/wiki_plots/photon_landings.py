import os
import subprocess

import numpy
import matplotlib.pyplot as plt

import _mypath
import chroma

def moffat1d(fwhm, beta, center=0.0):
    alpha = fwhm / (2.0 * numpy.sqrt(2.0**(1./beta) - 1.0))
    def f(x):
        u = ((x - center) / alpha)**2
        p = 1.0 / ((u + 1.0)**beta)
        return p / p.max()
    return f

cwave = numpy.linspace(500, 860, 256)
fig, ax = plt.subplots(2, 2, figsize=(9,5), dpi=60)

spectra = ['../../data/SEDs/'+s for s in ('ukg5v.ascii', 'KIN_Sa_ext.ascii')]
redshifts = [0.0, 1.3]
filters = ['../../data/filters/'+s for s in ('LSST_r.dat', 'LSST_i.dat')]

ax[1,0].set_xlabel('Wavelength (nm)')
ax[1,1].set_xlabel('Refraction (arcsec)')
for i, s in enumerate(spectra):
    wave, flux = numpy.genfromtxt(s).T
    wave *= (1 + redshifts[i])
    photons = flux * wave
    scale = 1.2 * photons[(wave > 500) & (wave < 850)].max()
    ax[i,0].plot(wave, photons/scale, color='black')
    ax[i,0].set_xlim(500, 850)
    ax[i,0].set_ylim(0.0, 1.0)
    ax[i,0].set_ylabel('$d(\mathrm{N_{photons}})/d\lambda$')
    ax[i,1].set_ylim(0.0, 1.0)
    ax[i,1].set_ylabel('$d(\mathrm{N_{photons}})/d\mathrm{R}$')
    xs = numpy.linspace(21.6, 22.6, 100)
    moffat = moffat1d(fwhm=0.6, beta=2.619, center=22.1)
    ax[i,1].plot(xs, moffat(xs)/1.2, color='black')

    for f in filters:
        fdata = numpy.genfromtxt(f)
        fwave, throughput = fdata[:,0], fdata[:,1]
        w = (fwave >= 500) & (fwave <= 1000)
        fwave = fwave[w]
        throughput = throughput[w]
        #plotwave = numpy.union1d(wave, fwave)
        plotwave = wave
        plotwave.sort()
        photons_filtered = numpy.interp(plotwave, wave, photons) \
          * numpy.interp(plotwave, fwave, throughput)
        color = numpy.interp(plotwave, cwave, numpy.linspace(1.0, 0.0, 256))

        # first axis is normal spectrum
        w = photons_filtered > 0.001 * photons_filtered.max()
        ax[i,0].plot(plotwave[w], 1.3 * photons_filtered[w]/scale, color='black')
        chroma.chroma_fill_plot(plotwave[w], 1.3 * photons_filtered[w]/scale, color[w], axes=ax[i,0])

        # second axis is photons binned by refraction
        R, angle_dens = chroma.wave_dens_to_angle_dens(plotwave, photons_filtered,
                                                       zenith=30.0 * numpy.pi/180)
        R *= 206265
        #import ipdb; ipdb.set_trace()
#        w = angle_dens > 0.001 * angle_dens
        ax[i,1].plot(R[w], angle_dens[w]/angle_dens[w].max() / 1.2, color='black')
        chroma.chroma_fill_plot(R[w], angle_dens[w]/angle_dens[w].max() / 1.2, color[w], axes=ax[i,1])

plt.savefig('output/photon_landings.pdf')
