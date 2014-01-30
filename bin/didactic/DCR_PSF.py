# Make some photon histograms against wavelength and against refraction

import os
import subprocess

import numpy
import matplotlib.pyplot as plt

import _mypath #adds chroma to PYTHONPATH
import chroma

def moffat1d(fwhm, beta, center=0.0):
    alpha = fwhm / (2.0 * numpy.sqrt(2.0**(1./beta) - 1.0))
    def f(x):
        u = ((x - center) / alpha)**2
        p = 1.0 / ((u + 1.0)**beta)
        return p / p.max()
    return f

def photon_hists(wave, photons, title, outfile):
    cwave = numpy.linspace(500, 860, 256)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,6.5), dpi=80)
    ax1.set_title(title)
    w = numpy.logical_and(wave > 500, wave < 850)
    scale = 1.2 * photons[w].max()
    ax1.plot(wave, photons/scale, color='black')
    ax1.set_xlim(500, 850)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('photons/s/cm$^2$/$\AA$')
    ax2.set_xlim(21.6, 22.6)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel('Refraction (arcsec)')
    ax2.set_ylabel('photons/s/cm$^2$/arcsec')


    filters = ['LSST_r', 'LSST_i']
    for f in filters:
        filter_file = '../../data/filters/{}.dat'.format(f)
        fwave, throughput = numpy.genfromtxt(filter_file).T
        w = numpy.logical_and(fwave >= 500, fwave <= 1000)
        fwave = fwave[w]
        throughput = throughput[w]
        plotwave = numpy.union1d(wave, fwave)
        plotwave.sort()
        photons_filtered = numpy.interp(plotwave, wave, photons) \
          * numpy.interp(plotwave, fwave, throughput)
        color = numpy.interp(plotwave, cwave, numpy.linspace(1.0, 0.0, 256))

        # first axis is normal spectrum
        w = photons_filtered > 0.001 * photons_filtered.max()
        ax1.plot(plotwave[w], 1.3 * photons_filtered[w]/scale, color='black')
        chroma.chroma_fill_plot(plotwave[w], 1.3 * photons_filtered[w]/scale, color[w], axes=ax1)

        # second axis is photons binned by refraction
        R, angle_dens = chroma.wave_dens_to_angle_dens(plotwave, photons_filtered,
                                                       zenith=30.0 * numpy.pi/180)
        R *= 206265
        w = angle_dens > 0.001 * angle_dens
        ax2.plot(R[w], angle_dens[w]/angle_dens[w].max() / 1.2, color='black')
        chroma.chroma_fill_plot(R[w], angle_dens[w]/angle_dens[w].max() / 1.2, color[w], axes=ax2)

    FWHM = 0.6 # arcseconds
    beta = 2.619 # measured from phoSim?
    moffat = moffat1d(FWHM, beta, 22.1)
    xs = numpy.linspace(21.6, 22.6, 100)
    ax2.plot(xs, moffat(xs) / 1.2, color='black')
    fig.savefig(outfile)

def keynote_figs():
    data_dir = '../../data/SEDs/'
    G5v_file = data_dir + 'ukg5v.ascii'
    swave, sflux = numpy.genfromtxt(G5v_file).T
    photons = sflux * swave
    photon_hists(swave, photons, 'G5v star, zenith angle = 30 degrees', 'output/d.G5v.png')
    Ellip_file = data_dir + 'CWW_E_ext.ascii'
    gwave, gflux = numpy.genfromtxt(Ellip_file).T
    gwave *= 2.3
    photons = gflux * gwave
    photon_hists(gwave, photons, 'E galaxy, z = 1.3, zenith angle = 30 degrees', 'output/d.E.png')
    Sa_file = data_dir + 'KIN_Sa_ext.ascii'
    gwave, gflux = numpy.genfromtxt(Sa_file).T
    gwave *= 2.3
    photons = gflux * gwave
    photon_hists(gwave, photons, 'Sa galaxy, z = 1.3, zenith angle = 30 degrees', 'output/d.Sa.png')

def main():
    stars = ['uko5v', 'ukf5v', 'ukm5v']
    starnames = ['O5v', 'F5v', 'M5v']
    for star, starname in zip(stars, starnames):
        star_file = '../../data/SEDs/{}.ascii'.format(star)
        swave, sflux = numpy.genfromtxt(star_file).T
        photons = sflux * swave
        photon_hists(swave, photons,
                     '{} star, zenith angle = 30 degrees'.format(starname),
                     'output/d.{}.png'.format(starname))

    gals = ['CWW_E', 'CWW_Sbc', 'CWW_Im']
    galnames = ['E', 'Sbc', 'Im']
    for gal, galname in zip(gals, galnames):
        gal_file = '../../data/SEDs/{}_ext.ascii'.format(gal)
        gwave, gflux = numpy.genfromtxt(gal_file).T
        photons = gflux * gwave
        gwave *= 2.0 # z = 1.0
        photon_hists(gwave, photons,
                     '{} galaxy, z = 1.0, zenith angle = 30 degrees'.format(galname),
                     'output/d.{}.png'.format(galname))

    cmd = 'convert -delay 200 -loop 0 '
    for name in starnames+galnames:
        cmd += 'output/d.{}.png '.format(name)
    cmd += 'output/DCR_PSF_anim.gif'
    subprocess.call(cmd, shell=True)

    for name in starnames+galnames:
        os.remove('output/d.{}.png'.format(name))

if __name__ == '__main__':
    main()