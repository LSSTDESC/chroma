# Make some photon histograms against wavelength and against refraction

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

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,5), dpi=80)
    ax1.set_title(title)
    w = numpy.logical_and(wave > 500, wave < 850)
    scale = 1.2 * photons[w].max()
    ax1.plot(wave, photons/scale, color='black')
    ax1.set_xlim(500, 850)
    ax1.set_ylim(0.0, 1.0)
    ax2.set_xlim(21.6, 22.6)
    ax2.set_ylim(0.0, 1.0)

    filters = ['LSST_r', 'LSST_i']
    for f in filters:
        filter_file = '../../data/filters/{}.dat'.format(f)
        fdata = numpy.genfromtxt(filter_file)
        fwave, throughput = fdata[:,0], fdata[:,1]
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

    FWHM = 0.6
    beta = 2.619 # measured from phoSim?
    moffat = moffat1d(FWHM, beta, 22.1)
    xs = numpy.linspace(21.6, 22.6, 100)
    ax2.plot(xs, moffat(xs) / 1.2, color='black')
    fig.savefig(outfile)

def main():
    stars = ['uko5v', 'ukf5v', 'ukm5v']
    starnames = ['O5v', 'F5v', 'M5v']
    for star, starname in zip(stars, starnames):
        star_file = '../../data/SEDs/{}.ascii'.format(star)
        sdata = numpy.genfromtxt(star_file)
        swave, sflux = sdata[:,0], sdata[:,1]
        photons = sflux * swave
        photon_hists(swave, photons,
                     '{} star, zenith = 30 degrees'.format(starname),
                     'output/DCR_PSF/DCR_PSF.{}.png'.format(starname))

    gals = ['CWW_E', 'CWW_Im', 'CWW_Sbc']
    galnames = ['E', 'Im', 'Sbc']
    for gal, galname in zip(gals, galnames):
        gal_file = '../../data/SEDs/{}_ext.ascii'.format(gal)
        gdata = numpy.genfromtxt(gal_file)
        gwave, gflux = gdata[:,0], gdata[:,1]
        photons = gflux * gwave
        gwave *= 2.0 # z = 1.0
        photon_hists(gwave, photons,
                     '{} galaxy, z = 1.0, zenith = 30 degrees'.format(galname),
                     'output/DCR_PSF/DCR_PSF.{}.png'.format(galname))


if __name__ == '__main__':
    main()
