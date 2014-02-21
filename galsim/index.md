---
layout: text
title: GalSim
permalink: /GalSim/home/
---

As part of this feature project, we have added the capability to simulate wavelength-dependent surface brightness profiles to the open source galaxy simulation package [GalSim](https://github.com/GalSim-developers/GalSim), which is being used to in the [GREAT3](http://www.great3challenge.info) shape measurement data challenge.  The new chromatic API make it convenient to simulate both the effects of chromatic PSFs and galaxies with color gradients.

As an example, a simulation of a bulge+disk galaxy convolved with an atmospheric PSF and observed in the r-band might look like:

    import galsim

    bulge_spectrum = galsim.SED('early_type_spec.dat')
    disk_spectrum = galsim.SED('late_type_spec.dat')
    bulge = galsim.deVaucouleurs(flux=0.1, half_light_radius=0.4) * bulge_spectrum
    disk = galsim.Exponential(flux=0.2, half_light_radius=0.7) * disk_spectrum
    galaxy = bulge + disk

    psf_500 = galsim.Kolmogorov(fwhm=0.67)
    psf = galsim.ChromaticAtmosphere(psf_500, 500, zenith_angle=30*galsim.degrees)

    pix = galsim.Pixel(0.2)

    final = galsim.Convolve(galaxy, psf, pix)

    bandpass = galsim.Bandpass('rband.dat')
    image = final.draw(bandpass)

A Euclid-like PSF that scales approximately like \\({\scriptsize \mathrm{FWHM} \propto \lambda^{0.6}}\\) is also convenient to implement:

    psf_750 = galsim.Gaussian(fwhm=0.2)
    psf_Euclid = galsim.ChromaticObject(psf_750).createDilated(lambda w: (w/750)**0.6)
