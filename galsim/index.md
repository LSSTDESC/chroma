---
layout: text
title: GalSim
permalink: /GalSim/home/
---

As part of this analysis, we have added the capability to simulate wavelength-dependent surface brightness profiles to the open source galaxy simulation package [GalSim](https://github.com/GalSim-developers/GalSim), which is being used in the [GREAT3](http://www.great3challenge.info) shape measurement data challenge.  The new chromatic API makes it convenient to simulate both the effects of chromatic PSFs and chromatic galaxies (also referred to as galaxies with color gradients).

As an example, a simulation of a bulge+disk galaxy convolved with an atmospheric PSF and observed in the _r_-band might look like:

{% highlight python %}
    import galsim

    # load bulge, disk spectra into galsim.SED objects
    bulge_spectrum = galsim.SED('early_type_spec.dat')
    disk_spectrum = galsim.SED('late_type_spec.dat')
    # define the bulge and the disk as separable products of a spatial profile
    # and an SED.
    bulge = galsim.deVaucouleurs(flux=0.1, half_light_radius=0.4) * bulge_spectrum
    disk = galsim.Exponential(flux=0.2, half_light_radius=0.7) * disk_spectrum
    # Add them together!
    galaxy = bulge + disk

    # Make a chromatic PSF by transforming a fiducial monochromatic PSF at 500 nm
    psf_500 = galsim.Kolmogorov(fwhm=0.67) # fiducial PSF
    # galsim.ChromaticAtmosphere adds differential chromatic refraction and
    # chromatic seeing to the fiducial PSF.
    psf = galsim.ChromaticAtmosphere(psf_500, 500, zenith_angle=30*galsim.degrees)

    # Convolve everything together, don't forget to convolve in a pixel!
    final = galsim.Convolve(galaxy, psf, galsim.Pixel(0.2))

    # Need a filter bandpass to draw through.
    bandpass = galsim.Bandpass('rband.dat')
    # Draw the image!
    image = final.draw(bandpass)
{% endhighlight %}

For Euclid, the PSF is chromatic due to the diffraction limit.  If this were the only contribution to the PSF, then the chromatic effect would be \\({\scriptsize \mathrm{FWHM} \propto \lambda}\\).  When combined with telescope jitter and the modulation transfer function from the CCD, however, the Euclid PSF will actually scale approximately like \\({\scriptsize \mathrm{FWHM} \propto \lambda^{0.6}}\\).  This effect is also convenient to implement in GalSim:

{% highlight python %}
    # Again, we'll apply a wavelength dependent transformation to a
    # monochromatic fiducial PSF.
    aperture_diameter = 1.2 # meters
    central_wavelength = 750e-9 # meters
    lam_over_diam = central_wavelength / aperture_diam * galsim.radians / galsim.arcsec
    psf_750 = galsim.Airy(lam_over_diam=lam_over_diam)
    # galsim.ChromaticObject chromaticizes the fiducial PSF so that
    # .createDilated() can accept a function of wavelength as its argument.
    psf_Euclid = galsim.ChromaticObject(psf_750).createDilated(lambda w: (w/750)**0.6)
{% endhighlight %}
