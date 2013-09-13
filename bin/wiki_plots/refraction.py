# Make some plots of atmospheric refraction vs. wavelength for different zenith angles.
# Overplot the LSST filter bandpasses for reference.

# Zeroth plot: refraction vs zenith.  Use different colors for different wavelengths

import numpy
import matplotlib.pyplot as plt

import _mypath #adds chroma to PYTHONPATH
import chroma

zeniths = numpy.linspace(0, 70, 71)
waves = numpy.arange(300, 1101, 200, dtype=numpy.float64)

#open figure for output
fig = plt.figure(figsize=(10,5), dpi=80)
ax = fig.add_subplot(111)
ax.set_title('Atmospheric Refraction')
ax.set_xlabel('zenith angle (degrees)')
ax.set_xlim(0, 70)
ax.set_ylabel('Refraction (arcsec)')
ax.set_ylim(0, 100)

for wave in waves:
    refrac_angles = numpy.array([chroma.atm_refrac(wave, z * numpy.pi/180) for z in zeniths])
    ax.plot(zeniths, refrac_angles * 206265, label=str(wave)+' nm')
ax.legend(fontsize='small', title='wavelength', loc='upper left')
fig.savefig('output/refraction_vs_zenith.png')

# First plot: refraction vs. wavelength for zenith angles between 0 deg and 50 deg

zeniths = [10.0, 20.0, 30.0, 40.0, 50.0] # degrees
waves = numpy.arange(300.0, 1100.0, 1.0) # nanometers

# open figure for output
fig = plt.figure(figsize=(10,5), dpi=80)
ax = fig.add_subplot(111)
ax.set_title('Refraction angle')
ax.set_xlabel('Wavelength (nm)')
ax.set_xlim(300, 1100)
ax.set_ylabel('Refraction (arcsec)')
ax.set_ylim(0, 50)

for zenith in zeniths:
    # chroma expects angles in radians, so need to convert deg to radians
    refrac_angle = chroma.atm_refrac(waves, zenith * numpy.pi/180)
    # chroma output is also in radians, so convert to arcsec here
    ax.plot(waves, refrac_angle * 206265, label=str(zenith)+' deg')

# load LSST filters
filter_dir = '../../data/filters/'
filter_files = [filter_dir+'LSST_{}.dat'.format(band) for band in 'ugrizy']
colors = ['Purple', 'Blue', 'Green', 'Orange', 'Magenta', 'Red']

for i, filter_file in enumerate(filter_files):
    # filters are stored in two columns: wavelength (nm), and throughput
    fdata = numpy.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]
    filter_name = filter_file.split('/')[-1].replace('.dat', '') # get filter name from file name
    ax.fill_between(fwave, throughput * 80, 0, color=colors[i], alpha=0.3)
ax.legend(fontsize='small', title='zenith angle')
fig.savefig('output/absolute_refraction_vs_wavelength.png')

# Second plot: refraction relative to 500 nanometers vs. wavelength for same zenith angles

fig = plt.figure(figsize=(10,5), dpi=80)
ax = fig.add_subplot(111)
ax.set_title('Relative refraction angle')
ax.set_xlabel('Wavelength (nm)')
ax.set_xlim(300, 1100)
ax.set_ylabel('Relative refraction (arcsec)')
ax.set_ylim(-1, 1)

for zenith in zeniths:
    # chroma expects angles in radians, so need to convert deg to radians
    refrac_angle = chroma.atm_refrac(waves, zenith * numpy.pi/180)
    refrac_ref = refrac_angle[numpy.argmin(abs(waves - 500))]
    # chroma output is also in radians, so convert to arcsec here
    ax.plot(waves, (refrac_angle - refrac_ref) * 206265, label=str(zenith)+' deg')

for i, filter_file in enumerate(filter_files):
    # filters are stored in two columns: wavelength (nm), and throughput
    fdata = numpy.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]
    filter_name = filter_file.split('/')[-1].replace('.dat', '') # get filter name from file name
    ax.fill_between(fwave, throughput * 3.2 - 1, -1, color=colors[i], alpha=0.3)
ax.legend(fontsize='small', title='zenith angle')
fig.savefig('output/relative_refraction_vs_wavelength.png')
fig.savefig('output/DCRfilters.pdf')

# Third plot: refraction relative to 690 nanometers vs wavelength but only for weak lensing shape
# measurement filters (r & i).

fig = plt.figure(figsize=(10,5), dpi=80)
ax = fig.add_subplot(111)
ax.set_title('Relative refraction angle, shape measurement filters')
ax.set_xlabel('Wavelength (nm)')
ax.set_xlim(500, 900)
ax.set_ylabel('Relative refraction (arcsec)')
ax.set_ylim(-0.5, 0.5)

for zenith in zeniths:
    # chroma expects angles in radians, so need to convert deg to radians
    refrac_angle = chroma.atm_refrac(waves, zenith * numpy.pi/180)
    refrac_ref = refrac_angle[numpy.argmin(abs(waves - 690))]
    # chroma output is also in radians, so convert to arcsec here
    ax.plot(waves, (refrac_angle - refrac_ref) * 206265, label=str(zenith)+' deg')

for i, filter_file in enumerate(filter_files[2:4]):
    # filters are stored in two columns: wavelength (nm), and throughput
    fdata = numpy.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]
    filter_name = filter_file.split('/')[-1].replace('.dat', '') # get filter name from file name
    ax.fill_between(fwave, throughput * 1.6 - 0.5, -0.5, color=colors[2:4][i], alpha=0.3)
ax.legend(fontsize='small', title='zenith angle')
fig.savefig('output/relative_refraction_vs_wavelength_r_i.png')
