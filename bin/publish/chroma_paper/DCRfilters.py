# Make some plots of atmospheric refraction vs. wavelength for different zenith angles.
# Overplot the LSST filter bandpasses for reference.

# Zeroth plot: refraction vs zenith.  Use different colors for different wavelengths

import os

import numpy
import matplotlib.pyplot as plt

import _mypath #adds chroma to PYTHONPATH
import chroma

# # load LSST filters
filter_dir = '../../../data/filters/'
filter_files = [filter_dir+'LSST_{}.dat'.format(band) for band in 'ugrizy']
colors = ['Purple', 'Blue', 'Green', 'Orange', 'Magenta', 'Red']

zeniths = [10.0, 20.0, 30.0, 40.0, 50.0] # degrees
waves = numpy.arange(300.0, 1200.0, 1.0) # nanometers

fig = plt.figure(figsize=(8,5))
#ax = fig.add_subplot(111)
ax = fig.add_axes([0.14, 0.13, 0.76, 0.78])
ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_xlim(300, 1200)
ax.set_ylabel('Relative refraction (arcsec)', fontsize=12)
ax.set_ylim(-1, 1.)

for zenith in zeniths:
    # chroma expects angles in radians, so need to convert deg to radians
    refrac_angle = chroma.atm_refrac(waves, zenith * numpy.pi/180)
    refrac_ref = refrac_angle[numpy.argmin(abs(waves - 500))]
    # chroma output is also in radians, so convert to arcsec here
    ax.plot(waves, (refrac_angle - refrac_ref) * 206265, label=str(zenith)+' deg')

# 350nm wide Euclid filter.
#ax.fill_between([0., 550., 550., 900., 900., 1200.], [-1, -1, 0.25, 0.25, -1, -1], -1,
#                color='black', alpha=0.15)

for i, filter_file in enumerate(filter_files):
    # filters are stored in two columns: wavelength (nm), and throughput
    fdata = numpy.genfromtxt(filter_file)
    fwave, throughput = fdata[:,0], fdata[:,1]
    filter_name = filter_file.split('/')[-1].replace('.dat', '') # get filter name from file name
    ax.fill_between(fwave, throughput * 2.5 - 1, -1, color=colors[i], alpha=0.3)
# Add in lambda^(-2/5) for chromatic seeing comparison integrand comparison
ax2 = ax.twinx()
ys = (waves/500.0)**(-2./5)
ax2.plot(waves, ys, 'k', lw=2, label='$\lambda^{-2/5}$')
ax.legend(fontsize=10, title='zenith angle')
ax2.legend(fontsize=10, title='chromatic seeing', loc='upper right', bbox_to_anchor = (0.78, 1))
ax2.set_xlim(300, 1100)
ax2.set_ylabel('Relative $r^2_\mathrm{PSF}$', fontsize=12)

for label in ax.get_xticklabels():
    label.set_fontsize(12)
for label in ax.get_yticklabels():
    label.set_fontsize(12)
for label in ax2.get_yticklabels():
    label.set_fontsize(12)

ax.text(350, -0.9, 'u')
ax.text(465, -0.9, 'g')
ax.text(620, -0.9, 'r')
ax.text(750, -0.9, 'i')
ax.text(870, -0.9, 'z')
ax.text(960, -0.9, 'y')

if not os.path.isdir('output/'):
    os.mkdir('output/')
fig.savefig('output/DCRfilters.png', dpi=220)
