import sys, os
import pickle

import numpy
import lmfit
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits

waves = {'u': numpy.arange(325, 401, 25),
         'g': numpy.arange(400, 551, 25),
         'r': numpy.arange(550, 701, 25),
         'i': numpy.arange(675, 826, 25),
         'z': numpy.arange(800, 951, 25),
         'Y': numpy.arange(900, 1100, 25)}
colors = {'u':'violet',
          'g':'blue',
          'r':'green',
          'i':'yellow',
          'z':'red',
          'Y':'black'}

nfiles = sum(map(len, waves.values())) * 2

values = pickle.load(open('seeing_vs_wave.pik'))

# fwhm plot
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('wavelength (nm)')
fwhm_ax.set_ylabel('FWHM (pixels)')
fwhm_ax.set_ylim(2.7, 4.5)
for k in waves.keys():
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 1)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm_x'], values[ind]['fwhm_x_err'],
                     ls='none', marker='o', color=colors[k])
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm_y'], values[ind]['fwhm_y_err'],
                     ls='none', marker='o', color=colors[k])
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 2)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm_x'], values[ind]['fwhm_x_err'],
                     ls='none', marker='o', color=colors[k], alpha=0.2)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm_y'], values[ind]['fwhm_y_err'],
                     ls='none', marker='o', color=colors[k], alpha=0.2)

x = numpy.linspace(300, 1100, 100)
w600 = numpy.where(numpy.logical_and(values['wave'] == 600, values['mode'] == 2))[0][0]
y600 = values[w600]['fwhm_x']
y= y600 * (x/600.)**(-0.2)
fwhm_ax.plot(x, y)

# beta plot
beta_fig = plt.figure()
beta_ax = plt.subplot(111)
for k in waves.keys():
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 1)
    beta_ax.errorbar(values[ind]['wave'], values[ind]['beta'], values[ind]['beta_err'],
                     ls='none', marker='o', color=colors[k])
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 2)
    beta_ax.errorbar(values[ind]['wave'], values[ind]['beta'], values[ind]['beta_err'],
                     ls='none', marker='o', color=colors[k], alpha=0.2)
beta_ax.set_xlabel('wavelength (nm)')
beta_ax.set_ylabel(r'$\beta$')

if not os.path.isdir('plots'):
    os.mkdir('plots')

fwhm_fig.savefig('plots/FWHM.pdf')
beta_fig.savefig('plots/beta.pdf')
