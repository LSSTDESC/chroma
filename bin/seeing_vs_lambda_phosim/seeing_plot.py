import os
import pickle

import numpy
import matplotlib.pyplot as plt

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

values = pickle.load(open('seeing_vs_wave.pik'))

if not os.path.isdir('plots'):
    os.mkdir('plots')


# fwhm plot with all filters and both with and without instrument
print 'Plotting FWHM for all filters both with and without instrument'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('wavelength (nm)')
fwhm_ax.set_ylabel('FWHM (pixels)')
fwhm_ax.set_ylim(2.7, 4.5)
for k in waves.keys():
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 1)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k])
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 2)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k], alpha=0.2)

x = numpy.linspace(300, 1100, 100)
w600 = numpy.where(numpy.logical_and(values['wave'] == 600, values['mode'] == 2))[0][0]
y600 = values[w600]['fwhm']
y= y600 * (x/600.)**(-0.2)
fwhm_ax.plot(x, y)
fwhm_fig.savefig('plots/FWHM.pdf')


# beta plot with all filters and both with and without instrument
print 'Plotting beta for all filters both with and without instrument'
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
beta_fig.savefig('plots/beta.pdf')

# changing colors for r+i only plots:
colors['r'] = 'blue'
colors['i'] = 'red'

# fwhm plot with only r+i bands and atmosphere only
print 'Plotting FWHM for r+i for atmosphere only'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('wavelength (nm)')
fwhm_ax.set_ylabel('FWHM (pixels)')
fwhm_ax.set_ylim(2.8, 3.6)
fwhm_ax.set_xlim(500, 900)
for k in ['r', 'i']:
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 2)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k])

x = numpy.linspace(300, 1100, 100)
w700 = numpy.where(numpy.logical_and(values['wave'] == 700, values['mode'] == 2))[0][0]
y700 = values[w700]['fwhm']
y= y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
# also plot filters in background
rwave, rthru = numpy.genfromtxt('../../data/filters/LSST_r.dat').T
rthru /= rthru.max()
rthru *= 0.3
rthru += 2.8
fwhm_ax.fill_between(rwave, rthru, color='blue', alpha=0.1)
iwave, ithru = numpy.genfromtxt('../../data/filters/LSST_i.dat').T
ithru /= ithru.max()
ithru *= 0.3
ithru += 2.8
fwhm_ax.fill_between(iwave, ithru, color='red', alpha=0.1)
# save
fwhm_fig.savefig('plots/FWHM.r+i.atmos.pdf')


# fwhm plot with only r+i bands and atmosphere only
print 'Plotting FWHM for r+i for atmosphere and instrument'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('wavelength (nm)')
fwhm_ax.set_ylabel('FWHM (pixels)')
fwhm_ax.set_ylim(2.8, 3.6)
fwhm_ax.set_xlim(500, 900)
for k in ['r', 'i']:
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 1)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k], alpha=0.2)
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 2)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k])

x = numpy.linspace(300, 1100, 100)
w700 = numpy.where(numpy.logical_and(values['wave'] == 700, values['mode'] == 2))[0][0]
y700 = values[w700]['fwhm']
y= y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
# also plot filters in background
rwave, rthru = numpy.genfromtxt('../../data/filters/LSST_r.dat').T
rthru /= rthru.max()
rthru *= 0.3
rthru += 2.8
fwhm_ax.fill_between(rwave, rthru, color='blue', alpha=0.1)
iwave, ithru = numpy.genfromtxt('../../data/filters/LSST_i.dat').T
ithru /= ithru.max()
ithru *= 0.3
ithru += 2.8
fwhm_ax.fill_between(iwave, ithru, color='red', alpha=0.1)
# save
fwhm_fig.savefig('plots/FWHM.r+i.atmos+inst.pdf')
