import os
import pickle

import numpy
import matplotlib.pyplot as plt
from matplotlib import ticker

def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = numpy.floor(numpy.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

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


# fwhm plot with all filters and both with and without instrument
print 'Plotting FWHM for all filters both with and without instrument'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('wavelength (nm)')
fwhm_ax.set_ylabel('FWHM (pixels)')
fwhm_ax.set_ylim(2.7, 4.5)
# for k, ws in waves.iteritems():
#     for w in ws:
#         print k, w
#         ind = numpy.logical_and(numpy.logical_and(values['filter'] == k,
#                                                   values['mode'] == 1),
#                                 values['wave'] == w)

#         fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
#                          ls='none', marker='o', color=colors[k])
#         ind = numpy.logical_and(numpy.logical_and(values['filter'] == k,
#                                                   values['mode'] == 2),
#                                 values['wave'] == w)
#         fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
#                          ls='none', marker='o', color=colors[k], alpha=0.3)
for k in waves.keys():
    ind = (values['filter'] == k) & (values['mode'] == 1) & (values['flux'] > 1e4)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k])
    ind = (values['filter'] == k) & (values['mode'] == 2) & (values['flux'] > 1e4)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k], alpha=0.3)

x = numpy.linspace(300, 1100, 100)
w600 = numpy.where(numpy.logical_and(values['wave'] == 600, values['mode'] == 2))[0][0]
y600 = values[w600]['fwhm']
y = y600 * (x/600.)**(-0.2)
fwhm_ax.plot(x, y)
fwhm_fig.savefig('plots/FWHM.pdf')


# log-log fwhm plot with all filters and both with and without instrument
print 'Plotting log-log FWHM for all filters both with and without instrument'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('log (wavelength (nm))')
fwhm_ax.set_ylabel('log (FWHM (pixels))')
fwhm_ax.set_xlim(300, 1100)
fwhm_ax.set_ylim(2.0, 5.0)
fwhm_ax.set_xscale('log')
fwhm_ax.set_yscale('log')
subs = numpy.arange(1.0, 9.01, 1.0)
fwhm_ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
fwhm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
fwhm_ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
fwhm_ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs))
fwhm_ax.yaxis.set_major_formatter(ticker.NullFormatter())
fwhm_ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))

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
y = y600 * (x/600.)**(-0.2)
fwhm_ax.plot(x, y)
fwhm_fig.savefig('plots/ll_FWHM.pdf')


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
y = y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
# also plot filters in background
rwave, rthru = numpy.genfromtxt('../../../../data/filters/LSST_r.dat').T
rthru /= rthru.max()
rthru *= 0.3
rthru += 2.8
fwhm_ax.fill_between(rwave, rthru, color='blue', alpha=0.1)
iwave, ithru = numpy.genfromtxt('../../../../data/filters/LSST_i.dat').T
ithru /= ithru.max()
ithru *= 0.3
ithru += 2.8
fwhm_ax.fill_between(iwave, ithru, color='red', alpha=0.1)
# save
fwhm_fig.savefig('plots/FWHM.r+i.atmos.pdf')


# log-log fwhm plot with only r+i bands and atmosphere only
print 'Plotting log-log FWHM for r+i for atmosphere only'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('log (wavelength (nm))')
fwhm_ax.set_ylabel('log (FWHM (pixels))')
fwhm_ax.set_ylim(2.8, 3.6)
fwhm_ax.set_xlim(500, 900)
fwhm_ax.set_xscale('log')
fwhm_ax.set_yscale('log')
subs = numpy.arange(1.0, 9.01, 1.0)
fwhm_ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
fwhm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
fwhm_ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
fwhm_ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs))
fwhm_ax.yaxis.set_major_formatter(ticker.NullFormatter())
fwhm_ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))

for k in ['r', 'i']:
    ind = numpy.logical_and(values['filter'] == k, values['mode'] == 2)
    fwhm_ax.errorbar(values[ind]['wave'], values[ind]['fwhm'], values[ind]['fwhm_err'],
                     ls='none', marker='o', color=colors[k])

x = numpy.linspace(300, 1100, 100)
w700 = numpy.where(numpy.logical_and(values['wave'] == 700, values['mode'] == 2))[0][0]
y700 = values[w700]['fwhm']
y = y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
# also plot filters in background
rwave, rthru = numpy.genfromtxt('../../../../data/filters/LSST_r.dat').T
rthru /= rthru.max()
rthru *= 0.3
rthru += 2.8
fwhm_ax.fill_between(rwave, rthru, color='blue', alpha=0.1)
iwave, ithru = numpy.genfromtxt('../../../../data/filters/LSST_i.dat').T
ithru /= ithru.max()
ithru *= 0.3
ithru += 2.8
fwhm_ax.fill_between(iwave, ithru, color='red', alpha=0.1)
# save
fwhm_fig.savefig('plots/ll_FWHM.r+i.atmos.pdf')


# fwhm plot with only r+i bands including instrument
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
y = y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
# also plot filters in background
rwave, rthru = numpy.genfromtxt('../../../../data/filters/LSST_r.dat').T
rthru /= rthru.max()
rthru *= 0.3
rthru += 2.8
fwhm_ax.fill_between(rwave, rthru, color='blue', alpha=0.1)
iwave, ithru = numpy.genfromtxt('../../../../data/filters/LSST_i.dat').T
ithru /= ithru.max()
ithru *= 0.3
ithru += 2.8
fwhm_ax.fill_between(iwave, ithru, color='red', alpha=0.1)
# save
fwhm_fig.savefig('plots/FWHM.r+i.atmos+inst.pdf')


# log-log fwhm plot with only r+i bands including instrument
print 'Plotting log-log FWHM for r+i for atmosphere and instrument'
fwhm_fig = plt.figure()
fwhm_ax = plt.subplot(111)
fwhm_ax.set_xlabel('log (wavelength (nm))')
fwhm_ax.set_ylabel('log (FWHM (pixels))')
fwhm_ax.set_ylim(2.8, 3.6)
fwhm_ax.set_xlim(500, 900)
fwhm_ax.set_xscale('log')
fwhm_ax.set_yscale('log')
subs = numpy.arange(1.0, 9.01, 1.0)
fwhm_ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
fwhm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
fwhm_ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
fwhm_ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs))
fwhm_ax.yaxis.set_major_formatter(ticker.NullFormatter())
fwhm_ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
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
y = y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
w700 = numpy.where(numpy.logical_and(values['wave'] == 700, values['mode'] == 1))[0][0]
y700 = values[w700]['fwhm']
y = y700 * (x/700.)**(-0.2)
fwhm_ax.plot(x, y, color='k')
# also plot filters in background
rwave, rthru = numpy.genfromtxt('../../../../data/filters/LSST_r.dat').T
rthru /= rthru.max()
rthru *= 0.3
rthru += 2.8
fwhm_ax.fill_between(rwave, rthru, color='blue', alpha=0.1)
iwave, ithru = numpy.genfromtxt('../../../../data/filters/LSST_i.dat').T
ithru /= ithru.max()
ithru *= 0.3
ithru += 2.8
fwhm_ax.fill_between(iwave, ithru, color='red', alpha=0.1)
# save
fwhm_fig.savefig('plots/ll_FWHM.r+i.atmos+inst.pdf')
