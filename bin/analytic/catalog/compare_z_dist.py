""" Plot the redshift distribution of galaxies from the CatSim output.  For comparison, overplot
the scaled distribution found in Chang+13.
"""


import cPickle

import numpy as np
import matplotlib.pyplot as plt

def hist_with_peak(x, bins=None, range=None, ax=None, orientation='vertical',
                   histtype=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    hist, bin_edges = np.histogram(x, bins=bins, range=range)
    hist_n = hist * 1.0/hist.max()
    width = bin_edges[1] - bin_edges[0]
    x = np.ravel(zip(bin_edges[:-1], bin_edges[:-1]+width))
    y = np.ravel(zip(hist_n, hist_n))
    if histtype == 'step':
        if orientation == 'vertical':
            plt.plot(x, y, **kwargs)
        elif orientation == 'horizontal':
            plt.plot(y, x, **kwargs)
        else:
            raise ValueError
    elif histtype == 'stepfilled':
        if orientation == 'vertical':
            plt.fill(x, y, **kwargs)
        elif orientation == 'horizontal':
            plt.fill(y, x, **kwargs)
        else:
            raise ValueError
    else:
        raise ValueError


#a = cPickle.load(open('output/galaxy_data.pkl'))
a = cPickle.load(open('output/corrected_galaxy_data.pkl'))

f = plt.figure(figsize=(5,3))
ax = f.add_subplot(111)
hist_with_peak(a.redshift, histtype='step', ax=ax, bins=50, label='CatSim i < 25.3')
ax.set_xlim(0, 4)
ax.set_ylim(0, 1.2)

zs = np.arange(0, 4, 0.1)
n = 5.0*zs**(1.27) * np.exp(-(zs/0.5)**1.02)

ax.plot(zs, n, label='$n_{eff}$ from Chang+13')

ax.set_xlabel('redshift')
ax.set_ylabel('relative number density')

ax.legend(prop={'size':10})
f.tight_layout()

plt.savefig('output/compare_z_dist.png', dpi=220)
