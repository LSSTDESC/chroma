""" Sanity check for machine learning performance.  Check that photometric redshifts match
input spectroscopic redshifts.
"""

import cPickle

import numpy as np
import matplotlib.pyplot as plt

a = cPickle.load(open('output/corrected_galaxy_data.pkl'))

axes1 = [0.12, 0.16, 0.5, 0.76]
axes2 = [0.7, 0.16, 0.25, 0.76]

f = plt.figure(figsize=(5, 3))
ax = f.add_axes(axes1)
ax.scatter(a.redshift, a.photo_redshift, s=1, alpha=0.5)
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 2.5)
ax.set_xlabel('$z_\mathrm{spec}$')
ax.set_ylabel('$z_\mathrm{phot}$')

data = (a.photo_redshift-a.redshift)/(1.+a.redshift)

ax2 = f.add_axes(axes2)
ax2.hist(data, bins=100, range=(-0.2, 0.2), histtype='stepfilled')
ax2.xaxis.set_ticklabels(['-0.2', '0.0', '0.2'])
ax2.xaxis.set_ticks([-0.2, 0.0, 0.2])
ax2.set_xlim(-0.2, 0.2)
ax2.set_xlabel('$(z_\mathrm{phot} - z_\mathrm{spec})/(1+z_\mathrm{spec})$', fontsize=10)

std = np.std(data)
area = 0.4/100* ((data > -0.2) & (data < 0.2)).sum()
xs = np.arange(-0.2, 0.2, 0.001)
ys = area/np.sqrt(2*np.pi*std**2)*np.exp(-0.5*xs**2/std**2)
ax2.plot(xs, ys, c='k', zorder=10)
std /= 4
ys = area/np.sqrt(2*np.pi*std**2)*np.exp(-0.5*xs**2/std**2)
ax2.plot(xs, ys, c='k', zorder=10)

f.savefig('output/photo_z_performance.pdf')
