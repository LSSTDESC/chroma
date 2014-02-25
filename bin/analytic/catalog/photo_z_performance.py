""" Sanity check for machine learning performance.  Check that photometric redshifts match
input spectroscopic redshifts.
"""


import cPickle

import matplotlib.pyplot as plt

a = cPickle.load(open('output/corrected_galaxy_data.pkl'))

axes1 = [0.12, 0.16, 0.5, 0.76]
axes2 = [0.7, 0.16, 0.25, 0.76]

f = plt.figure(figsize=(5, 3))
ax = f.add_axes(axes1)
ax.scatter(a.redshift, a.photo_redshift, s=1)
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel('$z_\mathrm{spec}$')
ax.set_ylabel('$z_\mathrm{phot}$')

ax2 = f.add_axes(axes2)
ax2.hist(a.photo_redshift-a.redshift, bins=50, range=(-0.2, 0.2), histtype='stepfilled')
ax2.xaxis.set_ticklabels(['-0.2', '0.0', '0.2'])
ax2.xaxis.set_ticks([-0.2, 0.0, 0.2])
ax2.set_xlim(-0.2, 0.2)
ax2.set_xlabel('$z_\mathrm{phot} - z_\mathrm{spec}$')

plt.savefig('output/photo_z_performance.png', dpi=300)
