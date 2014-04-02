import matplotlib.pyplot as plt
import numpy as np

# DCR only plot
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)
plt.setp( ax1.get_xticklabels(), visible=False)

ax1.text(0.1, 0.15, 'DCR only', fontsize=14)

ax1.set_ylabel('m')
ax2.set_ylabel('c')
ax2.set_xlabel('redshift')

ax1.set_xlim(0.0, 2.0)
ax1.set_ylim(-0.01, 0.2)

ax2.set_xlim(0.0, 2.0)
ax2.set_ylim(-0.01, 0.1)

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/GG_DCR.dat').T

ax1.plot(z, m1a, color='k', label='analytic')
ax1.plot(z, m2a, color='k')
ax2.plot(z, c1a, color='k', label='analytic')
ax2.plot(z, c2a, color='k')

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='blue', label='Gaussian gal, Gaussian PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='blue')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='blue', label='Gaussian gal, Gaussian PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='blue')


z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/SG_DCR.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='red', label='DeV gal, Gaussian PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='red')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='red', label='DeV gal, Gaussian PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='red')

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/GM_DCR.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='magenta', label='Gaussian gal, Moffat PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='magenta')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='magenta', label='Gaussian gal, Moffat PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='magenta')

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/SM_DCR.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='green', label='DeV gal, Moffat PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='green')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='green', label='DeV gal, Moffat PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='green')

ax1.legend(fontsize=9)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig('output/DCR.png', dpi=220)

# Chromatic seeing only plot
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)
plt.setp( ax1.get_xticklabels(), visible=False)

ax1.text(0.1, 0.15, 'Chromatic seeing only', fontsize=14)

ax1.set_ylabel('m')
ax2.set_ylabel('c')
ax2.set_xlabel('redshift')

ax1.set_xlim(0.0, 2.0)
ax1.set_ylim(-0.01, 0.2)

ax2.set_xlim(0.0, 2.0)
ax2.set_ylim(-0.01, 0.1)

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/GG_CS.dat').T

ax1.plot(z, m1a, color='k', label='analytic')
ax1.plot(z, m2a, color='k')
ax2.plot(z, c1a, color='k', label='analytic')
ax2.plot(z, c2a, color='k')

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='blue', label='Gaussian gal, Gaussian PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='blue')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='blue', label='Gaussian gal, Gaussian PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='blue')


z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/SG_CS.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='red', label='DeV gal, Gaussian PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='red')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='red', label='DeV gal, Gaussian PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='red')

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/GM_CS.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='magenta', label='Gaussian gal, Moffat PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='magenta')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='magenta', label='Gaussian gal, Moffat PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='magenta')

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/SM_CS.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='green', label='DeV gal, Moffat PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='green')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='green', label='DeV gal, Moffat PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='green')

ax1.legend(fontsize=9)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig('output/CS.png', dpi=220)

# both DCR and chromatic seeing plot
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)
plt.setp( ax1.get_xticklabels(), visible=False)

ax1.text(0.1, 0.15, 'DCR and chromatic seeing', fontsize=14)

ax1.set_ylabel('m')
ax2.set_ylabel('c')
ax2.set_xlabel('redshift')

ax1.set_xlim(0.0, 2.0)
ax1.set_ylim(-0.01, 0.2)

ax2.set_xlim(0.0, 2.0)
ax2.set_ylim(-0.01, 0.1)

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/GG_both.dat').T

ax1.plot(z, m1a, color='k', label='analytic')
ax1.plot(z, m2a, color='k')
ax2.plot(z, c1a, color='k', label='analytic')
ax2.plot(z, c2a, color='k')

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='blue', label='Gaussian gal, Gaussian PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='blue')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='blue', label='Gaussian gal, Gaussian PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='blue')


z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/SG_both.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='red', label='DeV gal, Gaussian PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='red')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='red', label='DeV gal, Gaussian PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='red')

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/GM_both.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='magenta', label='Gaussian gal, Moffat PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='magenta')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='magenta', label='Gaussian gal, Moffat PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='magenta')

z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt('output/SM_both.dat').T

ax1.scatter(z, m1r, color='None', marker='x', edgecolor='green', label='DeV gal, Moffat PSF')
ax1.scatter(z, m2r, color='None', marker='s', edgecolor='green')
ax2.scatter(z, c1r, color='None', marker='x', edgecolor='green', label='DeV gal, Moffat PSF')
ax2.scatter(z, c2r, color='None', marker='s', edgecolor='green')

ax1.legend(fontsize=9)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig('output/both.png', dpi=220)
