import cPickle
import matplotlib.pyplot as plt

a = cPickle.load(open('output/star_data.pkl'))

f = plt.figure(figsize=(5,3))
ax = f.add_subplot(111)

ax.hist(a.mag.LSST_u - a.magCalc.LSST_u, bins=100, range=(-0.2, 0.05), alpha=0.4,
        histtype='stepfilled', label='u')
ax.hist(a.mag.LSST_g - a.magCalc.LSST_g, bins=100, range=(-0.2, 0.05), alpha=0.4,
        histtype='stepfilled', label='g')
ax.hist(a.mag.LSST_r - a.magCalc.LSST_r, bins=100, range=(-0.2, 0.05), alpha=0.4,
        histtype='stepfilled', label='r')
ax.hist(a.mag.LSST_i - a.magCalc.LSST_i, bins=100, range=(-0.2, 0.05), alpha=0.4,
        histtype='stepfilled', label='i')
ax.hist(a.mag.LSST_z - a.magCalc.LSST_z, bins=100, range=(-0.2, 0.05), alpha=0.4,
        histtype='stepfilled', label='z')
ax.hist(a.mag.LSST_y - a.magCalc.LSST_y, bins=100, range=(-0.2, 0.05), alpha=0.4,
        histtype='stepfilled', label='y')
ax.legend(prop={'size':8})

ax.set_xlim(-0.2, 0.05)
ax.set_xlabel('catalog mag - synphot mag')
ax.set_ylabel('#')
f.tight_layout()
plt.savefig('output/compare_star_mags.png', dpi=300)
