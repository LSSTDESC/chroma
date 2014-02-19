""" Script to make plot of DCR ring test and analytic results, for several values of Sersic index n.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

def add_to_plot(infile, ax1, ax2, color, n):
    calib = {'z':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[], 'm_analytic':[], 'c_analytic':[]}

    data = np.genfromtxt(infile).T
    z, m1_analytic, m1_ring, m2_analytic, m2_ring, c1_analytic, c1_ring, c2_analytic, c2_ring = data

    ax1.scatter(z, m1_ring, marker='s', c='None', label=n, edgecolor=color)
    ax1.scatter(z, m2_ring, marker='x', c='None', edgecolor=color)
    ax1.plot(z, m1_analytic, color='black')
    ax1.plot(z, m2_analytic, color='black')

    ax2.scatter(z, c1_ring, marker='s', c='None', label=n, edgecolor=color)
    ax2.scatter(z, c2_ring, marker='x', c='None', edgecolor=color)
    ax2.plot(z, c1_analytic, color='black')
    ax2.plot(z, c2_analytic, color='black')

    ax1.legend(title='Sersic index', fontsize=9)


if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

    ax1.set_ylabel('m')
    ax2.set_ylabel('c')
    ax2.set_xlabel('z')

    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(-0.01, 0.08)

    ax2.set_xlim(0.0, 3.0)
    ax2.set_ylim(-0.01, 0.06)

    color = ['blue', 'green', 'red', 'magenta']
    i=0
    for n in [0.5, 1.0, 2.5, 4.0]:
        add_to_plot('output/DCR_ring_vs_z_n{}.dat'.format(n), ax1, ax2,
                    color[i], str(n))
        i += 1
    fig.tight_layout()
    fig.savefig('output/DCR_ring_vs_z.png', dpi=300)
