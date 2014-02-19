""" Script to make plot of DCR ring test and analytic results, for several values of Sersic index n.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

def add_to_plot(infile, ax, color, n):
    calib = {'z':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[], 'm_analytic':[], 'c_analytic':[]}

    data = np.genfromtxt(infile).T
    z, m1_analytic, m1_ring, m2_analytic, m2_ring, c1_analytic, c1_ring, c2_analytic, c2_ring = data

    ax.scatter(z, m1_ring, marker='s', c='None', label=n, edgecolor=color)
    ax.scatter(z, m2_ring, marker='x', c='None', edgecolor=color)
    ax.plot(z, m1_analytic, color='black')
    ax.plot(z, m2_analytic, color='black')

    ax.legend(title='Sersic index', fontsize=9)


if __name__ == '__main__':
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    ax.set_ylabel('m')
    ax.set_xlabel('z')

    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(-0.01, 0.12)

    color = ['blue', 'green', 'red', 'magenta']
    i=0
    for n in [0.5, 1.0, 2.5, 4.0]:
        add_to_plot('output/ChromaticSeeing_ring_vs_z_n{}.dat'.format(n), ax, color[i], str(n))
        i += 1
    fig.tight_layout()
    fig.savefig('output/ChromaticSeeing_ring_vs_z.png', dpi=300)
