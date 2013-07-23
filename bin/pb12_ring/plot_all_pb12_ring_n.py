import sys

import matplotlib.pyplot as plt

def add_to_plot(infile, ax1, ax2, color, n):
    calib = {'z':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[], 'm_analytic':[], 'c_analytic':[]}
    with open(infile, 'r') as fil:
        for line in fil:
            line = line.replace('(', ' ')
            line = line.replace(')', ' ')
            line = line.replace(',', ' ')
            line = ' '.join(line.split())
            z, c1, c2, m1, m2, m_analytic = line.split(' ')
            calib['z'].append(float(z))
            calib['c1'].append(float(c1))
            calib['c2'].append(float(c2))
            calib['m1'].append(float(m1))
            calib['m2'].append(float(m2))
            calib['m_analytic'].append(float(m_analytic))
            calib['c_analytic'].append(-float(m_analytic)/2.0)

    ax1.scatter(calib['z'], calib['m1'], marker='s', c='None',
                label=n, edgecolor=color)
    ax1.scatter(calib['z'], calib['m2'], marker='x', c=color)
    ax1.plot(calib['z'], calib['m_analytic'], color='cyan')

    ax2.scatter(calib['z'], calib['c1'], marker='s', c='None',
                label=n, edgecolor=color)
    ax2.scatter(calib['z'], calib['c2'], marker='x', c=color)
    ax2.plot(calib['z'], calib['c_analytic'], color='cyan')
    ax2.plot(calib['z'], [0.0] * len(calib['z']), color='cyan')
    ax1.legend(title='Sersic index', fontsize=9)


if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4), dpi=100)
    fig.subplots_adjust(left=0.18, right=0.93)
    ax1.set_ylabel('m')

    ax2.set_ylabel('c')
    ax2.set_xlabel('z')
    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(-0.04, 0.12)
    ax2.set_xlim(0.0, 3.0)
    ax2.set_ylim(-0.04, 0.1)

    color = ['blue', 'green', 'red']
    i=0
    for n in [0.5, 1.0, 4.0]:
        add_to_plot('output/calib_vs_redshift.r.CWW_E_ext.ukg5v.{:3.1f}.z50.dat'.format(n), ax1, ax2,
                    color[i], str(n))
        i += 1
    fig.savefig('output/all_pb12_ring_n.pdf')
