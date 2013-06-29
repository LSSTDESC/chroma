import sys

import matplotlib.pyplot as plt

def main(infile):
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
            calib['c_analytic'].append(float(m_analytic)/2.0)

    fig = plt.figure(figsize=(5.5, 4), dpi=100)
    ax1 = fig.add_subplot(211)
    fig.subplots_adjust(left=0.18, right=0.93)
    ax1.set_ylabel('m')
    ax1.scatter(calib['z'], calib['m1'], marker='s', c='None', edgecolor='red')
    ax1.scatter(calib['z'], calib['m2'], marker='x', c='red')
    ax1.plot(calib['z'], calib['m_analytic'], color='blue')

    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('c')
    ax2.set_xlabel('z')
    ax2.scatter(calib['z'], calib['c1'], marker='s', c='None', edgecolor='red')
    ax2.scatter(calib['z'], calib['c2'], marker='x', c='red')
    ax2.plot(calib['z'], calib['c_analytic'], color='blue')
    ax2.plot(calib['z'], [0.0] * len(calib['z']), color='blue')

    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(-0.05, 0.05)
    ax2.set_xlim(0.0, 3.0)
    ax2.set_ylim(-0.05, 0.05)
    fig.savefig(infile.replace('.dat', '.pdf'))

if __name__ == '__main__':
    for fn in sys.argv[1:]:
        main(fn)
