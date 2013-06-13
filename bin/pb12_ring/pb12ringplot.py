#plots results of m_vs_redshift

import sys

import matplotlib.pyplot as plt

def main(argv):
    if len(argv) > 1:
        infile = argv[1]
    else:
        infile = 'output/m_vs_redshift.dat'
    calib = {'z':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[], 'm_analytic':[], 'c_analytic':[]}
    try:
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
    except IOError:
        pass

    fig = plt.figure(figsize=(5.5, 4), dpi=100)
    ax1 = fig.add_subplot(211)
    fig.subplots_adjust(left=0.18, right=0.93)
    ax1.set_ylabel('m')
    ax1.set_ylim(-0.05, 0.1)
    ax1.plot(calib['z'], calib['m1'], 's', ms=5, mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['z'], calib['m1'], color='red')
    ax1.plot(calib['z'], calib['m2'], 'x', ms=5, mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['z'], calib['m2'], color='red')
    ax1.plot(calib['z'], calib['m_analytic'], color='blue')

    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('c')
    ax2.set_xlabel('z')
    ax2.set_ylim(-0.05, 0.1)
    ax2.plot(calib['z'], calib['c1'], 's', ms=5, mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['z'], calib['c1'], color='red')
    ax2.plot(calib['z'], calib['c2'], 'x', ms=5, mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['z'], calib['c2'], color='red')
    ax2.plot(calib['z'], calib['c_analytic'], color='blue')
    ax2.plot(calib['z'], [0.0] * len(calib['z']), color='blue')

    fig.savefig('output/m_vs_redshift.pdf')

if __name__ == '__main__':
    main(sys.argv)
