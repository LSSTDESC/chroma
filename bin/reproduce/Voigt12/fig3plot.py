import numpy
import matplotlib.pyplot as plt

def fig3plot():
    #setup plots
    fig = plt.figure(figsize=(5.5,7), dpi=100)
    fig.subplots_adjust(left=0.18)
    ax1 = fig.add_subplot(211)
    ax1.set_yscale('log')
    ax1.set_ylabel('|m|')
    ax1.set_ylim(5.e-5, 2.e-2)
    ax1.set_xlim(150, 450)
    ax1.fill_between([150,450], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([150,450], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([150,450], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax2 = fig.add_subplot(212)
    ax2.set_yscale('log')
    ax2.set_xlabel('Filter width (nm)')
    ax2.set_ylabel('|c|')
    ax2.set_ylim(1.5e-5, 1.e-3)
    ax2.set_xlim(150, 450)
    ax2.fill_between([150,450], [3.e-4, 3.e-4], [1.5e-5, 1.5e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([150,450], [3.e-4/2, 3.e-4/2], [1.5e-5, 1.5e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([150,450], [3.e-4/5, 3.e-4/5], [1.5e-5, 1.5e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    #load fiducial galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_fiducial.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='red')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='red')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='red')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='red')


    #load varied redshift galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_redshift.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='blue', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='blue', linestyle='--')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='blue', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='blue', linestyle='--')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='blue', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='blue', linestyle='--')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='blue', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='blue', linestyle='--')


    #load varied bulge radius galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_bulge_radius.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='green', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='green', linestyle='-.')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='green', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='green', linestyle='-.')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='green', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='green', linestyle='-.')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='green', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='green', linestyle='-.')


    #load varied disk spectrum galaxy

    calib = {'width':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig3_disk_spectrum.dat', 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                width, c1, c2, m1, m2 = line.split(' ')
                calib['width'].append(int(width))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='black', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m1'])), color='black', linestyle=':')
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='black', mew=1.3)
    ax1.plot(calib['width'], abs(numpy.array(calib['m2'])), color='black', linestyle=':')
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), 's', mfc='None', mec='black', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c1'])), color='black', linestyle=':')
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), 'x', mfc='None', mec='black', mew=1.3)
    ax2.plot(calib['width'], abs(numpy.array(calib['c2'])), color='black', linestyle=':')


    plt.savefig('output/fig3.pdf')

if __name__ == '__main__':
    fig3plot()
