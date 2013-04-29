import numpy
import matplotlib.pyplot as plt

def fig4plot():
    #setup plots
    fig = plt.figure(figsize=(10.0, 7.5), dpi=60)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
    ax1 = fig.add_subplot(221)
    ax1.set_yscale('log')
    ax1.set_ylabel('|m|')
    ax1.set_ylim(5.e-5, 1.e-2)
    ax1.set_xlabel('n$_{\mathrm{s, b}}$')
    ax1.set_xlim(1.5, 4.0)

    ax2 = fig.add_subplot(222)
    ax2.set_yscale('log')
    ax2.set_ylabel('|m|')
    ax2.set_ylim(5.e-5, 1.e-2)
    ax2.set_xlabel('B/T')
    ax2.set_xlim(0.0, 1.0)

    ax3 = fig.add_subplot(223)
    ax3.set_yscale('log')
    ax3.set_ylabel('|m|')
    ax3.set_ylim(5.e-5, 1.e-2)
    ax3.set_xlabel('e$_{\mathrm{g}}$')
    ax3.set_xlim(0.1, 0.6)

    ax4 = fig.add_subplot(224)
    ax4.set_yscale('log')
    ax4.set_ylabel('|m|')
    ax4.set_ylim(5.e-5, 1.e-2)
    ax4.set_xlabel('y$_0$')
    ax4.set_xlim(0.0, 0.5)

    ax1.fill_between([1.5, 4.0], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([1.5, 4.0], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax1.fill_between([1.5, 4.0], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax2.fill_between([0.0, 1.0], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([0.0, 1.0], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax2.fill_between([0.0, 1.0], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax3.fill_between([0.1, 0.6], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax3.fill_between([0.1, 0.6], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax3.fill_between([0.1, 0.6], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    ax4.fill_between([0.0, 0.5], [1.e-3, 1.e-3], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax4.fill_between([0.0, 0.5], [1.e-3/2, 1.e-3/2], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')
    ax4.fill_between([0.0, 0.5], [1.e-3/5, 1.e-3/5], [1.e-5, 1.e-5],
                     color='grey', alpha=0.2, edgecolor='None')

    # load bulge sersic index data

    calib = {'bulge_n':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_bulge_sersic_index.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                bulge_n, c1, c2, m1, m2 = line.split(' ')
                calib['bulge_n'].append(float(bulge_n))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax1.plot(calib['bulge_n'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['bulge_n'], abs(numpy.array(calib['m1'])), color='red')
    ax1.plot(calib['bulge_n'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax1.plot(calib['bulge_n'], abs(numpy.array(calib['m2'])), color='red', ls='--')

    # load bulge flux data

    calib = {'bulge_flux':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_bulge_flux.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                bulge_flux, c1, c2, m1, m2 = line.split(' ')
                calib['bulge_flux'].append(float(bulge_flux))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax2.plot(calib['bulge_flux'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['bulge_flux'], abs(numpy.array(calib['m1'])), color='red')
    ax2.plot(calib['bulge_flux'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax2.plot(calib['bulge_flux'], abs(numpy.array(calib['m2'])), color='red', ls='--')

    # load galaxy ellipticity data

    calib = {'gal_ellip':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_gal_ellip.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                gal_ellip, c1, c2, m1, m2 = line.split(' ')
                calib['gal_ellip'].append(float(gal_ellip))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax3.plot(calib['gal_ellip'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax3.plot(calib['gal_ellip'], abs(numpy.array(calib['m1'])), color='red')
    ax3.plot(calib['gal_ellip'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax3.plot(calib['gal_ellip'], abs(numpy.array(calib['m2'])), color='red', ls='--')

    # load y0 data

    calib = {'y0':[], 'c1':[], 'c2':[], 'm1':[], 'm2':[]}
    try:
        with open('output/fig4_y0.dat') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = ' '.join(line.split())
                y0, c1, c2, m1, m2 = line.split(' ')
                calib['y0'].append(float(y0))
                calib['c1'].append(float(c1))
                calib['c2'].append(float(c2))
                calib['m1'].append(float(m1))
                calib['m2'].append(float(m2))
    except IOError:
        pass

    ax4.plot(calib['y0'], abs(numpy.array(calib['m1'])), 's', mfc='None', mec='red', mew=1.3)
    ax4.plot(calib['y0'], abs(numpy.array(calib['m1'])), color='red')
    ax4.plot(calib['y0'], abs(numpy.array(calib['m2'])), 'x', mfc='None', mec='red', mew=1.3)
    ax4.plot(calib['y0'], abs(numpy.array(calib['m2'])), color='red', ls='--')

    plt.savefig('output/fig4.pdf')

if __name__ == '__main__':
    fig4plot()
