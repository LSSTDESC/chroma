import numpy
import pyfits
import matplotlib.pyplot as plt

def plot_dists():
    simard_dir = '../../data/simard/'
    bulge_phot = pyfits.getdata(simard_dir+'table3e.fits')
    disk_phot = pyfits.getdata(simard_dir+'table3f.fits')

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('bulge $r_{eff} (kpc)$')
    ax.set_ylabel('disk $r_{eff} (kpc)$')
    b_r_e = bulge_phot['re814']
    d_r_e = disk_phot['rd814'] * numpy.log(2)
    ax.scatter(b_r_e, d_r_e)
    plt.savefig('output/b_r_e_vs_d_r_e.pdf')

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('B/T')
    ax.set_ylabel('number')
    b_flux = 10**(-0.4 * bulge_phot['V606AB'])
    d_flux = 10**(-0.4 * disk_phot['V606AB'])
    ax.hist(b_flux/(b_flux+d_flux),40)
    plt.savefig('output/BT_hist.pdf')

if __name__ == '__main__':
    plot_dists()
