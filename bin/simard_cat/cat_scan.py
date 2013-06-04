import sys

import pyfits
import linear_spec

def cat_scan():
    simard_dir = '../../data/simard/'
    bulge_phot = pyfits.getdata(simard_dir+'table3e.fits')
    disk_phot = pyfits.getdata(simard_dir+'table3f.fits')

    count = 0
    for b, d in zip(bulge_phot, disk_phot):
        if b['DEEP-GSS'] != d['DEEP-GSS']:
            print 'ERROR ERROR ERROR'
            sys.exit()
        b_m, b_c = linear_spec.linear_spec(b['V606AB'], b['I814AB'])
        d_m, d_c = linear_spec.linear_spec(d['V606AB'], d['I814AB'])

        V_L = 470.0
        V_U = 750.0

        I_L = 690.0
        I_U = 1000.0

        if (b_m * V_L + b_c < 0) or (b_m * I_U + b_c < 0):
            continue
        if (d_m * V_L + d_c < 0) or (d_m * I_U + d_c < 0):
            continue
        count += 1
        print b['DEEP-GSS'], count

if __name__ == '__main__':
    cat_scan()
