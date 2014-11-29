import os
import sys

import numpy as np
from astropy.io import fits
import re

filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}

def encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed):
    mode = 0
    if atm:
        mode += 1
    if telescope:
        mode += 2
    if sensor:
        mode += 4
    mode_digit = str(mode)
    wave_digits = "{:02d}".format(wavelength / 25)
    filter_digit = filter_number[filter_name]
    seed_digits = "{:03d}".format(seed-1000)
    return mode_digit + wave_digits + filter_digit + seed_digits

def mag(atm, telescope, sensor, wavelength, filter_name, seed, photon_target=None):
    if photon_target is None:
        return 27.0
    filter_num = filter_number[filter_name]
    obshistid = encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed)
    # determine input magnitude
    catfilename = 'catalogs/stargrid_{}'.format(obshistid)
    lines = open(catfilename).readlines()
    for line in lines:
        if re.search('object', line):
            s = line.split()
            input_mag = float(s[4])
    # determine number of drawn photons
    image_file = 'output/lsst_e_{}_f{}_R22_S11_E000.fits.gz'.format(obshistid, filter_num)
    hdulist = fits.open(image_file)
    drawn_photons = hdulist[0].data.sum()/64
    hdulist.close()
    new_mag = input_mag - 2.5 * np.log10(photon_target / drawn_photons)
    print filter_name, str(wavelength), ':', input_mag, '  ->  ', new_mag
    return new_mag

def write_object_outstring(id_, spec, redshift, mag_norm, RA, DEC):
    outstring = 'object {} {} {} {} {} {} {} {} {} {} {} star none none\n'
    outstring = outstring.format(id_, RA, DEC,
                                 mag_norm, spec, redshift,
                                 0.0, 0.0, 0.0, 0.0, 0.0)
    return outstring

def create_instance_catalog(atm, telescope, sensor, wavelength, filter_name, seed,
                            photon_target=None):
    if wavelength == 500.0:
        return
    filter_num = filter_number[filter_name]
    obshistid = encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed)
    print obshistid
    outfilename = 'stargrid_{}'.format(obshistid)
    outstring = '''Unrefracted_RA_deg 0
Unrefracted_Dec_deg 0
Unrefracted_Azimuth 0
Unrefracted_Altitude 89
Slalib_date 1994/7/19/0.298822999997
Opsim_rotskypos 0
Opsim_rottelpos 0
Opsim_moondec -90
Opsim_moonra 180
Opsim_expmjd 49552.3
Opsim_moonalt -90
Opsim_sunalt -90
Opsim_filter {}
Opsim_dist2moon 180.0
Opsim_moonphase 10.0
Opsim_obshistid {}
Opsim_rawseeing 0.67
SIM_SEED     {:d}
SIM_MINSOURCE 1
SIM_TELCONFIG 0
SIM_CAMCONFIG 1
SIM_VISTIME 15.0
SIM_NSNAP 1
'''.format(filter_num, obshistid, seed)

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    redshift = 0.0
    mag_norm = mag(atm, telescope, sensor, wavelength, filter_name, seed, photon_target=photon_target)
    for ichip, chip in enumerate([1, 4, 7]):
        for iRA, RA in enumerate(RAs):
            for iDEC, DEC in enumerate(DECs):
                id_ = ichip*10 + iRA + 0.1 * iDEC
                objRA = RA + (chip-7)*0.235
                outstring += write_object_outstring(id_, 'mono/mono.{}.spec'.format(wavelength),
                                                    redshift, mag_norm, objRA, DEC)
            outstring += '\n'

    if not os.path.exists('catalogs/'):
        os.mkdir('catalogs/')
    f = open('catalogs/' + outfilename, 'w')
    f.write(outstring)
    f.close()

    f = open('catalogs/' + outfilename + '_extra', 'w')
    f.write('''zenith_v 1000.0
raydensity 0.0
saturation 0
blooming 0
cleartracking
clearperturbations
clearclouds
qevariation 0.0
airglowvariation 0
centroidfile 1
''')
    if not atm:
        f.write('''clearturbulence
clearopacity
''')
    if not telescope:
        f.write('''telescopemode 0
''')
    if not sensor:
        f.write('''detectormode 0
''')
    f.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    for s in 'ugrizY':
        parser.add_argument('-'+s, action='store_true')
    parser.add_argument('--photon_target', type=int)
    args = parser.parse_args()

    for sensor in [0, 1]:
        for telescope in [0, 1]:
            for atm in [0, 1]:
                if (atm == 0) and (telescope == 0) and (sensor == 0):
                    continue
                if args.u:
                    for w in np.arange(325, 401, 25):
                        create_instance_catalog(atm, telescope, sensor, w, 'u', 1000,
                                                photon_target=args.photon_target)
                if args.g:
                    for w in np.arange(400, 551, 25):
                        create_instance_catalog(atm, telescope, sensor, w, 'g', 1000,
                                                photon_target=args.photon_target)
                if args.r:
                    for w in np.arange(550, 701, 25):
                        create_instance_catalog(atm, telescope, sensor, w, 'r', 1000,
                                                photon_target=args.photon_target)
                if args.i:
                    for w in np.arange(675, 826, 25):
                        create_instance_catalog(atm, telescope, sensor, w, 'i', 1000,
                                                photon_target=args.photon_target)
                if args.z:
                    for w in np.arange(800, 951, 25):
                        create_instance_catalog(atm, telescope, sensor, w, 'z', 1000,
                                                photon_target=args.photon_target)
                if args.Y:
                    for w in np.arange(900, 1051, 25):
                        create_instance_catalog(atm, telescope, sensor, w, 'Y', 1000,
                                                photon_target=args.photon_target)
