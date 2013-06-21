import os

import numpy

def write_object_outstring(id_, spec, redshift, mag_norm, RA, DEC):
    outstring = 'object {} {} {} {} {} {} {} {} {} {} {} star none none\n'
    outstring = outstring.format(id_, RA, DEC,
                                 mag_norm, spec, redshift,
                                 0.0, 0.0, 0.0, 0.0, 0.0)
    return outstring

def encode_obshistid(mono_wave, filter_name, zenith, seed):
    #first four digits are wavelength in nm
    wave_digits = str(int(round(mono_wave)))
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return wave_digits + filter_digit + zenith_digit + seed_digit

def create_instance_catalog(mono_wave, filter_name, zenith, seed):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_num = filter_number[filter_name]
    obshistid = encode_obshistid(mono_wave, filter_name, zenith, seed)
    outfilename = 'stargrid_{}'.format(obshistid)
    outstring = '''Unrefracted_RA_deg 0
Unrefracted_Dec_deg 0
Unrefracted_Azimuth 0
Unrefracted_Altitude {:d}
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
'''.format(int(round(90 - zenith)), filter_num, obshistid, seed)

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    for iRA, RA in enumerate(RAs):
        for iDEC, DEC in enumerate(DECs):
            id_ = iRA + 0.1 * iDEC
            outstring += write_object_outstring(id_, 'mono/mono.{}.spec'.format(mono_wave),
                                                0.0, 25.0, RA, DEC)
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
    f.close()

if __name__ == '__main__':
    for w in numpy.arange(300, 401, 25):
        create_instance_catalog(w, 'u', 0, 1000)
    for w in numpy.arange(400, 551, 25):
        create_instance_catalog(w, 'g', 0, 1000)
    for w in numpy.arange(550, 701, 25):
        create_instance_catalog(w, 'r', 0, 1000)
    for w in numpy.arange(675, 826, 25):
        create_instance_catalog(w, 'i', 0, 1000)
    for w in numpy.arange(800, 951, 25):
        create_instance_catalog(w, 'z', 0, 1000)
    for w in numpy.arange(900, 1101, 25):
        create_instance_catalog(w, 'Y', 0, 1000)
