import os

import numpy
import scipy.integrate

def AB(wave, flambda, AB_wave):
    """Returns the AB magnitude at `AB_wave` (nm) of spectrum specified by
    `wave` (in nm), and `flambda` (in erg/s/cm^2/Ang).
    """
    speed_of_light = 2.99792458e18 # units are Angstrom Hz
    fNu = flambda * (wave * 10)**2 / speed_of_light # `* 10` is wave from nm -> Angstrom
    AB_val = -2.5 * numpy.log10(numpy.interp(AB_wave, wave, fNu)) - 48.6
    return AB_val

def integrated_flux(wave, flambda, f_wave, f_throughput, exptime=15.0, eff_diam=670):
    """Integrates product of SED and filter throughput, and multiplies
    by typical LSST exposure time (in seconds) and collecting area
    (specified by effective diameter in cm) to estimate the number of
    photons collected by CCD.

    Units
    -----

    wave, f_wave : nm (corresponds to SED and filter resp.)
    flambda : erg/s/cm^2/Ang
    f_throughput : dimensionless
    exptime : seconds
    eff_diam : cm (effective diameter of aperture)
    """
    wave_union = numpy.union1d(wave, f_wave) #note union1d sorts its output
    flambda_i = numpy.interp(wave_union, wave, flambda)
    throughput_i = numpy.interp(wave_union, f_wave, f_throughput)

    hc = 1.98644521e-9 # (PlanckConstant * speedOfLight) in erg nm
    integrand = flambda_i * throughput_i * wave_union * 10 / hc
    photon_rate = scipy.integrate.trapz(integrand, wave_union)
    return photon_rate * numpy.pi * (eff_diam / 2)**2 * exptime # total photons


def mag_norm_to_LSST_flux(SED_file, filter_file, mag_norm, redshift=0.0):
    """Predict LSST PhoSim flux (in total number of collected photons)
    for an object with SED specified by `SED_file` through a filter
    specified by `filter_file`, and a PhoSim normalization of `mag_norm`.

    The format of the SED_file is 2 columns with first column the
    wavelength in nm, and the second column the flambda flux in
    erg/s/cm2/Ang.

    The format of the filter_file is 2 columns with first column the
    wavelength in nm and the second column the throughput (assumed to
    be everything: sky, filter, CCD, etc.) in fraction of surviving
    photons.
    """
    SED_data = numpy.genfromtxt(SED_file)
    wave, flambda = SED_data[:,0], SED_data[:,1]
    f_data = numpy.genfromtxt(filter_file)
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    AB_val = AB(wave, flambda, 500.0)
    flux = integrated_flux(wave * (1.0 + redshift), flambda / (1.0 + redshift), f_wave, f_throughput)
    return flux * 10**(-0.4 * (mag_norm - AB_val)) * 0.706 #empirical fudge factor!

def encode_obshistid(SED_type, filter_name, zenith, seed, redshift):
    SED_types = {'G5v':'1', 'star':'2', 'gal':'3'}
    SED_digit = SED_types[SED_type]
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    redshift_digits = '{:02d}'.format(int(round((redshift / 0.03))))
    return SED_digit + filter_digit + zenith_digit + seed_digit + redshift_digits

def write_object_outstring(id_, spec, redshift, mag_norm, RA, DEC):
    outstring = 'object {} {} {} {} {} {} {} {} {} {} {} star none none\n'
    outstring = outstring.format(id_, RA, DEC,
                                 mag_norm, spec, redshift,
                                 0.0, 0.0, 0.0, 0.0, 0.0)
    return outstring

def create_instance_catalog(SED_type, filter_name, zenith, seed, redshift):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_num = filter_number[filter_name]
    obshistid = encode_obshistid(SED_type, filter_name, zenith, seed, redshift)
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
    #load filter
    filter_dir = '../../data/filters/'
    filter_file = filter_dir + 'LSST_{}.dat'.format(filter_name)
    #load spectra
    spec_dir = '../../data/SEDs/'
    if SED_type == 'gal':
        spectra = ['CWW_E_ext.ascii',
                   'KIN_Sa_ext.ascii',
                   'KIN_Sb_ext.ascii',
                   'CWW_Sbc_ext.ascii',
                   'CWW_Scd_ext.ascii',
                   'CWW_Im_ext.ascii',
                   'KIN_SB1_ext.ascii',
                   'KIN_SB6_ext.ascii']
    elif SED_type == 'star':
        spectra = ['uko5v.ascii',
                   'ukb5iii.ascii',
                   'uka5v.ascii',
                   'ukf5v.ascii',
                   'ukg5v.ascii',
                   'ukk5v.ascii',
                   'ukm5v.ascii',
                   'ukg5v.ascii'] #extra G5v star to make 8
    elif SED_type == 'G5v':
        spectra = ['ukg5v.ascii']*8

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    for iRA, RA in enumerate(RAs):
        spec = spectra[iRA]
        mag_try = 20.0
        photons = mag_norm_to_LSST_flux(spec_dir + spec, filter_file, mag_try, redshift=redshift)
        mag_norm = mag_try - 2.5 * numpy.log10(1e6 / photons)
        for iDEC, DEC in enumerate(DECs):
            id_ = iRA + 0.1 * iDEC
            outstring += write_object_outstring(id_, 'PB12/' + spec, redshift, mag_norm, RA, DEC)
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
    spec_dir = '../../data/SEDs/'
    filter_dir = '../../data/filters/'
    spec_file = spec_dir + 'CWW_E_ext.ascii'
    filter_file = filter_dir + 'LSST_r.dat'
    mag_try = 20.0
    redshift = 0.03
    photons = mag_norm_to_LSST_flux(spec_file, filter_file, mag_try, redshift=redshift)
    mag_norm = mag_try - 2.5 * numpy.log10(1e6 / photons)
    print mag_norm
