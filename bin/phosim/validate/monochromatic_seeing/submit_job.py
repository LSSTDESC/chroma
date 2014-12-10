import os
import subprocess

import numpy as np

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

def submit_job(atm, telescope, sensor, wavelength, filter_name, seed, test=False):
    if wavelength == 500.0:
        return
    obshistid = encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed)
    phosim_dir = '/nfs/slac/g/ki/ki19/jmeyers3/phosim-3.4.2/'
    sub_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/phosim/validate/monochromatic_seeing/'
    cat_dir = sub_dir + 'catalogs/'
    cat_file = cat_dir + 'stargrid_' + obshistid
    extra_file = cat_file + '_extra'
    if not os.path.exists('output/'):
        os.mkdir('output/')
    if not os.path.exists('stdout/'):
        os.mkdir('stdout/')
    out_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/phosim/validate/monochromatic_seeing/output/'
    command = '"cd {} && python phosim.py {} -c {} -e0 -o {}"'.format(phosim_dir,
                                                                      cat_file,
                                                                      extra_file,
                                                                      out_dir)
    stdout_file = sub_dir + 'stdout/' + obshistid
    job_name = obshistid
    full_command = 'bsub -q long -oo {} -J {} {}'.format(stdout_file, job_name, command)
    print full_command
    if not test:
        subprocess.call(full_command, shell=True)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    for s in 'ugrizY':
        parser.add_argument('-'+s, action='store_true')
    args = parser.parse_args()

    for sensor in [0, 1]:
        for telescope in [0, 1]:
            for atm in [0, 1]:
                if (atm == 0) and (telescope == 0) and (sensor == 0):
                    continue
                if args.u:
                    for w in np.arange(325, 401, 25):
                        submit_job(atm, telescope, sensor, w, 'u', 1000)
                if args.g:
                    for w in np.arange(400, 551, 25):
                        submit_job(atm, telescope, sensor, w, 'g', 1000)
                if args.r:
                    for w in np.arange(550, 701, 25):
                        submit_job(atm, telescope, sensor, w, 'r', 1000)
                if args.i:
                    for w in np.arange(675, 826, 25):
                        submit_job(atm, telescope, sensor, w, 'i', 1000)
                if args.z:
                    for w in np.arange(800, 951, 25):
                        submit_job(atm, telescope, sensor, w, 'z', 1000)
                if args.Y:
                    for w in np.arange(900, 1051, 25):
                        submit_job(atm, telescope, sensor, w, 'Y', 1000)
