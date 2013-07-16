import os
import subprocess

import numpy

def encode_obshistid(mode, mono_wave, filter_name, zenith, seed):
    mode_digits = str(mode)
    wave_digits = '{:04d}'.format(int(round(mono_wave)))
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = '{:03d}'.format(seed - 1000)
    return mode_digits + wave_digits + filter_digit + zenith_digit + seed_digit

def submit_job(mode, mono_wave, filter_name, zenith, seed, test=False):
    if mode > 2:
        print 'invalid mode'
        sys.exit()

    obshistid = encode_obshistid(mode, mono_wave, filter_name, zenith, seed)
    phosim_dir = '/nfs/slac/g/ki/ki19/jmeyers3/phosim-3.2.9/'
    sub_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/seeing_vs_lambda_phosim/'
    cat_dir = sub_dir + 'catalogs/'
    cat_file = cat_dir + 'stargrid_' + obshistid
    extra_file = cat_file + '_extra'
    if not os.path.exists('output/'):
        os.mkdir('output/')
    if not os.path.exists('stdout/'):
        os.mkdir('stdout/')
    out_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/seeing_vs_lambda_phosim/output/'
    command = '"cd {} && ./phosim {} -c {} -e0 -sR22_S11 -o {}"'.format(phosim_dir,
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
    # trying different wavelengths, and w/ & w/o telescopemode 0
    for w in numpy.arange(300, 401, 25):
        submit_job(1, w, 'u', 0, 1000)
        submit_job(2, w, 'u', 0, 1000)
    for w in numpy.arange(400, 551, 25):
        submit_job(1, w, 'g', 0, 1000)
        submit_job(2, w, 'g', 0, 1000)
    for w in numpy.arange(550, 701, 25):
        submit_job(1, w, 'r', 0, 1000)
        submit_job(2, w, 'r', 0, 1000)
    for w in numpy.arange(675, 826, 25):
        submit_job(1, w, 'i', 0, 1000)
        submit_job(2, w, 'i', 0, 1000)
    for w in numpy.arange(800, 951, 25):
        submit_job(1, w, 'z', 0, 1000)
        submit_job(2, w, 'z', 0, 1000)
    for w in numpy.arange(900, 1101, 25):
        submit_job(1, w, 'Y', 0, 1000)
        submit_job(2, w, 'Y', 0, 1000)
    # try a few different seeds for a specific wavelength
    for sd in numpy.arange(1001, 1100):
        submit_job(1, 600, 'r', 0, sd)
